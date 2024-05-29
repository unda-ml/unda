macro_rules! create_test {
    ($name:ident, $op:ident, $dtype:ident, $in1:expr, $in2:expr, $exp:expr) => {
        #[test]
        fn $name() {
            let mut ctx = Context::new();
            let x = ctx.parameter("x", [], xla::ElementType::$dtype).expect("x");
            let y = ctx.parameter("y", [], xla::ElementType::$dtype).expect("y");

            let operation = ctx.$op(x, y).expect("operation");

            let client = xla::PjRtClient::gpu(0.7, false).expect("client");
            let name = "test";
            let executable = ctx.compile(&name, [operation], &client).expect("executable");

            let x_input = xla::Literal::scalar($in1);
            let y_input = xla::Literal::scalar($in2);

            let device_result = executable.execute(&[x_input, y_input]).expect("execute");
            let host_result = device_result[0][0]
                .to_literal_sync()
                .expect("to_literal_sync");
            let untupled_result = host_result.to_tuple1().expect("untuple");
            let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
            println!("{:?}", rust_result);

            assert_eq!(rust_result[0], $exp);
        }
    };
    ($name:ident, $op:ident, $dtype:ident, $in:expr, $exp:expr) => {
        #[test]
        fn $name() {
            let mut ctx = Context::new();
            let x = ctx.parameter("x", [], xla::ElementType::$dtype).expect("x");

            let operation = ctx.$op(x).expect("operation");

            let client = xla::PjRtClient::gpu(0.7, false).expect("client");
            let name = "test";
            let executable = ctx.compile(&name, [operation], &client).expect("executable");

            let x_input = xla::Literal::scalar($in);

            let device_result = executable.execute(&[x_input]).expect("execute");
            let host_result = device_result[0][0]
                .to_literal_sync()
                .expect("to_literal_sync");
            let untupled_result = host_result.to_tuple1().expect("untuple");
            let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
            println!("{:?}", rust_result);

            assert_eq!(rust_result[0], $exp);
        }
    };

}

#[cfg(test)]
mod tests {
    use crate::graph::{callsite::callsite, ConstantBinding, Context, Node, Operation};
    use xla::{FromRawBytes, Literal, Shape};

    create_test!(test_pow_f32_100_squared, pow, F32, 10f32, 2f32, 100f32);
    create_test!(test_pow_f32_3_squared, pow, F32, 3f32, 2f32, 9f32);
    create_test!(test_ln_10, log, F32, 10f32, f32::ln(10f32));
    create_test!(test_ln_e, log, F32, 1f32, 0f32);
    create_test!(test_add_1_2, add, F32, 1f32, 2f32, 3f32);
    create_test!(test_sub_1_2, sub, F32, 1f32, 2f32, -1f32);


    /*#[test]
    fn test_inv_perm_transpose() {
        let before = &[1,0];
        let after = Context::inv_perm(before);

        assert_eq!(&[0,1], after.as_slice());
    }*/

    #[test]
    fn test_inv_perm_cplx() {
        let before = &[4,8,0,7,1,5,3,6,2];
        let after = Context::inv_perm(before);

        assert_eq!(&[2,4,8,6,0,5,7,3,1], after.as_slice())
    }

    #[test]
    fn test_inv_perm() {
        let before = &[1,2,0,3];
        let after = Context::inv_perm(before);

        assert_eq!(&[2,0,1,3], after.as_slice());
    }

    #[test]
    fn test_no_const_fold() {
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");

        let add = ctx.add(x, y).expect("x + y");

        //Check that const fold did NOT do it's thing
        assert!(!ctx.fold_consts(add, usize::MAX).expect("Add fold"));
    }

    #[test]
    fn deep_const_fold() {
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let zero = ctx.scalar(0, xla::ElementType::F32).expect("0");

        let const_sum = ctx.add(x, zero).expect("x + 0");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");

        let x_y_mul = ctx.mul(const_sum, y).expect("(x + 0) * y");
        assert!(ctx.fold_consts(x_y_mul, usize::MAX).expect("deep fold"));
    }

    #[test]
    fn test_const_fold_compiles() {
        let mut ctx = Context::new();
        let five = ctx.scalar(5, xla::ElementType::F32).expect("5");
        let zero = ctx.scalar(0, xla::ElementType::F32).expect("0");
        let ten = ctx.scalar(10, xla::ElementType::F32).expect("10");

        let const_sum = ctx.add(five, zero).expect("x + 0");

        let five_ten_add = ctx.add(const_sum, ten).expect("(x + 0) + y");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [five_ten_add], &client)
            .expect("executable");

        // args are just provided in the order they are defined, would be nice to pass a dict or something
        // a pjrtbuffer is just an array slice on some device
        // but im not sure why its a nested vector instead of just one vector
        let device_result = executable.execute::<Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");

        assert_eq!(rust_result[0], 15f32);
    }

    #[test]
    fn test_const_fold_compiles_params() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");
        let zero = ctx.scalar(0, xla::ElementType::F32).expect("zero");

        let sum = ctx.add(x, zero).expect("x + 0");
        let x_y_product = ctx.mul(sum, y).expect("(x + 0) * y");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [x_y_product], &client)
            .expect("executable");

        let x_in = Literal::scalar(10f32);
        let y_in = Literal::scalar(10f32);

        let device_result = executable.execute(&[x_in, y_in]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");

        assert_eq!(rust_result[0], 100f32);
    }

    #[test]
    fn test_mul_by_zero_folds() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");
        let zero = ctx.scalar(0, xla::ElementType::F32).expect("zero");

        let mul = ctx.mul(x, zero).expect("x * 0");
        let x_y_product = ctx.mul(mul, y).expect("(x * 0) * y");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [x_y_product], &client)
            .expect("executable");

        let x_in = Literal::scalar(10f32);
        let y_in = Literal::scalar(10f32);

        let device_result = executable.execute(&[x_in, y_in]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");

        assert_eq!(rust_result[0], 0f32);
    }

    #[test]
    fn test_mul_folds() {
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let one = ctx.scalar(1, xla::ElementType::F32).expect("1");

        let const_sum = ctx.mul(x, one).expect("x * 1");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");

        let x_y_mul = ctx.mul(const_sum, y).expect("(x * 1) * y");
        assert!(ctx.fold_consts(x_y_mul, usize::MAX).expect("deep fold"));
    }

    #[test]
    fn test_mul_compiles() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");
        let one = ctx.scalar(1, xla::ElementType::F32).expect("1");

        let mul = ctx.mul(x, one).expect("x * 1");
        let x_y_product = ctx.mul(mul, y).expect("(x * 0) * y");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [x_y_product], &client)
            .expect("executable");

        let x_in = Literal::scalar(5.5f32);
        let y_in = Literal::scalar(10f32);

        let device_result = executable.execute(&[x_in, y_in]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");

        assert_eq!(rust_result[0], 55f32);
    }

    #[test]
    fn ensure_is_zero_scalar() {
        let mut ctx = Context::new();
        let zeroes = ctx.scalar(0, xla::ElementType::F32).expect("zero scalar");
        let node = ctx.nodes.get(zeroes).expect("node of zero");

        assert!(node.is_zero().expect("is zero"));
    }

    #[test]
    fn ensure_is_zero_vector() {
        let mut ctx = Context::new();
        let zeroes = ctx
            .vector([0.0, 0.0, 0.0, 0.0], xla::ElementType::F64)
            .expect("zero vector");
        let node = ctx.nodes.get(zeroes).expect("node of zeroes");

        assert!(node.is_zero().expect("is zero"));
    }

    #[test]
    fn ensure_is_zero_unique_types() {
        let mut ctx = Context::new();
        let zeroes = ctx
            .matrix(
                [[0u64, 0u64], [0u64, 0u64], [0u64, 0u64]],
                xla::ElementType::U64,
            )
            .expect("zero matrix u64");
        let node = ctx.nodes.get(zeroes).expect("node of zeroes");

        assert!(node.is_zero().expect("is zero"));
    }

    #[test]
    fn ensure_is_const() {
        let mut ctx = Context::new();
        let scalar_const = ctx.scalar(15, xla::ElementType::F32).expect("fifteen");
        let node = ctx.nodes.get(scalar_const).expect("node of 15");

        assert!(node.is_const().is_some());
    }


    #[test]
    fn test_exp() {
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let exp = ctx.exp(x).expect("e ^ x");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [exp], &client).expect("executable");

        let x_input = xla::Literal::scalar(1f32);
        // args are just provided in the order they are defined, would be nice to pass a dict or something
        // a pjrtbuffer is just an array slice on some device
        // but im not sure why its a nested vector instead of just one vector
        let device_result = executable.execute(&[x_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], f32::exp(1f32));

    }

    #[test]
    fn test_pow() {
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");

        let pow = ctx.pow(x, y).expect("x ^ y");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [pow], &client).expect("executable");

        let x_input = xla::Literal::scalar(3f32);
        let y_input = xla::Literal::scalar(2f32);

        let device_result = executable.execute(&[x_input, y_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], 9f32);

    }

    #[test]
    fn test_log(){
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let log = ctx.log(x).expect("lnx");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [log], &client).expect("executable");

        let x_input = xla::Literal::scalar(f32::exp(2f32));

        let device_result = executable.execute(&[x_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], 2f32);
    }

    #[test]
    fn test_mul_add_scalar_consts_and_params() {
        let mut ctx = Context::new();

        let three = ctx.scalar(3, xla::ElementType::F32).expect("three");

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");

        let product = ctx.mul(x, three).expect("product");
        let sum = ctx.add(product, y).expect("sum");

        // output XLA
        // client must be exposed to the user, it is very nice to control device, memory fraction, and pre-allocation
        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [sum], &client).expect("executable");

        let x_input = xla::Literal::scalar(2f32);
        let y_input = xla::Literal::scalar(3f32);
        // args are just provided in the order they are defined, would be nice to pass a dict or something
        // a pjrtbuffer is just an array slice on some device
        // but im not sure why its a nested vector instead of just one vector
        let device_result = executable.execute(&[x_input, y_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], 9f32);
    }

    #[test]
    fn test_vector_matrix_bf16() {
        let mut ctx = Context::new();

        let foo = ctx.vector([1, 2, 3], xla::ElementType::Bf16).expect("foo");
        let bar = ctx
            .matrix([[4, 5, 6], [7, 8, 9], [10, 11, 12]], xla::ElementType::Bf16)
            .expect("bar");
        println!("output!");
        if let Operation::Constant(binding) = ctx.nodes[bar].operation.clone() {
            match binding.value.element_type() {
                Ok(ty) => println!("{}", ty),
                Err(_) => println!("Error getting element type"),
            }
        }

        let baz = ctx.reshape_const(foo, [1, 3]).expect("baz");
        if let Operation::Constant(binding) = ctx.nodes[baz].operation.clone() {
            match binding.value.element_type() {
                Ok(ty) => println!("{}", ty),
                Err(_) => println!("Error getting element type"),
            }
        }
        let barbaz = ctx.mul(bar, baz).expect("barbaz");
        if let Operation::Constant(binding) = ctx.nodes[barbaz].operation.clone() {
            match binding.value.element_type() {
                Ok(ty) => println!("{}", ty),
                Err(_) => println!("Error getting element type"),
            }
        }

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");

        let executable = ctx.compile("test", [barbaz], &client).expect("executable");

        let device_result = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let f32_result = untupled_result
            .convert(xla::ElementType::F32.primitive_type())
            .expect("f32 conversion");
        let rust_result = f32_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(
            rust_result.as_slice(),
            &[4f32, 10f32, 18f32, 7f32, 16f32, 27f32, 10f32, 22f32, 36f32]
        );
    }

    #[test]
    fn test_npy_loading() {
        let mut ctx = Context::new();

        let my_const = ctx.const_from_npy("test.npy").expect("my_const");
        println!("{}", ctx.nodes[my_const].dtype);
        let my_param = ctx
            .parameter("my_param", [2, 2], xla::ElementType::S64)
            .expect("my_param");

        let sum = ctx.add(my_const, my_param).expect("sum");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let executable = ctx.compile("test", [sum], &client).expect("executable");

        let my_param_input = xla::Literal::read_npy("test.npy", &()).expect("my_param_input");

        let device_result = executable.execute(&[my_param_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<i64>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[0, 2, 2, 0]);
    }

    #[test]
    fn test_multiple_outputs() {
        let mut ctx = Context::new();

        let three = ctx.scalar(3, xla::ElementType::F32).expect("three");

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", [], xla::ElementType::F32).expect("y");

        let product = ctx.mul(x, three).expect("product");
        let sum = ctx.add(product, y).expect("sum");
        let sum2 = ctx.add(three, x).expect("sum2");

        // output XLA
        // client must be exposed to the user, it is very nice to control device, memory fraction, and pre-allocation
        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [sum, product, sum2], &client)
            .expect("executable");

        let x_input = xla::Literal::scalar(2f32);
        let y_input = xla::Literal::scalar(3f32);
        // args are just provided in the order they are defined, would be nice to pass a dict or something
        // a pjrtbuffer is just an array slice on some device
        // but im not sure why its a nested vector instead of just one vector
        let device_result = executable.execute(&[x_input, y_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let (eval_sum, eval_product, eval_sum2) = host_result.to_tuple3().expect("untuple");
        let rust_result1 = eval_sum.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result1);

        assert_eq!(rust_result1[0], 9f32);
        let rust_result2 = eval_product.to_vec::<f32>().expect("to_vec");
        assert_eq!(rust_result2[0], 6f32);
        let rust_result3 = eval_sum2.to_vec::<f32>().expect("to_vec");
        assert_eq!(rust_result3[0], 5f32)
    }

    #[test]
    fn test_minimum() {
        let mut ctx = Context::new();

        let test_const1 = ctx.const_from_npy("test.npy").expect("test_const1");
        let test_const2 = ctx.const_from_npy("test2.npy").expect("test_const2");
        let min = ctx.minimum(test_const1, test_const2).expect("min");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [min], &client).expect("executable");

        let device_result = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<i64>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[-2, 1, 1, -2]);
    }

    #[test]
    fn test_relu() {
        let mut ctx = Context::new();

        let test_const = ctx.const_from_npy("test2.npy").expect("test_const");
        let relu = ctx.relu(test_const).expect("relu");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [relu], &client).expect("executable");

        let device_result = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<i64>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[0, 4, 4, 0]);
    }

    #[test]
    fn test_slice_in_dim() {
        let mut ctx = Context::new();

        let test_const = ctx.const_from_npy("test2.npy").expect("test_const");
        let relu = ctx.relu(test_const).expect("relu");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [relu], &client).expect("executable");

        let device_result = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<i64>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[0, 4, 4, 0]);
    }

    #[test]
    fn test_gradient_descent_polynomial() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let x2 = ctx.mul(x, x).expect("x2");
        let x4 = ctx.mul(x2, x2).expect("x4");
        let half = ctx.scalar(0.5, xla::ElementType::F32).expect("half");
        let quadratic_term = ctx.mul(half, x2).expect("quadratic_term");
        let quarter = ctx.scalar(0.25, xla::ElementType::F32).expect("half");
        let quartic_term = ctx.mul(quarter, x4).expect("quartic_term");
        let y = ctx.sub(quartic_term, quadratic_term).expect("y");

        let dydx = ctx.diff(y, x).expect("dydx");
        ctx.fold_consts(dydx, usize::max_value())
            .expect("fold consts");
        println!("{}", ctx.to_string(dydx));
        let lr = ctx.scalar(0.75, xla::ElementType::F32).expect("lr");
        let update = ctx.mul(lr, dydx).expect("update");
        let new_x = ctx.sub(x, update).expect("new_x");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [y, dydx, new_x], &client)
            .expect("executable");

        let mut x_rust = 0.5f32;
        println!("x = {}", x_rust);
        let y_vals: [f32; 5] = [
            -0.109375,
            -0.21204352,
            -0.24990773,
            -0.24997526,
            -0.24999404,
        ];

        for i in 0..5 {
            let x_xla = xla::Literal::scalar(x_rust);
            let buffers = executable.execute(&[x_xla]).expect("execute");
            let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
            let (y, dydx, x) = literals.to_tuple3().expect("untuple");
            let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
            let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
            x_rust = x.to_vec::<f32>().expect("to_vec")[0];
            println!("y = {}", y_rust);
            assert_eq!(y_rust, y_vals[i]);
            println!("dydx = {}", dydx_rust);
            println!("x = {}", x_rust);
        }
        let x_xla = xla::Literal::scalar(x_rust);
        let buffers = executable.execute(&[x_xla]).expect("execute");
        let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
        let (y, dydx, x) = literals.to_tuple3().expect("untuple");
        let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
        let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
        println!("y = {}", y_rust);
        println!("dydx = {}", dydx_rust);
    }

    #[test]
    fn test_gradient_descent_relu() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let rx = ctx.relu(x).expect("rx");
        let nx = ctx.neg(x);
        let rnx = ctx.relu(nx).expect("rnx");
        let y = ctx.add(rnx, rx).expect("y");

        let dydx = ctx.diff(y, x.into()).expect("dydx");
        println!("{}", ctx.to_string(dydx));
        let lr = ctx.scalar(0.1, xla::ElementType::F32).expect("lr");
        let update = ctx.mul(lr, dydx).expect("update");
        let new_x = ctx.sub(x, update).expect("new_x");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [y, dydx, new_x], &client)
            .expect("executable");

        let mut x_rust = 1f32;
        println!("x = {}", x_rust);
        let y_vals: [f32; 10] = [
            1.0, 0.9, 0.79999995, 0.6999999, 0.5999999, 0.4999999, 0.39999992, 0.29999992,
            0.19999993, 0.09999993,
        ];

        for i in 0..10 {
            let x_xla = xla::Literal::scalar(x_rust);
            let buffers = executable.execute(&[x_xla]).expect("execute");
            let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
            let (y, dydx, x) = literals.to_tuple3().expect("untuple");
            let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
            let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
            x_rust = x.to_vec::<f32>().expect("to_vec")[0];
            assert_eq!(y_rust, y_vals[i]);
            println!("y = {}", y_rust);
            println!("dydx = {}", dydx_rust);
            println!("x = {}", x_rust);
        }
        let x_xla = xla::Literal::scalar(x_rust);
        let buffers = executable.execute(&[x_xla]).expect("execute");
        let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
        let (y, dydx, x) = literals.to_tuple3().expect("untuple");
        let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
        let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
        println!("y = {}", y_rust);
        println!("dydx = {}", dydx_rust);
    }

    #[test]
    fn test_gradient_descent_div() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");
        let rx = ctx.relu(x).expect("rx");
        let nx = ctx.neg(x);
        let rnx = ctx.relu(nx).expect("rnx");
        let abs = ctx.add(rnx, rx).expect("y");
        let one = ctx.scalar(1.0, xla::ElementType::F32).expect("one");
        let abs1 = ctx.add(one, abs).expect("abs1");
        let div = ctx.div(one, abs1).expect("div");
        // 1 - 1/(1 + abs(x))
        let y = ctx.sub(one, div).expect("div1");

        let dydx = ctx.diff(y, x.into()).expect("dydx");
        println!("{}", ctx.to_string(dydx));
        let lr = ctx.scalar(0.05, xla::ElementType::F32).expect("lr");
        let update = ctx.mul(lr, dydx).expect("update");
        let new_x = ctx.sub(x, update).expect("new_x");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [y, dydx, new_x], &client)
            .expect("executable");

        let mut x_rust = 0.5f32;
        println!("x = {}", x_rust);

        for i in 0..20 {
            let x_xla = xla::Literal::scalar(x_rust);
            let buffers = executable.execute(&[x_xla]).expect("execute");
            let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
            let (y, dydx, x) = literals.to_tuple3().expect("untuple");
            let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
            let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
            x_rust = x.to_vec::<f32>().expect("to_vec")[0];
            println!("y = {}", y_rust);
            println!("dydx = {}", dydx_rust);
            println!("x = {}", x_rust);
        }
        let x_xla = xla::Literal::scalar(x_rust);
        let buffers = executable.execute(&[x_xla]).expect("execute");
        let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
        let (y, dydx, x) = literals.to_tuple3().expect("untuple");
        let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
        let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
        println!("y = {}", y_rust);
        println!("dydx = {}", dydx_rust);
        assert_eq!(x_rust, 0.0052729174);
        assert_eq!(y_rust, 0.0052452087);
    }

    #[test]
    fn test_gradient_descent_reduce_mean() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [2], xla::ElementType::F32).expect("x");
        let x2 = ctx.mul(x, x).expect("x2");
        let y = ctx.reduce_mean(x2, 0, true).expect("y");

        let dydx = ctx.diff(y, x.into()).expect("dydx");
        ctx.fold_consts(dydx, usize::max_value()).expect("fold_consts");
        println!("{}", ctx.to_string(dydx));
        assert_eq!(ctx.to_string(dydx), "Mul (Mul (Constant Scalar 2) (Parameter Vector2 x)) (Constant Scalar 0.5)");
        let lr = ctx.scalar(1, xla::ElementType::F32).expect("lr");
        let update = ctx.mul(lr, dydx).expect("update");
        let new_x = ctx.sub(x, update).expect("new_x");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [y, dydx, new_x], &client)
            .expect("executable");

        let mut x_rust = [1f32, 1f32];
        println!("x = [{}, {}]", x_rust[0], x_rust[1]);

        for _ in 0..1 {
            let x_xla = xla::Literal::vec1(&x_rust);
            let buffers = executable.execute(&[x_xla]).expect("execute");
            let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
            let (y, dydx, x) = literals.to_tuple3().expect("untuple");
            let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
            let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
            let as_vec = x.to_vec::<f32>().expect("to_vec");
            x_rust = [as_vec[0], as_vec[1]];
            println!("y = {}", y_rust);
            assert_eq!(y_rust, 1f32);
            println!("dydx = {}", dydx_rust);
            println!("x = [{}, {}]", x_rust[0], x_rust[1]);
        }
        let x_xla = xla::Literal::vec1(&x_rust);
        let buffers = executable.execute(&[x_xla]).expect("execute");
        let literals = buffers[0][0].to_literal_sync().expect("to_literal_sync");
        let (y, dydx, x) = literals.to_tuple3().expect("untuple");
        let y_rust = y.to_vec::<f32>().expect("to_vec")[0];
        let dydx_rust = dydx.to_vec::<f32>().expect("to_vec")[0];
        println!("y = {}", y_rust);
        println!("dydx = {}", dydx_rust);
    }

}
