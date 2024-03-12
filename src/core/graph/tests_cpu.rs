macro_rules! create_test {
    ($name:ident, $op:ident, $dtype:ident, $in1:expr, $in2:expr, $exp:expr) => {
        #[test]
        fn $name() {
            let mut ctx = Context::new();
            let x = ctx.parameter("x", [], xla::ElementType::$dtype).expect("x");
            let y = ctx.parameter("y", [], xla::ElementType::$dtype).expect("y");

            let operation = ctx.$op(x, y).expect("operation");

            let client = xla::PjRtClient::cpu().expect("client");
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

            assert_eq!(rust_result[0], $exp);
        }
    };
    ($name:ident, $op:ident, $dtype:ident, $in:expr, $exp:expr) => {
        #[test]
        fn $name() {
            let mut ctx = Context::new();
            let x = ctx.parameter("x", [], xla::ElementType::$dtype).expect("x");

            let operation = ctx.$op(x).expect("operation");

            let client = xla::PjRtClient::cpu().expect("client");
            let name = "test";
            let executable = ctx.compile(&name, [operation], &client).expect("executable");

            let x_input = xla::Literal::scalar($in);

            let device_result = executable.execute(&[x_input]).expect("execute");
            let host_result = device_result[0][0]
                .to_literal_sync()
                .expect("to_literal_sync");
            let untupled_result = host_result.to_tuple1().expect("untuple");
            let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");

            assert_eq!(rust_result[0], $exp);
        }
    };

}

#[cfg(test)]
mod tests {
    use crate::core::graph::Context;
    use xla::{FromRawBytes, Literal, Shape};

    create_test!(test_pow_f32_100_squared, pow, F32, 10f32, 2f32, 100f32);
    create_test!(test_pow_f32_3_squared, pow, F32, 3f32, 2f32, 9f32);
    create_test!(test_pow_f32_zeroth_pow, pow, F32, 3f32, 0f32, 1f32);
    create_test!(test_ln_10, log, F32, 10f32, f32::ln(10f32));
    create_test!(test_ln_e, log, F32, 1f32, 0f32);
    create_test!(test_add_1_2, add, F32, 1f32, 2f32, 3f32);
    create_test!(test_sub_1_2, sub, F32, 1f32, 2f32, -1f32);

    #[test]
    fn test_transpose() {
        let mut ctx = Context::new();
        let mat = ctx.matrix([[1,2,3], [4,5,6]], xla::ElementType::F32).expect("initial mat");

        let transpose = ctx.transpose(mat, &[1,0]).expect("transpose mat");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [transpose], &client).expect("executable");

        let device_result = executable.execute::<Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        match untupled_result.shape().unwrap() {
            Shape::Array(shape) => {
                assert_eq!(shape.dims(), &[3,2]);
            },
            _ => {
                panic!("matrix transpose result is not an ArrayShape");
            }
        }
        //assert_eq!(untupled_result.shape()?, [3, 2]);
        assert_eq!(rust_result.as_slice(), &[1f32,4f32,2f32,5f32,3f32,6f32]);
    }

    /*#[test]
    fn test_inv_perm_transpose() {
        let before = &[1,0];
        let after = Context::inv_perm(before);

        assert_eq!(&[0,1], after.as_slice());
    }*/

    #[test]
    fn test_inv_perm() {
        let before = &[1,2,0,3];
        let after = Context::inv_perm(before);

        assert_eq!(&[2,0,1,3], after.as_slice());
    }

    #[test]
    fn test_inv_perm_cplx() {
        let before = &[4,8,0,7,1,5,3,6,2];
        let after = Context::inv_perm(before);

        assert_eq!(&[2,4,8,6,0,5,7,3,1], after.as_slice())
    }

    #[test]
    fn test_exp() {
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let exp = ctx.exp(x).expect("e ^ x");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [exp], &client).expect("executable");

        let x_input = xla::Literal::scalar(2f32);
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

        assert_eq!(rust_result[0], f32::exp(2f32));

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
        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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

        let baz = ctx.reshape_const(foo, [1, 3]).expect("baz");
        let barbaz = ctx.mul(bar, baz).expect("barbaz");

        let client = xla::PjRtClient::cpu().expect("client");//(0.7, false).expect("client");
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

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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
        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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

        let dydx = ctx.diff(y, x.into()).expect("dydx");
        println!("{}", ctx.to_string(dydx));
        let lr = ctx.scalar(0.75, xla::ElementType::F32).expect("lr");
        let update = ctx.mul(lr, dydx).expect("update");
        let new_x = ctx.sub(x, update).expect("new_x");

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
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

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx
            .compile(&name, [y, dydx, new_x], &client)
            .expect("executable");

        let mut x_rust = 1f32;
        println!("x = {}", x_rust);
        let y_vals: [f32; 10] = [
            1.0,
            0.9,
            0.79999995,
            0.6999999,
            0.5999999,
            0.4999999,
            0.39999992,
            0.29999992,
            0.19999993,
            0.09999993,
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
}
