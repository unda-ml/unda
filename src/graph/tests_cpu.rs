#[allow(unused_macros)]
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
    use std::collections::HashMap;

    use crate::graph::{Context, Node};
    use xla::{FromRawBytes, Literal, Shape};

    create_test!(test_pow_f32_100_squared, pow, F32, 10f32, 2f32, 100f32);
    create_test!(test_pow_f32_3_squared, pow, F32, 3f32, 2f32, 9f32);
    create_test!(test_pow_f32_zeroth_pow, pow, F32, 3f32, 0f32, 1f32);
    create_test!(test_ln_10, log, F32, 10f32, f32::ln(10f32));
    create_test!(test_ln_e, log, F32, 1f32, 0f32);
    create_test!(test_add_1_2, add, F32, 1f32, 2f32, 3f32);
    create_test!(test_sub_1_2, sub, F32, 1f32, 2f32, -1f32);

    #[test]
    fn test_normal_dist() {
        let mut ctx = Context::new();
        let mu = ctx.scalar(0, xla::ElementType::F32).expect("mu = 0");
        let sigma = ctx.scalar(1, xla::ElementType::F32).expect("sigma = 1");
        let mat = ctx.rng_normal(mu, sigma, &[2,3]).expect("sample the normal distribution");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [mat], &client).expect("executable");

        let device_result = executable.execute::<Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        match untupled_result.shape().unwrap() {
            Shape::Array(shape) => {
                assert_eq!(shape.dims(), &[2,3]);
            },
            _ => {
                panic!("Shape is not correct");
            }
        }
    }

    #[test]
    fn test_uniform_dist() {
        let mut ctx = Context::new();
        let min = ctx.scalar(0, xla::ElementType::F32).expect("min = 0");
        let max = ctx.scalar(1, xla::ElementType::F32).expect("max = 10");
        let mat = ctx.rng_uniform(min, max, &[10,1]).expect("sample the uniform distribution");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [mat], &client).expect("executable");

        let device_result = executable.execute::<Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        match untupled_result.shape().unwrap() {
            Shape::Array(shape) => {
                assert_eq!(shape.dims(), &[10,1]);
            },
            _ => {
                panic!("Shape is not correct");
            }
        }
    }


    #[test]
    fn test_large_cte() {
        let mut ctx = Context::new();
        let a = ctx.parameter("a", [], xla::ElementType::F32).expect("a");
        let two = ctx.scalar(2, xla::ElementType::F32).expect("2");

        let a_2 = ctx.pow(a, two).expect("a^2");
        let a_21 = ctx.pow(a, two).expect("a^2");
        let a_22 = ctx.pow(a, two).expect("a^2");
        let a_23 = ctx.pow(a, two).expect("a^2");
        let a_24 = ctx.pow(a, two).expect("a^2");
        let a_25 = ctx.pow(a, two).expect("a^2");
        let a_26 = ctx.pow(a, two).expect("a^2");
        let a_27 = ctx.pow(a, two).expect("a^2");


        let sum1 = ctx.add(a_2, a_21).expect("a^2 + a^2");
        let sum2 = ctx.add(a_22, a_23).expect("a^2 + a^2");
        let sum3 = ctx.add(a_24, a_25).expect("a^2 + a^2");
        let sum4 = ctx.add(a_26, a_27).expect("a^2 + a^2");

        let nest_sum1 = ctx.add(sum1, sum2).expect("(a^2 + a^2) + (a^2 + a^2)");
        let nest_sum2 = ctx.add(sum3, sum4).expect("(a^2 + a^2) + (a^2 + a^2)");

        let res = ctx.add(nest_sum1, nest_sum2).expect("((a^2 + a^2) + (a^2 + a^2)) + ((a^2 + a^2) + (a^2 + a^2))");
        let subterm_extract = ctx.extract_subterms(&[res], usize::MAX).expect("CTE");

        assert!(subterm_extract);
    }

    #[test]
    fn test_cte_happened() {
        let mut ctx = Context::new();
        let a = ctx.parameter("a", [], xla::ElementType::F32).expect("a");
        let b = ctx.parameter("b", [], xla::ElementType::F32).expect("b");

        let a_p_b = ctx.add(a, b).expect("a + b");
        let a_p_b_again = ctx.add(a, b).expect("a + b again");

        let res = ctx.mul(a_p_b, a_p_b_again).expect("(a + b) * (a + b)");
        let subterm_extract = ctx.extract_subterms(&[res], 10).expect("CTE");

        assert!(subterm_extract);
    }

    #[test]
    fn test_cte_no_false_positives() {
        let mut ctx = Context::new();
        let a = ctx.parameter("a", [], xla::ElementType::F32).expect("a");
        let b = ctx.parameter("b", [], xla::ElementType::F32).expect("b");

        let a_p_b = ctx.add(a, b).expect("a + b");

        let c = ctx.parameter("c", [], xla::ElementType::F32).expect("c");
        let res = ctx.mul(a_p_b, c).expect("(a+b) * c");
        let subterm_extract = ctx.extract_subterms(&[res], 10).expect("CTE");

        assert!(!subterm_extract);
    }

    #[test]
    fn test_hash_node() {
        let mut ctx = Context::new();
        let mut hash_map: HashMap<Node, f32> = HashMap::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let three = ctx.scalar(3, xla::ElementType::F32).expect("three");
        let three_x_b = ctx.mul(three, x).expect("3x");
        let three_x_a = ctx.mul(three, x).expect("3x again");

        let node_a = ctx.nodes[three_x_a].clone();
        let node_b = ctx.nodes[three_x_b].clone();

        hash_map.insert(node_a, 1.0);
        hash_map.insert(node_b, 2.0);

        assert_eq!(hash_map.keys().len(), 1)
    }

    #[test]
    fn test_mat_mul_panics() {
        let mut ctx = Context::new();
        let mat_a = ctx.matrix([[1,2], [3,4], [5,6]], xla::ElementType::S32).expect("initial mat");
        let mat_b = ctx.matrix([[7,8], [9, 10], [11, 12]], xla::ElementType::S32).expect("initial mat");

        assert!(ctx.matmul(mat_a, mat_b).is_err());
    }

    #[test]
    fn test_mat_mul_batch() {
        let mut ctx = Context::new();
        let mat_a = ctx.tensor_4d([[[1,2], [3,4], [5,6]],[[7,8],[9,10],[11,12]]], xla::ElementType::S32).expect("initial batch mat");
        let mat_b = ctx.matrix([[7,8,9], [10,11,12]], xla::ElementType::S32).expect("initial mat");

        let mul = ctx.matmul(mat_a, mat_b).expect("MatMul A x B");

        assert_eq!(&ctx.nodes[mul].shape.sizes.as_slice(), &[3,2,3])
    }
    #[test]
    fn test_mat_mul() {
        let mut ctx = Context::new();
        let mat_a = ctx.matrix([[1,2], [3,4], [5,6]], xla::ElementType::S32).expect("initial mat");
        let mat_b = ctx.matrix([[7,8,9], [10,11,12]], xla::ElementType::S32).expect("initial mat");

        let mul = ctx.matmul(mat_a, mat_b).expect("MatMul A x B");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [mul], &client).expect("executable");

        let device_result = executable.execute::<Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<i32>().expect("to_vec");
        println!("{:?}", rust_result);

        match untupled_result.shape().unwrap() {
            Shape::Array(shape) => {
                assert_eq!(shape.dims(), &[3,3]);
            },
            _ => {
                panic!("matrix transpose result is not an ArrayShape");
            }
        }
        //assert_eq!(untupled_result.shape()?, [3, 2]);
        assert_eq!(rust_result.as_slice(), &[27,30,33,61,68,75,95,106,117]);
    }

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
    fn test_mul_scalar_consts_and_params() {
        let mut ctx = Context::new();

        let one = ctx.scalar(1, xla::ElementType::F32).expect("Scalar 1");

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let product = ctx.mul(x, one).expect("product");

        // output XLA
        // client must be exposed to the user, it is very nice to control device, memory fraction, and pre-allocation
        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
        let name = "test";
        /*let executable = ctx.compile(&name, [product], &client).expect("executable");

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
        println!("{:?}", rust_result);*/
        let fold_const = ctx.fold_consts(product, usize::MAX).expect("fold it");
        ctx.compile(name, [product], &client).expect("Compile");
        println!("{}", ctx.to_string(product));

        for (_, val) in ctx.nodes {
            println!("{}", val.to_string());
        }

        assert!(fold_const);
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



    /*#[test]
    fn test_softmax_diff() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [4],xla::ElementType::F32).expect("test_const");
        let softmax = ctx.softmax(x).expect("softmax");
        println!("{}", ctx.nodes[softmax].shape);
        let dydx = ctx.diff(softmax, x).expect("dy/dx");
        println!("{:?}\n", ctx.to_string(softmax));
        println!("{:?}", ctx.to_string(dydx));

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [dydx], &client).expect("executable");

        let x_input = xla::Literal::vec1(&[1.0f32,3.0f32,4.0f32,0.5f32]);

        let device_result = executable.execute::<Literal>(&[x_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);
    }*/

    /*#[test]
    fn test_reducesum_diff() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [4],xla::ElementType::F32).expect("test_const");
        //let softmax = ctx.softmax(x).expect("softmax");
        let sum = ctx.reduce_sum(x, 0, true).expect("sum");
        let div = ctx.div(x, sum).expect("div");
        //println!("{}", ctx.nodes[div].shape);
        let dydx = ctx.diff(div, x).expect("dy/dx");
        println!("{:?}\n", ctx.to_string(div));
        println!("{:?}", ctx.to_string(dydx));

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [dydx], &client).expect("executable");

        let x_input = xla::Literal::vec1(&[1.0f32,3.0f32,4.0f32,0.5f32]);

        let device_result = executable.execute::<Literal>(&[x_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);
    }*/

    #[test]
    fn test_softmax() {
        let mut ctx = Context::new();

        let test_const = ctx.vector([100f32,100f32,40f32,10f32], xla::ElementType::F32).expect("test_const");
        let relu = ctx.softmax(test_const).expect("softmax");

        let client = xla::PjRtClient::cpu().expect("client");//gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [relu], &client).expect("executable");

        let device_result = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[0.5, 0.5, 4.3782554e-27, 4.097e-40]);
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
    fn test_gradient_descent_polynomial_pows() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let two = ctx.scalar(2, xla::ElementType::F32).expect("2");
        let four = ctx.scalar(4, xla::ElementType::F32).expect("4");

        let x2 = ctx.pow(x, two).expect("x2");
        let x4 = ctx.pow(x, four).expect("x4");
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
            -0.24990776,
            -0.24997526,
            -0.24999407,
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
