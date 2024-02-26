#[cfg(test)]
mod tests {
    use crate::core::graph::{Context, Node, callsite::callsite, ConstantBinding};
    use xla::{FromRawBytes, Literal, Shape};

    #[test]
    fn ensure_is_zero_scalar(){
        let mut ctx = Context::new();
        let zeroes = ctx.scalar(0, xla::ElementType::F32).expect("zero scalar");
        let node = ctx.nodes.get(zeroes).expect("node of zero");

        assert!(node.is_zero().expect("is zero"));
    }

    #[test]
    fn ensure_is_zero_vector(){
        let mut ctx = Context::new();
        let zeroes = ctx.vector([0.0,0.0,0.0,0.0], xla::ElementType::F64).expect("zero vector");
        let node = ctx.nodes.get(zeroes).expect("node of zeroes");

        assert!(node.is_zero().expect("is zero"));
    }

    #[test]
    fn ensure_is_zero_unique_types(){
        let mut ctx = Context::new();
        let zeroes = ctx.matrix([[0u64,0u64],[0u64,0u64],[0u64,0u64]], xla::ElementType::U64).expect("zero matrix u64");
        let node = ctx.nodes.get(zeroes).expect("node of zeroes");

        assert!(node.is_zero().expect("is zero"));
    }

    #[test]
    fn ensure_is_const(){
        let mut ctx = Context::new();
        let scalar_const = ctx.scalar(15, xla::ElementType::F32).expect("fifteen");
        let node = ctx.nodes.get(scalar_const).expect("node of 15");

        assert!(node.is_const().is_some());
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

        let baz = ctx.reshape_const(foo, [1, 3]).expect("baz");
        let barbaz = ctx.mul(bar, baz).expect("barbaz");

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
        let executable = ctx.compile(&name, [sum, product, sum2], &client).expect("executable");

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
}
