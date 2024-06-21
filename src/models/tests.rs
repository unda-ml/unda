#[cfg(test)]
mod tests {
    use crate::{graph::Context, models::activations};
    use xla::Literal;

    #[test]
    fn test_tanh(){
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let lr = ctx.tanh(x).expect("tanh(x)");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [lr], &client).expect("executable");

        for i in -10..10 {
            let x_input = xla::Literal::scalar(i as f32);
            let device_result = executable.execute(&[x_input]).expect("execute");
            let host_result = device_result[0][0]
                .to_literal_sync()
                .expect("to_literal_sync");
            let untupled_result = host_result.to_tuple1().expect("untuple");
            let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
            println!("{:?}", rust_result);

            assert!((rust_result[0] - f32::tanh(i as f32)) <= 0.0000001);
        }

    }

    #[test]
    fn test_leaky_relu(){
        let mut ctx = Context::new();
        let x = ctx.parameter("x", [], xla::ElementType::F32).expect("x");

        let lr = ctx.leaky_relu(x, 0.001).expect("leaky_relu x");

        let client = xla::PjRtClient::cpu().expect("client");
        let name = "test";
        let executable = ctx.compile(&name, [lr], &client).expect("executable");

        let x_input = xla::Literal::scalar(2f32);
        let device_result = executable.execute(&[x_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], 2f32);
        let x_input = xla::Literal::scalar(-2f32);
        let device_result = executable.execute(&[x_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], -0.002);
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
    fn test_tanh_diff() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [4],xla::ElementType::F32).expect("test_const");
        let tanh = ctx.tanh(x).expect("tanh");
        let dydx = ctx.diff(tanh, x).expect("dy/dx");

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
    }

    #[test]
    fn test_sigmoid_diff() {
        let mut ctx = Context::new();

        let x = ctx.parameter("x", [4],xla::ElementType::F32).expect("test_const");
        let sigmoid = ctx.sigmoid(x).expect("sigmoid");
        let dydx = ctx.diff(sigmoid, x).expect("dy/dx");

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
    }

    #[test]
    fn test_fusion() {
        let mut f = Context::new();
        let x = f.parameter("x", [], xla::ElementType::F32).expect("x");
        let two = f.scalar(2, xla::ElementType::F32).expect("2");
        let mult = f.mul(x, two).expect("2x");

        let mut g = Context::new();
        let y = g.parameter("y", [], xla::ElementType::F32).expect("y");
        let two = g.scalar(2, xla::ElementType::F32).expect("2 2");
        let square = g.pow(y, two).expect("y^2");

        let output = f.merge_graphs(&g, &[square]).expect("Merge f and g");
        let name = "y";
        f.find_and_replace_params(&[(name, &[mult])]).expect("Fuse mult to y");

        let client = xla::PjRtClient::cpu().expect("Client");
        let name = "test";
        let exec = f.compile(&name, [output[0]], &client).expect("executable");

        let x_in = xla::Literal::scalar(2);
        let device_result = exec.execute::<Literal>(&[x_in]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let untupled_result = host_result.to_tuple1().expect("untuple");
        let rust_result = untupled_result.to_vec::<f32>().expect("to_vec");

        assert_eq!(16f32, rust_result[0])
    }

}
