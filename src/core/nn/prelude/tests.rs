#[cfg(test)]
mod tests {
    use xla::PjRtClient;

    use crate::core::{graph::Context, nn::prelude::initializers::Initializer};

    #[test]
    fn test_he() {
        let mut ctx = Context::new();
        let he = Initializer::He;
        
        let curr_scalar = ctx.scalar(0, xla::ElementType::F32).expect("0");
        let rng_scalar = he.initialize(&mut ctx, curr_scalar, 10).expect("He initialization");

        let client = PjRtClient::cpu().expect("CPU Client");
        let executable = ctx.compile("test", [rng_scalar], &client).expect("Compile");

        let res = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = res[0][0]
            .to_literal_sync()
            .expect("To literal sync");
        let untupled_rs = host_result.to_tuple1().expect("untuple");
        let f32_res = untupled_rs.convert(xla::ElementType::F32.primitive_type()).expect("res -> f32");
        let rust_res = f32_res.to_vec::<f32>().expect("res!");

        assert!(rust_res[0] <= 2f32 && rust_res[0] >= -2f32) //Could fail very infrequently because
                                                             //normal distribution is technically
                                                             //cts on (-inf, inf). But for the most
                                                             //part with n = 10, bounds should be
                                                             //within [-1.5,1.5]
    }

    #[test]
    fn test_xavier() {
        let mut ctx = Context::new();
        let xavier = Initializer::Xavier;
        
        let curr_scalar = ctx.scalar(0, xla::ElementType::F32).expect("0");
        let rng_scalar = xavier.initialize(&mut ctx, curr_scalar, 10).expect("Xavier initialization");

        let client = PjRtClient::cpu().expect("CPU Client");
        let executable = ctx.compile("test", [rng_scalar], &client).expect("Compile");

        let res = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = res[0][0]
            .to_literal_sync()
            .expect("To literal sync");
        let untupled_rs = host_result.to_tuple1().expect("untuple");
        let f32_res = untupled_rs.convert(xla::ElementType::F32.primitive_type()).expect("res -> f32");
        let rust_res = f32_res.to_vec::<f32>().expect("res!");

        assert!(rust_res[0] <= 0.4 && rust_res[0] >= -0.4)
    }

    #[test]
    fn test_norm() {
        let mut ctx = Context::new();
        let uniform = Initializer::Default;
        
        let curr_scalar = ctx.scalar(0, xla::ElementType::F32).expect("0");
        let rng_scalar = uniform.initialize(&mut ctx, curr_scalar, 10).expect("Xavier initialization");

        let client = PjRtClient::cpu().expect("CPU Client");
        let executable = ctx.compile("test", [rng_scalar], &client).expect("Compile");

        let res = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = res[0][0]
            .to_literal_sync()
            .expect("To literal sync");
        let untupled_rs = host_result.to_tuple1().expect("untuple");
        let f32_res = untupled_rs.convert(xla::ElementType::F32.primitive_type()).expect("res -> f32");
        let rust_res = f32_res.to_vec::<f32>().expect("res!");

        assert!(rust_res[0] <= 1f32 && rust_res[0] >= -1f32)       
    }
}
