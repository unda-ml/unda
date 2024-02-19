mod callsite;
mod context;
mod shape;
mod operation;
use callsite::*;
use smallvec::SmallVec;
pub use context::*;
pub use shape::*;

//#[test]
pub fn example() {
    let mut ctx = Context::new();

    let three = ctx.scalar(3f32);
    //let up = ctx.vector([0.0, 0.0, 1.0]);
    //let id3x3 = ctx.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let x = ctx.parameter("x", SmallVec::new());
    let y = ctx.parameter("y", SmallVec::new());

    let product = ctx.mul(x, three);
    let sum = ctx.add(product, y);
    //let diff_x = ctx.diff(sum, x);
    // type safety: we cant differentiate with respect to something that isnt a parameter
    // The wrapper type `Parameter` constructor isnt be exposed to users to prevent this entirely.
    // Try uncommenting this, it doesnt compile.
    //let diff_x = ctx.diff(sum, product);

    // Dimensional safety:
    // prints "Dimension mismatch Vector3 vs Matrix3x3 at: Add unda/src/core/graph/mod.rs:29"
    // let invalid = ctx.add(up, id3x3);

    // issue: this also errors, proper dim check is not implemented. see context.rs line 116
    //let matmul = ctx.mul(up, id3x3);

    //let result = ctx.mul(diff_x, matmul);

    // output XLA
    // client must be exposed to the user, it is very nice to contorl device, memory fraction, and pre-allocation
    let maybe_client = xla::PjRtClient::gpu(0.1, false);
    let client = match maybe_client {
        Ok(c) => c,
        Err(_) => panic!("Failed to construct XLA client!")
    };
    let name = "test";
    let executable = ctx.compile(sum, &name, &client);

    let x_input = xla::Literal::scalar(2f32);
    let y_input = xla::Literal::scalar(3f32);
    // args are just provided in the order they are defined, would be nice to pass a dict or something
    // a pjrtbuffer is just an array slice on some device
    // but im not sure why its a nested vector instead of just one vector
    let device_result = match executable.execute(&[x_input, y_input]) {
        Ok(r) => r,
        Err(_) => panic!("XLA internal execution error")
    };
    let host_result = match device_result[0][0].to_literal_sync() {
        Ok(x) => x,
        Err(_) => panic!("Error while getting literal from XLA buffer")
    };
    let rust_result = match host_result.to_vec::<f32>() {
        Ok(x) => x,
        Err(_) => panic!("Error while converting XLA literal to Rust vector")
    };
    println!("{:?}", rust_result);
}
