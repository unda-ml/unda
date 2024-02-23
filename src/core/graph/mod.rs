mod callsite;
mod context;
mod shape;
mod operation;
use callsite::*;
use smallvec::SmallVec;
use xla::ElementType::F32;
pub use context::*;
pub use shape::*;

use crate::core::error;

//#[test]
pub fn example() {
    let mut ctx = Context::new();

    let three = ctx.scalar(3f32, F32).expect("three");
    //let up = ctx.vector([0.0, 0.0, 1.0]);
    //let id3x3 = ctx.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let x = ctx.parameter("x", &[], xla::ElementType::F32);
    let y = ctx.parameter("y", &[], xla::ElementType::F32);

    let product = ctx.mul(x, three).expect("product");
    let sum = ctx.add(product, y).expect("sum");
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
    let client = xla::PjRtClient::gpu(0.7, false).expect("client");
    let name = "test";
    let executable = ctx.compile(sum, &name, &client).expect("executable");

    let x_input = xla::Literal::scalar(2f32);
    let y_input = xla::Literal::scalar(3f32);
    // args are just provided in the order they are defined, would be nice to pass a dict or something
    // a pjrtbuffer is just an array slice on some device
    // but im not sure why its a nested vector instead of just one vector
    let device_result = executable.execute(&[x_input, y_input]).expect("execute");
    let host_result = device_result[0][0].to_literal_sync().expect("to_literal_sync");
    let rust_result = host_result.to_vec::<f32>().expect("to_vec");
    println!("{:?}", rust_result);
}
