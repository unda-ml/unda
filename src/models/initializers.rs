use crate::graph::shape::Shape;

pub trait Initializer {
    fn initialize(seed: i64, shape: Shape) -> xla::Literal;
}