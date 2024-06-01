use crate::graph::shape::Shape;

pub trait Initializer {
    fn initialize(&self, seed: i64, shape: &Shape, dtype: xla::ElementType) -> xla::Result<xla::Literal>;
}

struct Zeroes {
}

impl Initializer for Zeroes {
    fn initialize(&self, seed: i64, shape: &Shape, dtype: xla::ElementType) -> xla::Result<xla::Literal> {
        let zeroes_vec = [0i64].repeat(shape.size());
        let zeroes_r1 = xla::Literal::vec1(&zeroes_vec);
        let shape_i64 = shape.sizes.iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let zeroes_shaped = xla::Literal::reshape(&zeroes_r1, &shape_i64)?;
        xla::Literal::convert(&zeroes_shaped, dtype.primitive_type())
    }
}