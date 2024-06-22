use crate::graph::shape::Shape;

pub trait Initializer {
    fn initialize(
        &self,
        seed: i64,
        shape: &Shape,
        dtype: xla::ElementType,
    ) -> xla::Result<xla::Literal>;
}

struct Constant {
    constant: f32
}

impl Initializer for Constant {
    fn initialize(
        &self,
        _: i64,
        shape: &Shape,
        dtype: xla::ElementType,
    ) -> xla::Result<xla::Literal> {
        let const_vec = [self.constant].repeat(shape.size());
        let const_r1 = xla::Literal::vec1(&const_vec);
        let shape_i64 = shape.sizes.iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let const_shaped = xla::Literal::reshape(&const_r1, &shape_i64)?;
        xla::Literal::convert(&const_shaped, dtype.primitive_type())
    }
}
