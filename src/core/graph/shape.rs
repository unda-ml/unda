use smallvec::SmallVec;
use std::fmt::{Display, Formatter};
use crate::core::error::{Error, Result};

/// array of sizes along each axis
/// scalar would be an vec![]
/// 3d vector would be vec![3]
/// 4x3 matrix would be vec![4,3]
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    /// smallvec to avoid indirection in the common case of dimension <= 4
    // TODO: is u16 enough here? tune this
    pub sizes: SmallVec<[u16; 4]>,
}

impl Shape {
    pub fn new() -> Self {
        Self {
            sizes: SmallVec::new(),
        }
    }

    pub fn scalar() -> Self {
        Self::new()
    }

    /// Allows syntax `Shape::of(N)`
    pub fn of(size: u16) -> Self {
        let mut sizes = SmallVec::new();
        sizes.push(size);
        Self { sizes }
    }

    /// Allows syntax `Shape::of(N).by(M)`
    pub fn by(self, size: u16) -> Self {
        let Self { mut sizes } = self;
        sizes.push(size);
        Self { sizes }
    }

    /// Convert from xla-rs shape
    pub fn from_xla_shape(shape: xla::Shape) -> Result<Shape> {
        match shape {
            xla::Shape::Tuple(_) => Err(Error::ShapeConversionError { msg: "Tuple".to_string() } ),
            xla::Shape::Unsupported(_) => Err(Error::ShapeConversionError { msg: "Unsupported".to_string() } ),
            xla::Shape::Array(s) => Ok(Shape { sizes: s.dims().iter().map(|d| *d as u16).collect::<SmallVec<[u16; 4]>>() } ),
        }
    }
}

impl Default for Shape {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self.sizes.len() {
            0 => write!(f, "Scalar"),
            1 => write!(f, "Vector"),
            2 => write!(f, "Matrix"),
            _ => write!(f, "Tensor"),
        }?;
        if self.sizes.is_empty() {
            return Ok(());
        }
        write!(f, "{}", self.sizes[0])?;
        for size in self.sizes.iter().skip(1) {
            write!(f, "x{}", size)?;
        }
        Ok(())
    }
}
