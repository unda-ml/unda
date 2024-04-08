use smallvec::SmallVec;
use xla::ArrayShape;

use super::callsite::Callsite;

/// array of sizes along each axis
/// scalar would be an []
/// 3d vector would be [3]
/// 4x3 matrix would be [4,3]
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct Shape {
    /// smallvec to avoid indirection in the common case of dimension <= 8
    pub sizes: SmallVec<[u32; 4]>,
}

#[derive(thiserror::Error, Debug)]
pub enum ShapeConversionError {
    #[error("Expected Array Shape but got Tuple Shape")]
    UnexpectedTupleShape,
    #[error("Expected Array Shape but got Unsupported Shape")]
    UnexpectedUnsupportedShape,
    #[error("Shapes {0} and {1} are of different sizes at {2}.")]
    MismatchedSizes(Shape, Shape, Callsite),
}

impl From<&[u32]> for Shape {
    fn from(value: &[u32]) -> Self {
        Shape {
            sizes: SmallVec::from_slice(value),
        }
    }
}

impl<const N: usize> From<[u32; N]> for Shape {
    fn from(value: [u32; N]) -> Self {
        Shape {
            sizes: SmallVec::from_slice(&value),
        }
    }
}

impl Shape {
    pub fn new() -> Self {
        Self {
            sizes: SmallVec::new(),
        }
    }

    pub fn ndims(&self) -> usize {
        self.sizes.len()
    }

    pub fn size(&self) -> usize {
        self.sizes.iter().fold(1, |x, y| x*(*y as usize))
    }

    /// Convert from xla-rs shape
    pub fn from_xla_shape(shape: xla::Shape) -> Result<Shape, ShapeConversionError> {
        match shape {
            xla::Shape::Tuple(_) => Err(ShapeConversionError::UnexpectedTupleShape),
            xla::Shape::Unsupported(_) => Err(ShapeConversionError::UnexpectedUnsupportedShape),
            xla::Shape::Array(s) => Ok(Shape {
                sizes: s.dims().iter().map(|d| *d as u32).collect(),
            }),
        }
    }

    pub fn to_array_shape(&self, dtype: xla::ElementType) -> ArrayShape {
        ArrayShape::new(self.sizes.iter().map(|d| *d as i64).collect(), dtype)
    }

    pub fn matmul_shape(&self, dim2: &[u32]) -> Option<Vec<u32>> {
        let dim1 = &self.sizes;
        if dim1.last()? == dim2.get(dim2.len().saturating_sub(2))? {
            let mut result_shape = Vec::new();

            for &dim in dim1.iter().take(dim1.len() - 1) {
                result_shape.push(dim);
            }

            for (i, &dim) in dim2.iter().enumerate() {
                if i != dim2.len() - 2 {
                    result_shape.push(dim);
                }
            }
            Some(result_shape)
        } else {
            None
        }
    }

    pub fn broadcast(&self, shape: &Shape) -> Option<Shape> {
        if self.sizes.is_empty() {
            Some(shape.clone())
        } else if shape.sizes.is_empty() {
            Some(self.clone())
        } else if self.sizes.len() != shape.sizes.len() {
            None
        } else {
            let (large_shape, small_shape) = {
                if self.size() > shape.size() {
                    (&self.sizes, &shape.sizes)
                } else {
                    (&shape.sizes, &self.sizes)
                }
            };
            for i in 0..self.ndims() {
                if small_shape[i] != large_shape[i] && small_shape[i] != 1 {
                    return None;
                }
            }
            Some(Shape{ sizes: large_shape.clone() })
        }
    }
}

impl Default for Shape {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
