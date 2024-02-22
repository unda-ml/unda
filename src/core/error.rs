use crate::core::graph::Shape;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Mismatched shapes
    #[error("Mismatched shapes {shape1} {shape2}")]
    ShapeError { shape1: Shape, shape2: Shape },

    //// Mismatched types
    #[error("Mismatched types {type1} {type2}")]
    TypeError { type1: xla::ElementType, type2: xla::ElementType},

    //// Error from XLA Library
    #[error("XLA error: {err}")]
    XlaError { err: xla::Error },

    //// Error processing graph
    #[error("Unda internal graph processing error {msg}")]
    GraphError { msg: String }
}

pub type Result<T> = std::result::Result<T, Error>;