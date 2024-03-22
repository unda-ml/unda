use super::*;
use std::fmt::{Display, Formatter, Result};
use strum_macros::EnumDiscriminants;

#[derive(Debug, Clone, EnumDiscriminants)]
pub enum Operation {
    Constant(ConstantBinding),
    Parameter(String),
    StopGradient(NodeIdentifier),
    Add(NodeIdentifier, NodeIdentifier),
    Sub(NodeIdentifier, NodeIdentifier),
    Mul(NodeIdentifier, NodeIdentifier),
    Div(NodeIdentifier, NodeIdentifier),
    Pow(NodeIdentifier, NodeIdentifier),
    Neg(NodeIdentifier),
    Log(NodeIdentifier),
    Exp(NodeIdentifier),

    Equal(NodeIdentifier, NodeIdentifier),
    NotEqual(NodeIdentifier, NodeIdentifier),
    LessThan(NodeIdentifier, NodeIdentifier),
    GreaterThan(NodeIdentifier, NodeIdentifier),
    LessThanEq(NodeIdentifier, NodeIdentifier),
    GreaterThanEq(NodeIdentifier, NodeIdentifier),

    Select{ pred: NodeIdentifier, on_true: NodeIdentifier, on_false: NodeIdentifier },

    TypeCast(NodeIdentifier, xla::ElementType),
    Reshape(NodeIdentifier),

    Transpose(NodeIdentifier, Vec<i64>),
    MatMul(NodeIdentifier, NodeIdentifier),
    SliceInDim{ node: NodeIdentifier, start: i64, stop: i64, stride: i64, dim: i64 },
    TileInDim{ node: NodeIdentifier, n_tiles: i64, dim: i64 },

    ZerosLike(NodeIdentifier),

    ReduceMax{ node: NodeIdentifier, dim: i64, },
    ReduceSum{ node: NodeIdentifier, dim: i64, },
    // TODO: This might not behave well for integral types! Figure out behavior.
    ReduceMean{ node: NodeIdentifier, dim: i64, },
    ReduceArgmax{ node: NodeIdentifier, dim: i64, },

    OneHot(NodeIdentifier),
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use Operation::*;
        let d = OperationDiscriminants::from(self);
        match self {
            Constant(constant) => write!(f, "{} {}", d, constant),
            Parameter(parameter) => write!(f, "{} {}", d, parameter),
            _ => write!(f, "{}", d),
        }
    }
}

impl Display for OperationDiscriminants {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{:?}", self)
    }
}
