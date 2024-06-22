use super::*;
use std::{
    fmt::{Display, Formatter, Result},
    hash::Hash,
};
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

    Select {
        pred: NodeIdentifier,
        on_true: NodeIdentifier,
        on_false: NodeIdentifier,
    },

    TypeCast(NodeIdentifier, xla::ElementType),
    Reshape(NodeIdentifier),

    Transpose(NodeIdentifier, Vec<i64>),
    MatMul(NodeIdentifier, NodeIdentifier),
    SliceInDim {
        node: NodeIdentifier,
        start: i64,
        stop: i64,
        stride: i64,
        dim: i64,
    },
    TileInDim {
        node: NodeIdentifier,
        n_tiles: i64,
        dim: i64,
    },

    ZerosLike(NodeIdentifier),

    ReduceMax {
        node: NodeIdentifier,
        dim: i64,
    },
    ReduceSum {
        node: NodeIdentifier,
        dim: i64,
    },
    ReduceMean {
        node: NodeIdentifier,
        dim: i64,
    },
    ReduceArgmax {
        node: NodeIdentifier,
        dim: i64,
    },

    OneHot(NodeIdentifier),
    RngUniform(NodeIdentifier, NodeIdentifier, Shape),
    RngNormal(NodeIdentifier, NodeIdentifier, Shape)
}

impl Hash for Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Add(a, b)
            | Self::Mul(a, b)
            | Self::Sub(a, b)
            | Self::Div(a, b)
            | Self::NotEqual(a, b)
            | Self::Equal(a, b)
            | Self::LessThan(a, b)
            | Self::LessThanEq(a, b)
            | Self::GreaterThanEq(a, b)
            | Self::GreaterThan(a, b)
            | Self::Pow(a, b)
            | Self::MatMul(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            Self::TypeCast(a, ty) => {
                a.hash(state);
                (ty.primitive_type() as usize).hash(state);
            }
            Self::Constant(a) => {
                //This is a little silly but it should work tbh
                //Might want to redo this later
                a.to_string().hash(state);
            }
            Self::Parameter(a) => {
                a.hash(state);
            }
            Self::StopGradient(a) => {
                a.hash(state);
            }
            Self::Log(a) | Self::Exp(a) | Self::Reshape(a) | Self::ZerosLike(a) | Self::Neg(a) => {
                a.hash(state);
            }
            Self::Select {
                pred,
                on_true,
                on_false,
            } => {
                pred.hash(state);
                on_true.hash(state);
                on_false.hash(state);
            }
            Self::ReduceMax { node, dim }
            | Self::ReduceMean { node, dim }
            | Self::ReduceSum { node, dim }
            | Self::ReduceArgmax { node, dim } => {
                node.hash(state);
                dim.hash(state)
            }
            Self::OneHot(node) => node.hash(state),
            Self::Transpose(a, dim) => {
                a.hash(state);
                dim.hash(state);
            }
            Self::SliceInDim {
                node,
                start,
                stop,
                stride,
                dim,
            } => {
                node.hash(state);
                start.hash(state);
                stop.hash(state);
                stride.hash(state);
                dim.hash(state);
            }
            Self::TileInDim { node, n_tiles, dim } => {
                node.hash(state);
                n_tiles.hash(state);
                dim.hash(state);
            }
            Self::RngUniform(a, b, dim) 
            | Self::RngNormal(a, b, dim) => {
                a.hash(state);
                b.hash(state);
                dim.hash(state);
            }
        }
    }
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        match (&self, &other) {
            //Order not matering. Ex: 1 + 2 equals 2 + 1, but 1 / 2 doesnt equal 2 /1 so we can
            //check these separately
            (&Self::Mul(a, b), &Self::Mul(c, d))
            | (&Self::Equal(a, b), &Self::Equal(c, d))
            | (&Self::NotEqual(a, b), &Self::NotEqual(c, d))
            | (&Self::Add(a, b), &Self::Add(c, d)) => (a == c && b == d) || (a == d && b == c),
            //Order does matter, so div, sub, pow etc
            (&Self::Div(a, b), &Self::Div(c, d))
            | (&Self::Pow(a, b), &Self::Pow(c, d))
            | (&Self::LessThan(a, b), &Self::LessThan(c, d))
            | (&Self::GreaterThan(a, b), &Self::GreaterThan(c, d))
            | (&Self::GreaterThanEq(a, b), &Self::GreaterThanEq(c, d))
            | (&Self::LessThanEq(a, b), &Self::LessThanEq(c, d))
            | (&Self::MatMul(a, b), &Self::MatMul(c, d))
            | (&Self::Sub(a, b), &Self::Sub(c, d)) => a == c && b == d,
            (&Self::StopGradient(a), &Self::StopGradient(b))
            | (&Self::Neg(a), &Self::Neg(b))
            | (&Self::ZerosLike(a), &Self::ZerosLike(b))
            | (&Self::Exp(a), &Self::Exp(b))
            | (&Self::Reshape(a), &Self::Reshape(b))
            | (&Self::Log(a), &Self::Log(b)) => a == b,
            (&Self::Constant(a), &Self::Constant(b)) => a.to_string() == b.to_string(),
            (&Self::Parameter(a), &Self::Parameter(b)) => a == b,
            (
                &Self::Select {
                    pred,
                    on_true,
                    on_false,
                },
                &Self::Select {
                    pred: pred2,
                    on_true: on_true2,
                    on_false: on_false2,
                },
            ) => pred == pred2 && on_true == on_true2 && on_false == on_false2,
            (&Self::TypeCast(a, ty), &Self::TypeCast(b, ty2)) => a == b && ty == ty2,
            (&Self::RngUniform(a, b, shape), &Self::RngUniform(a2, b2, shape2)) => a == a2 && b == b2 && shape == shape2,
            (&Self::RngNormal(a, b, shape), &Self::RngNormal(a2, b2, shape2)) => a == a2 && b == b2 && shape == shape2,
            (&Self::Transpose(a, dim), &Self::Transpose(b, dim2)) => a == b && dim == dim2,
            (
                &Self::SliceInDim {
                    node,
                    start,
                    stop,
                    stride,
                    dim,
                },
                &Self::SliceInDim {
                    node: node2,
                    start: start2,
                    stop: stop2,
                    stride: stride2,
                    dim: dim2,
                },
            ) => {
                node == node2
                    && start == start2
                    && stop == stop2
                    && stride == stride2
                    && dim == dim2
            }
            (
                &Self::TileInDim { node, n_tiles, dim },
                &Self::TileInDim {
                    node: node2,
                    n_tiles: n_tiles2,
                    dim: dim2,
                },
            ) => node == node2 && n_tiles == n_tiles2 && dim == dim2,
            (
                &Self::ReduceMax { node, dim },
                &Self::ReduceMax {
                    node: node2,
                    dim: dim2,
                },
            )
            | (
                &Self::ReduceMean { node, dim },
                &Self::ReduceMean {
                    node: node2,
                    dim: dim2,
                },
            )
            | (
                &Self::ReduceSum { node, dim },
                &Self::ReduceSum {
                    node: node2,
                    dim: dim2,
                },
            ) => node == node2 && dim == dim2,
            _ => false,
        }
    }
}

impl Eq for Operation {}

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
