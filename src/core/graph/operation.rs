use super::*;
use std::fmt::{Display, Formatter, Result};
use strum_macros::EnumDiscriminants;

#[derive(Debug, Clone, EnumDiscriminants)]
pub enum Operation {
    Constant(ConstantBinding),
    Parameter(ParameterBinding),
    StopGradient(NodeIdentifier),
    Diff(NodeIdentifier, Parameter),
    Add(NodeIdentifier, NodeIdentifier),
    Mul(NodeIdentifier, NodeIdentifier),

    LessThan(NodeIdentifier, NodeIdentifier),
    GreaterThan(NodeIdentifier, NodeIdentifier),
    LessThanEq(NodeIdentifier, NodeIdentifier),
    GreaterThanEq(NodeIdentifier, NodeIdentifier),

    Select{ pred: NodeIdentifier, on_true: NodeIdentifier, on_false: NodeIdentifier },
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
