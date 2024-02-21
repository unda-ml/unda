use super::{Callsite, Shape};
use slotmap::new_key_type;
use std::fmt::{Display, Formatter, Result};
use strum_macros::EnumDiscriminants;

/// A node in the compute graph
pub struct Node {
    /// helps identify where in the user's source code this node originated
    // TODO: gate this so its not present at all in release builds
    pub(crate) callsite: Callsite,
    /// shape of the output of this node
    pub(crate) shape: Shape,
    /// the operation this node performs
    pub(crate) operation: Operation,
}

impl Node {
    /// Identifies constant operation node for easier 
    /// constant folding in context.rs
    pub(crate) fn is_const(&self) -> bool {
        return match self.operation{
            Operation::Constant(_) => true,
            _ => false
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{} {}", self.operation, self.callsite)
    }
}

#[derive(Debug, Clone)]
pub struct ParameterBinding {
    // TODO: store something meaningful to XLA here
    pub(crate) name: String,
}

impl Display for ParameterBinding {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone)]
pub struct ConstantBinding {
    /// unstructured float data. only makes sense combined with Node::dimension
    pub(crate) value: Vec<f32>,
}

impl Display for ConstantBinding {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if self.value.is_empty() {
            write!(f, "0")?;
            return Ok(());
        }
        write!(f, "{}", self.value[0])?;
        // TODO: proper matrix printing?
        if self.value.len() > 1 {
            write!(f, "..")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, EnumDiscriminants)]
pub enum Operation {
    Constant(ConstantBinding),
    Parameter(ParameterBinding),
    Add(NodeIdentifier, NodeIdentifier),
    Mul(NodeIdentifier, NodeIdentifier),
    Diff(NodeIdentifier, Parameter),
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

new_key_type! {
    pub struct NodeIdentifier;
}

/// Wraps a NodeIdentifier to have type-safe autodiff.
#[derive(Debug, Clone, Copy)]
pub struct Parameter {
    pub(crate) node: NodeIdentifier,
}

impl From<Parameter> for NodeIdentifier {
    fn from(value: Parameter) -> Self {
        value.node
    }
}
