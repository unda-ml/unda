use super::*;
use slotmap::new_key_type;
use std::fmt::{Display, Formatter, Result};

/// A node in the compute graph
pub struct Node {
    /// helps identify where in the user's source code this node originated
    // TODO: gate this so its not present at all in release builds
    pub(crate) callsite: Callsite,
    /// shape of the output of this node
    pub(crate) shape: Shape,
    /// the operation this node performs
    pub(crate) operation: Operation,
    //// output type of the operation
    pub(crate) dtype: xla::ElementType,
}

new_key_type! {
    pub struct NodeIdentifier;
}

impl Node {
    /// Identifies constant operation node for easier
    /// constant folding in context.rs
    pub(crate) fn is_const(&self) -> bool {
        return match self.operation {
            Operation::Constant(_) => true,
            _ => false,
        };
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{} {}", self.operation, self.callsite)
    }
}
