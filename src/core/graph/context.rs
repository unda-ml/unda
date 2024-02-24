use super::*;
use slotmap::SlotMap;

/// XLA computation graph context.
// TODO: rename this to something meaningful
pub struct Context {
    pub(crate) nodes: SlotMap<NodeIdentifier, Node>,
    pub(crate) param_indices: Vec<NodeIdentifier>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ContextError {
    #[error("Mismatched shapes {0} {1} at {2}")]
    IncompatibleOperandShapes(Shape, Shape, Callsite),

    #[error("Mismatched types {0} {1} at {2}")]
    IncompatibleOperandTypes(xla::ElementType, xla::ElementType, Callsite),

    #[error("Tried to call reshape_const on non-constant node at {0}")]
    NonConstantReshape(Callsite),

    #[error("Tried to call typecast_const on non-constant node at {0}")]
    NonConstantTypecast(Callsite),

    #[error("XLA error: {0}")]
    Xla(#[from] xla::Error),

    #[error("Unda internal graph processing error {0}")]
    CompileError(#[from] CompileError),

    #[error("Unexpected Array Shape encountered {0}")]
    ShapeConversion(#[from] ShapeConversionError),

    #[error("Parameter \"{0}\" {1} already exists in the context at {2}")]
    DuplicateParameter(String, Callsite, Callsite),
}

pub type Result<T> = std::result::Result<T, ContextError>;

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            param_indices: Vec::new(),
        }
    }

    pub fn to_string<A: Into<NodeIdentifier> + Copy>(&self, input: A) -> String {
        let input_node = &self.nodes[input.into()];

        match input_node.operation.clone() {
            Operation::Constant(a) => format!("Constant {} {}", input_node.shape, a),
            Operation::Parameter(a) => format!("Parameter {} {}", input_node.shape, a),
            Operation::Add(a, b) => format!("Add ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Mul(a, b) => format!("Mul ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Diff(a, b) => format!("Diff ({}) {}", self.to_string(a), self.to_string(b)),
        }
    }
}
