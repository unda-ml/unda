use std::{collections::HashMap, fmt::format};

use super::*;

use slotmap::SlotMap;

/// XLA computation graph context.
// TODO: rename this to something meaningful
pub struct Context {
    pub nodes: SlotMap<NodeIdentifier, Node>,
    pub(crate) constants: Vec<NodeIdentifier>,
    pub(crate) parameters: Vec<NodeIdentifier>,
    pub(crate) dependent_nodes: HashMap<NodeIdentifier, Vec<NodeIdentifier>>,
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

    #[error("Expected shape {0} to have size greater than shape {1} at {2}")]
    ExpectedGreaterSize(Shape, Shape, Callsite),

    #[error("Tried to call reshape_const on non-constant node at {0}")]
    NonConstantReshape(Callsite),

    #[error("Tried to call typecast_const on non-constant node at {0}")]
    NonConstantTypecast(Callsite),

    #[error("XLA internal error: {0}. Unless this is a device error, Unda should not produce internal XLA errors. Please create a github issue.")]
    Xla(#[from] xla::Error),

    #[error("Unda internal graph processing error {0}")]
    CompileError(#[from] CompileError),

    #[error("Unexpected Array Shape encountered {0}")]
    ShapeConversion(#[from] ShapeConversionError),

    #[error("Parameter \"{0}\" {1} already exists in the context at {2}")]
    DuplicateParameter(String, Callsite, Callsite),

    #[error("Tried to call Context::return more than once.")]
    MultipleReturns(),

    #[error("Operation is not differentiable, to use it as a constant in a differentiable computation, wrap it with Context::stop_gradient.")]
    NonDifferentiableOpError(Callsite),

    #[error("Type is not differentiable, differentiable types are F16, Bf16, F32, F64, C64, C128")]
    NonDifferentiableTypeError(Callsite),

    #[error("Expected integral type, got {0}. Integral types are S8, S16, S32, S64, U8, U16, U32, U64")]
    IntegralTypeError(xla::ElementType, Callsite),

    #[error("Expected real type, got {0}. Real types are F16, Bf16, F32, F64")]
    RealTypeError(xla::ElementType, Callsite),

    #[error("Expected floating point type, got {0}. Real types are F16, Bf16, F32, F64, C64, C128")]
    FPTypeError(xla::ElementType, Callsite),

    #[error("Expected tensor of rank {0}, got {1}")]
    RankError(usize, usize, Callsite),

    #[error("Invalid permutation passed to transpose. Expected permutation of length {0}, got {1}")]
    TransposeLenError(usize, usize, Callsite),
}

pub type Result<T> = std::result::Result<T, ContextError>;

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            constants: Vec::new(),
            parameters: Vec::new(),
            dependent_nodes: HashMap::new(),
        }
    }

    pub fn to_string(&self, input: NodeIdentifier) -> String {
        let input_node = &self.nodes[input];

        match input_node.operation.clone() {
            Operation::Constant(a) => format!("Constant {} {}", input_node.shape, a),
            Operation::Parameter(a) => format!("Parameter {} {}", input_node.shape, a),
            Operation::StopGradient(a) => {
                format!("StopGradient ({})", self.to_string(a))
            }
            Operation::Add(a, b) => format!("Add ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Sub(a, b) => format!("Sub ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Mul(a, b) => format!("Mul ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::MatMul(a, b) => format!("MatMul ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Div(a, b) => format!("Div ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Pow(a, b) => format!("Pow ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Neg(a) => format!("Neg ({})", self.to_string(a)),
            Operation::Exp(a) => format!("Exp ({})", self.to_string(a)),
            Operation::Log(a) => format!("Log ({})", self.to_string(a)),
            Operation::Transpose(a, b) => format!("Transpose: ({}) ({:?})", self.to_string(a), b),
            Operation::RngUniform(a, b, shape) => format!("RngUniform: ({}) ({}) ({})", self.to_string(a), self.to_string(b), shape),
            Operation::RngNormal(a, b, shape) => format!("RngNormal: ({}) ({}) ({})", self.to_string(a), self.to_string(b), shape),
            Operation::Equal(a, b) => {
                format!("LessThan ({}) ({})", self.to_string(a), self.to_string(b))
            }
            Operation::NotEqual(a, b) => {
                format!("NotEqual ({}) ({})", self.to_string(a), self.to_string(b))
            }
            Operation::LessThan(a, b) => {
                format!("LessThan ({}) ({})", self.to_string(a), self.to_string(b))
            }
            Operation::GreaterThan(a, b) => format!(
                "GreaterThan ({}) ({})",
                self.to_string(a),
                self.to_string(b)
            ),
            Operation::LessThanEq(a, b) => {
                format!("LessThanEq ({}) ({})", self.to_string(a), self.to_string(b))
            }
            Operation::GreaterThanEq(a, b) => format!(
                "GreaterThanEq ({}) ({})",
                self.to_string(a),
                self.to_string(b)
            ),
            Operation::Select {
                pred,
                on_true,
                on_false,
            } => format!(
                "Select ({}) ({}) ({})",
                self.to_string(pred),
                self.to_string(on_true),
                self.to_string(on_false)
            ),
            Operation::TypeCast(a, ty) => format!("TypeCast ({}) {}", self.to_string(a), ty),
            Operation::Reshape(a) => format!(
                "Reshape ({}) {}",
                self.to_string(a),
                self.nodes[input].shape
            ),
            Operation::SliceInDim {
                node,
                start,
                stop,
                stride,
                dim,
            } => format!(
                "SliceInDim ({}) {} {} {} {}",
                self.to_string(node),
                start,
                stop,
                stride,
                dim
            ),
            Operation::TileInDim { node, n_tiles, dim } => {
                format!("TileInDim ({}) {} {}", self.to_string(node), n_tiles, dim)
            }
            Operation::ZerosLike(node) => format!("ZerosLike {}", self.to_string(node)),
            Operation::OneHot(node) => format!(
                "OneHot ({}) {} {}",
                self.to_string(node),
                input_node.shape.sizes[1],
                input_node.dtype
            ),
            Operation::ReduceMax {
                node,
                dim,
            } => format!("ReduceMax {} {}", self.to_string(node), dim),
            Operation::ReduceArgmax {
                node,
                dim,
            } => format!("ReduceArgmax {} {}", self.to_string(node), dim),
            Operation::ReduceSum {
                node,
                dim,
            } => format!("ReduceSum {} {}", self.to_string(node), dim),
            Operation::ReduceMean {
                node,
                dim,
            } => format!("ReduceMean {} {}", self.to_string(node), dim),
        }
    }
}
