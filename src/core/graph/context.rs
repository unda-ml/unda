use self::operation::{
    ConstantBinding, Node, NodeIdentifier, Operation, Parameter, ParameterBinding,
};
use super::*;
use slotmap::SlotMap;

/// XLA computation graph context.
// TODO: rename this to something meaningful
pub struct Context {
    nodes: SlotMap<NodeIdentifier, Node>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
        }
    }

    pub fn scalar(&mut self, value: f32) -> Parameter {
        Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                dimension: Dimension::new(),
                operation: Operation::Constant(ConstantBinding { value: vec![value] }),
            }),
        }
    }

    pub fn vector<const N: usize>(&mut self, values: [f32; N]) -> Parameter {
        Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                dimension: Dimension::of(N as u32),
                operation: Operation::Constant(ConstantBinding {
                    value: values.to_vec(),
                }),
            }),
        }
    }

    pub fn matrix<const N: usize, const M: usize>(&mut self, values: [[f32; M]; N]) -> Parameter {
        Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                dimension: Dimension::of(N as u32).by(M as u32),
                operation: Operation::Constant(ConstantBinding {
                    value: values.iter().flat_map(|f| f.iter()).copied().collect(),
                }),
            }),
        }
    }

    pub fn parameter<S: AsRef<str>>(&mut self, name: S) -> Parameter {
        Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                // TODO: proper dimension here
                dimension: Dimension::new(),
                operation: Operation::Parameter(ParameterBinding {
                    name: name.as_ref().to_string(),
                }),
            }),
        }
    }

    pub fn diff(&mut self, node: NodeIdentifier, with_respect_to: Parameter) -> NodeIdentifier {
        self.nodes.insert(Node {
            callsite: callsite!(1),
            dimension: Dimension::new(),
            operation: Operation::Diff(node, with_respect_to),
        })
    }

    // TODO: use trait aliases for `Into<NodeIdentifier> + Copy`
    // when they get stablized: https://github.com/rust-lang/rust/issues/41517
    pub fn add<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> NodeIdentifier {
        let node_a = &self.nodes[a.into()];
        let node_b = &self.nodes[b.into()];
        let node = Node {
            callsite: callsite!(1),
            dimension: Dimension::new(),
            operation: Operation::Add(a.into(), b.into()),
        };
        if node_a.dimension != node_b.dimension {
            eprintln!(
                "Dimension mismatch {} vs {} at: {}",
                node_a.dimension, node_b.dimension, node
            );
        }
        self.nodes.insert(node)
    }

    pub fn mul<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> NodeIdentifier {
        let node_a = &self.nodes[a.into()];
        let node_b = &self.nodes[b.into()];
        let node = Node {
            callsite: callsite!(1),
            dimension: Dimension::new(),
            operation: Operation::Mul(a.into(), b.into()),
        };
        // TODO: check dimensional compatibility correctly
        if node_a.dimension != node_b.dimension {
            eprintln!(
                "Dimension mismatch {} vs {} at: {}",
                node_a.dimension, node_b.dimension, node
            );
        }
        self.nodes.insert(node)
    }

    pub fn compile<A: Into<NodeIdentifier>>(&mut self, a: A) {
        let node = &self.nodes[a.into()];

        // TODO: do shit instead of erroring lmao
        // walk the ast and propagate constants/autodiff
        println!("Error: {}", node);
    }
}
