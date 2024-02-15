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
        // TODO: special case adding const zero
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

    /// expand all Diff nodes into analytic derivatives
    fn autodiff<A: Into<NodeIdentifier> + Copy>(&mut self, input: A) {
        let input_node = &self.nodes[input.into()];

        // traverse nodes until we find a Diff node or a leaf
        match input_node.operation {
            // leaf nodes mean no further processing
            Operation::Constant(_) => (),
            Operation::Parameter(_) => (),
            // operations mean we need to go deeper
            Operation::Add(a, b) => {
                self.autodiff(a);
                self.autodiff(b);
            }
            Operation::Mul(a, b) => {
                self.autodiff(a);
                self.autodiff(b);
            }
            // finally a Diff node, lets distribute it
            Operation::Diff(outer, outer_param) => {
                let outer_node = &self.nodes[outer];
                match outer_node.operation.clone() {
                    Operation::Constant(_) => {
                        // derivative of a constant with respect to anything is 0
                        self.nodes[input.into()].operation =
                            Operation::Constant(ConstantBinding { value: vec![] });
                        self.nodes[input.into()].dimension = Dimension::scalar();
                    }
                    Operation::Parameter(_) => {
                        // derivative of a parameter with respect to itself is one, and otherwise zero
                        self.nodes[input.into()].operation = Operation::Constant(ConstantBinding {
                            value: vec![(outer == outer_param.into()) as u32 as f32],
                        });
                        self.nodes[input.into()].dimension = Dimension::scalar();
                    }
                    Operation::Add(a, b) => {
                        // derivative of a sum is the sum of derivatives
                        // Diff (Sum a b) x = Sum (Diff a x) (Diff b x)
                        let diff_a_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            dimension: self.nodes[a].dimension.clone(),
                            operation: Operation::Diff(a, outer_param),
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            dimension: self.nodes[b].dimension.clone(),
                            operation: Operation::Diff(b, outer_param),
                        };
                        // propagate original Add callsite to the new Add node
                        self.nodes[input.into()].callsite = outer_node.callsite.clone();
                        let diff_a = self.nodes.insert(diff_a_node);
                        let diff_b = self.nodes.insert(diff_b_node);
                        self.nodes[input.into()].operation = Operation::Add(diff_a, diff_b);
                        // rerun autodiff on the node we replaced
                        self.autodiff(input);
                    }
                    Operation::Mul(a, b) => {
                        // product rule
                        // Diff (Mul a b) x = Sum (Mul (Diff a x) b) (Mul a (Diff b x))
                        let diff_a_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            dimension: self.nodes[a].dimension.clone(),
                            operation: Operation::Diff(a, outer_param),
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            dimension: self.nodes[b].dimension.clone(),
                            operation: Operation::Diff(b, outer_param),
                        };
                        // propagate original Mul callsite to the new Add node
                        self.nodes[input.into()].callsite = outer_node.callsite.clone();
                        let diff_a = self.nodes.insert(diff_a_node);
                        let diff_b = self.nodes.insert(diff_b_node);
                        let prod_a_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input.into()].callsite.clone(),
                            dimension: self.nodes[a].dimension.clone(),
                            operation: Operation::Mul(diff_a, b),
                        };
                        let prod_b_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input.into()].callsite.clone(),
                            dimension: self.nodes[b].dimension.clone(),
                            operation: Operation::Mul(a, diff_b),
                        };
                        let prod_a = self.nodes.insert(prod_a_node);
                        let prod_b = self.nodes.insert(prod_b_node);
                        self.nodes[input.into()].operation = Operation::Add(prod_a, prod_b);
                        // rerun autodiff on the node we replaced
                        self.autodiff(input);
                    }
                    Operation::Diff(inner, _) => {
                        // derivative of a derivative, apply the inner one first then try again on the outer.
                        self.autodiff(inner);
                        self.autodiff(outer);
                    }
                }
            }
        }
    }

    pub fn compile<A: Into<NodeIdentifier> + Copy>(&mut self, a: A) {
        self.autodiff(a);
        // TODO: propagate constants
        // TODO: compile to XLA
    }
}
