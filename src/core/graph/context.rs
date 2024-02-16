use self::operation::{
    ConstantBinding, Node, NodeIdentifier, Operation, Parameter, ParameterBinding,
};
use super::*;
use slotmap::SlotMap;
use std::collections::HashMap;

/// XLA computation graph context.
// TODO: rename this to something meaningful
pub struct Context {
    nodes: SlotMap<NodeIdentifier, Node>,
    param_indices: Vec<NodeIdentifier>,
    const_indices: Vec<NodeIdentifier>
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
            param_indices: Vec::new(),
            const_indices: Vec::new()
        }
    }

    pub fn scalar(&mut self, value: f32) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
                callsite: callsite!(1),
                dimension: Dimension::new(),
                operation: Operation::Constant(ConstantBinding { value: vec![value] }),
        });
        self.const_indices.push(node_id);
        node_id
    }

    pub fn vector<const N: usize>(&mut self, values: [f32; N]) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
                callsite: callsite!(1),
                dimension: Dimension::new(),
                operation: Operation::Constant(ConstantBinding { value: values.to_vec() }),
        });
        self.const_indices.push(node_id);
        node_id
    }

    pub fn matrix<const N: usize, const M: usize>(&mut self, values: [[f32; M]; N]) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            dimension: Dimension::of(N as u32).by(M as u32),
            operation: Operation::Constant(ConstantBinding {
                value: values.iter().flat_map(|f| f.iter()).copied().collect(),
            }),
        });
        self.const_indices.push(node_id);
        node_id
    }

    pub fn parameter<S: AsRef<str>>(&mut self, name: S) -> Parameter {
        let param = Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                // TODO: proper dimension here
                dimension: Dimension::new(),
                operation: Operation::Parameter(ParameterBinding {
                    name: name.as_ref().to_string(),
                }),
            }),
        };
        self.param_indices.push(param.node);
        param
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
    /// return true if there may be further changes needed
    fn autodiff<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        input: A,
        modification_limit: usize,
    ) -> bool {
        if modification_limit == 0 {
            return true;
        }
        let input_node = &self.nodes[input.into()];

        // traverse nodes until we find a Diff node or a leaf
        match input_node.operation {
            // leaf nodes mean no further processing
            Operation::Constant(_) => false,
            Operation::Parameter(_) => false,
            // operations mean we need to go deeper
            Operation::Add(a, b) => {
                let r = self.autodiff(a, modification_limit);
                self.autodiff(b, modification_limit - (r as usize)) || r
            }
            Operation::Mul(a, b) => {
                let r = self.autodiff(a, modification_limit);
                self.autodiff(b, modification_limit - (r as usize)) || r
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
                        true
                    }
                    Operation::Parameter(_) => {
                        // derivative of a parameter with respect to itself is one, and otherwise zero
                        self.nodes[input.into()].operation = Operation::Constant(ConstantBinding {
                            value: vec![(outer == outer_param.into()) as u32 as f32],
                        });
                        self.nodes[input.into()].dimension = Dimension::scalar();
                        true
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
                        self.autodiff(input, modification_limit - 1)
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
                        self.autodiff(input, modification_limit - 1)
                    }
                    Operation::Diff(inner, _) => {
                        // derivative of a derivative, apply the inner one first then try again on the outer.
                        let r = self.autodiff(inner, modification_limit);
                        self.autodiff(outer, modification_limit - (r as usize)) || r
                    }
                }
            }
        }
    }

    /// Folds constants in place by replacing any node whose both inputs are Constant
    /// with a Constant of the result of the operation. All existing references to
    /// the old node will still point to it once its replaced, and this process is
    /// repeated until there are no more nodes whose inputs are all constants.
    fn foldconsts<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        _input: A,
        modification_limit: usize,
    ) -> bool {
        if modification_limit == 0 {
            return true;
        }
        // TODO: implement this
        false
    }

    fn get_dependent_nodes<A: Into<NodeIdentifier> + Copy>(&self, a: A, dep_nodes: &mut HashMap<NodeIdentifier, Vec<NodeIdentifier>>) -> Result<(), String> {
        let this_node_id = a.into();
        let input_node = &self.nodes[this_node_id];
        match input_node.operation {
            Operation::Constant(_) => Ok(()),
            Operation::Parameter(_) => Ok(()),
            Operation::Diff(_, _) => Err("Found Diff Node during XLA conversion!".to_string()),
            Operation::Add(node1, node2) => {
                if dep_nodes.contains_key(&node1) {
                    dep_nodes.get(&node1).unwrap().push(this_node_id);
                } else {
                    dep_nodes.insert(node1, vec![this_node_id]);
                }
                if dep_nodes.contains_key(&node2) {
                    dep_nodes.get(&node2).unwrap().push(this_node_id);
                } else {
                    dep_nodes.insert(node2, vec![this_node_id]);
                }
                self.get_dependent_nodes(node1, dep_nodes);
                self.get_dependent_nodes(node2, dep_nodes)
            },
            Operation::Mul(node1, node2) => {
                if dep_nodes.contains_key(&node1) {
                    dep_nodes.get(&node1).unwrap().push(this_node_id);
                } else {
                    dep_nodes.insert(node1, vec![this_node_id]);
                }
                if dep_nodes.contains_key(&node2) {
                    dep_nodes.get(&node2).unwrap().push(this_node_id);
                } else {
                    dep_nodes.insert(node2, vec![this_node_id]);
                }
                self.get_dependent_nodes(node1, dep_nodes);
                self.get_dependent_nodes(node2, dep_nodes)
            }
        }

    }

    pub fn compile<A: Into<NodeIdentifier> + Copy>(&mut self, a: A, builder: xla::XlaBuilder) {
        // TODO: gate debug mode behind a feature flag

        //self.autodiff(a, usize::MAX);
        println!("{}", self.to_string(a));
        while self.autodiff(a, 1) {
            println!("{}", self.to_string(a));
        }

        //self.foldconsts(a, usize::MAX);
        while self.foldconsts(a, 1) {
            println!("{}", self.to_string(a));
        }

        // TODO: compile to XLA
        let mut dependent_nodes: HashMap<NodeIdentifier, Vec<NodeIdentifier>> = HashMap::new();
        self.get_dependent_nodes(a, &mut dependent_nodes);
    }

    pub fn to_string<A: Into<NodeIdentifier> + Copy>(&self, input: A) -> String {
        let input_node = &self.nodes[input.into()];

        match input_node.operation.clone() {
            Operation::Constant(a) => format!("Constant {} {}", input_node.dimension, a),
            Operation::Parameter(a) => format!("Parameter {} {}", input_node.dimension, a),
            Operation::Add(a, b) => format!("Add ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Mul(a, b) => format!("Mul ({}) ({})", self.to_string(a), self.to_string(b)),
            Operation::Diff(a, b) => format!("Diff ({}) {}", self.to_string(a), self.to_string(b)),
        }
    }
}
