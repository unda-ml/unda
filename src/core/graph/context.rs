use std::collections::HashMap;

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
        input: A,
        modification_limit: usize,
    ) -> bool {
        if modification_limit == 0 {
            return true;
        }
        // TODO: implement this
        let input_node = &self.nodes[input.into()];
        return match input_node.operation {
            Operation::Add(a, b) => {
                let node_a = &self.nodes[a];
                let node_b = &self.nodes[b];

                if node_a.is_const() && node_b.is_const(){
                    //TODO: Do replacement
                }
                false
            },
            Operation::Mul(a, b) => {
                let node_a = &self.nodes[a];
                let node_b = &self.nodes[b];

                if node_a.is_const() && node_b.is_const(){
                    //TODO: Do replacement
                }
                false
            },
            _ => //TODO: Not fully sure if const folding needs to happen when the 
                 //operation isn't addition or multiplication, returnign false
                 //if the operation isn't either of these for now, but definitely
                 //let me know if this should be other behavior
                 false
        }
    }

    /// Traverses graph context, building a hashmap of Node -> NodeIdentifier pairs
    /// If a duplicate Node is found, we can reference the other NodeIdentifier with
    /// the already existant node instead of having duplicates
    /// Note from Ro's email:
    /// make sure to update entry for the modified node, as the hash will change. 
    /// do not include callsite when calculating the hash.
    /// Returns a count of how many duplicates were removed, could be used to 
    /// debug print "removed {n} duplicates during CTE"
    /// TODO: Is u8 appropriate here?
    fn extract_common_terms(&mut self) -> u16 {
        if self.nodes.len() <= 1 {
            return 0
        }
        let mut node_map: HashMap<String, NodeIdentifier> = HashMap::new();
        let mut sum: u16 = 0;
        for (mut identifier, node) in self.nodes.iter_mut() {
            //TODO: Build a HashMap out of all nodes, check if a node already 'exists'
            //If node exists, remove all references to its NodeIdentifier and replace with the
            //prexisting NodeIdentifier
        }
        sum
    }

    pub fn compile<A: Into<NodeIdentifier> + Copy>(&mut self, a: A) {
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

        let cte_count = self.extract_common_terms();
        println!("Extracted {} common terms", cte_count);

        // TODO: compile to XLA
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
