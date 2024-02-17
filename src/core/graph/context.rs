use self::operation::{
    ConstantBinding, Node, NodeIdentifier, Operation, Parameter, ParameterBinding,
};
use self::shape::Shape;
use super::*;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::{HashMap, VecDeque};
use xla::XlaOp;

/// XLA computation graph context.
// TODO: rename this to something meaningful
pub struct Context {
    nodes: SlotMap<NodeIdentifier, Node>,
    param_indices: Vec<NodeIdentifier>,
    const_indices: Vec<NodeIdentifier>,
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
            const_indices: Vec::new(),
        }
    }

    pub fn scalar(&mut self, value: f32) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::new(),
            operation: Operation::Constant(ConstantBinding { value: vec![value] }),
        });
        self.const_indices.push(node_id);
        node_id
    }

    pub fn vector<const N: usize>(&mut self, values: [f32; N]) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::of(N as u16),
            operation: Operation::Constant(ConstantBinding {
                value: values.to_vec(),
            }),
        });
        self.const_indices.push(node_id);
        node_id
    }

    pub fn matrix<const N: usize, const M: usize>(
        &mut self,
        values: [[f32; M]; N],
    ) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::of(N as u16).by(M as u16),
            operation: Operation::Constant(ConstantBinding {
                value: values.iter().flat_map(|f| f.iter()).copied().collect(),
            }),
        });
        self.const_indices.push(node_id);
        node_id
    }

    pub fn parameter<S: AsRef<str>>(&mut self, name: S, shape: SmallVec<[u16; 4]>) -> Parameter {
        let param = Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                // TODO: proper shape here
                // It makes sense to pass the shape of the parameter to its constructor
                shape: Shape { sizes: shape },
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
            // what should go here?
            shape: Shape::new(),
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
            // need to implement automatic shape broadcasting
            shape: Shape::new(),
            operation: Operation::Add(a.into(), b.into()),
        };
        // TODO: special case adding const zero
        if node_a.shape != node_b.shape {
            eprintln!(
                "Shape mismatch {} vs {} at: {}",
                node_a.shape, node_b.shape, node
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
            // need to implement automatic shape broadcasting
            shape: Shape::new(),
            operation: Operation::Mul(a.into(), b.into()),
        };
        // TODO: check shapeal compatibility correctly
        if node_a.shape != node_b.shape {
            eprintln!(
                "Shape mismatch {} vs {} at: {}",
                node_a.shape, node_b.shape, node
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
                        self.nodes[input.into()].shape = Shape::scalar();
                        true
                    }
                    Operation::Parameter(_) => {
                        // derivative of a parameter with respect to itself is one, and otherwise zero
                        self.nodes[input.into()].operation = Operation::Constant(ConstantBinding {
                            value: vec![(outer == outer_param.into()) as u32 as f32],
                        });
                        self.nodes[input.into()].shape = Shape::scalar();
                        true
                    }
                    Operation::Add(a, b) => {
                        // derivative of a sum is the sum of derivatives
                        // Diff (Sum a b) x = Sum (Diff a x) (Diff b x)
                        let diff_a_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[a].shape.clone(),
                            operation: Operation::Diff(a, outer_param),
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
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
                            shape: self.nodes[a].shape.clone(),
                            operation: Operation::Diff(a, outer_param),
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Diff(b, outer_param),
                        };
                        // propagate original Mul callsite to the new Add node
                        self.nodes[input.into()].callsite = outer_node.callsite.clone();
                        let diff_a = self.nodes.insert(diff_a_node);
                        let diff_b = self.nodes.insert(diff_b_node);
                        let prod_a_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input.into()].callsite.clone(),
                            shape: self.nodes[a].shape.clone(),
                            operation: Operation::Mul(diff_a, b),
                        };
                        let prod_b_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input.into()].callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
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

    fn get_dependent_nodes<A: Into<NodeIdentifier> + Copy>(
        &self,
        a: A,
        dep_nodes: &mut HashMap<NodeIdentifier, Vec<NodeIdentifier>>,
    ) -> Result<(), String> {
        let this_node_id = a.into();
        let input_node = &self.nodes[this_node_id];
        match input_node.operation {
            Operation::Constant(_) => Ok(()),
            Operation::Parameter(_) => Ok(()),
            Operation::Diff(_, _) => Err("Found Diff Node during XLA conversion!".to_string()),
            Operation::Add(node1, node2) => {
                if dep_nodes.contains_key(&node1) {
                    dep_nodes.entry(node1).and_modify(|v| v.push(this_node_id));
                } else {
                    dep_nodes.insert(node1, vec![this_node_id]);
                }
                if dep_nodes.contains_key(&node2) {
                    dep_nodes.entry(node2).and_modify(|v| v.push(this_node_id));
                } else {
                    dep_nodes.insert(node2, vec![this_node_id]);
                }
                self.get_dependent_nodes(node1, dep_nodes);
                self.get_dependent_nodes(node2, dep_nodes)
            }
            Operation::Mul(node1, node2) => {
                if dep_nodes.contains_key(&node1) {
                    dep_nodes.entry(node1).and_modify(|v| v.push(this_node_id));
                } else {
                    dep_nodes.insert(node1, vec![this_node_id]);
                }
                if dep_nodes.contains_key(&node2) {
                    dep_nodes.entry(node2).and_modify(|v| v.push(this_node_id));
                } else {
                    dep_nodes.insert(node2, vec![this_node_id]);
                }
                self.get_dependent_nodes(node1, dep_nodes);
                self.get_dependent_nodes(node2, dep_nodes)
            }
        }
    }

    pub fn compile<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        name: &str,
        client: &xla::PjRtClient,
    ) -> xla::PjRtLoadedExecutable {
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
        let builder = xla::XlaBuilder::new(name);
        // Get the bottom-up dependencies of the compute graph
        let mut dependent_nodes: HashMap<NodeIdentifier, Vec<NodeIdentifier>> = HashMap::new();
        self.get_dependent_nodes(a, &mut dependent_nodes);

        // Prepare to loop through the unda compute graph and construct the XLA compute graph
        let mut xla_op_slotmap: SlotMap<NodeIdentifier, xla::XlaOp> = SlotMap::with_key();
        let mut unda_op_queue: VecDeque<NodeIdentifier> = VecDeque::new();
        let mut unda_xla_map: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();

        // declare parameters with the XLA builder
        for (i, unda_id) in self.param_indices.iter().enumerate() {
            let node = &self.nodes[*unda_id];

            let shape = node
                .shape
                .sizes
                .iter()
                .map(|d| *d as i64)
                .collect::<SmallVec<[i64; 4]>>();

            let param_name: &String = match &node.operation {
                Operation::Parameter(param_binding) => &param_binding.name,
                _ => panic!("Parameter indices pointed to a non-parameter node!"),
            };

            let xla_param = match builder.parameter(
                i as i64,
                xla::ElementType::F32,
                &shape.as_slice(),
                &param_name,
            ) {
                Ok(p) => p,
                Err(_) => panic!("XLA builder failed to declare parameter."),
            };

            let xla_id = xla_op_slotmap.insert(xla_param);
            unda_xla_map.insert(*unda_id, xla_id);
            unda_op_queue.push_back(*unda_id);
        }

        // Initialize constants for the XLA builder
        // >1 dimensions not yet supported for constants
        for (i, unda_id) in self.const_indices.iter().enumerate() {
            let node = &self.nodes[*unda_id];

            let const_val: &Vec<f32> = match &node.operation {
                Operation::Constant(const_binding) => &const_binding.value,
                _ => panic!("Parameter indices pointed to a non-parameter node!"),
            };

            // this might be necessary?
            let sizes = &node.shape.sizes;
            let maybe_xla_const = match sizes.len() {
                0 => builder.constant_r0(const_val[0]),
                1 => builder.constant_r1(&const_val),
                _ => panic!("Multidimensional constants not yet implemented!"),
            };
            let xla_const = match maybe_xla_const {
                Ok(c) => c,
                Err(_) => panic!("XLA builder failed to declare constant."),
            };
            let xla_id = xla_op_slotmap.insert(xla_const);
            unda_xla_map.insert(*unda_id, xla_id);
            unda_op_queue.push_back(*unda_id);
        }

        // using the set of bottom up dependencies we constructed
        // loop through a queue of all operations in the context
        // and add them to the XLA context
        while unda_op_queue.len() > 0 {
            let unda_id = unda_op_queue.pop_front().unwrap();

            for dependent_op in dependent_nodes.get(&unda_id).unwrap() {
                match self.nodes[*dependent_op].operation {
                    Operation::Parameter(_) => panic!("Parameter found as dependent node!"),
                    Operation::Constant(_) => panic!("Constant found as dependent node!"),
                    Operation::Diff(_, _) => panic!("Diff node found during XLA conversion!"),

                    Operation::Mul(node1, node2) => {
                        let maybe_xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                            .mul_(&xla_op_slotmap[unda_xla_map[&node2]]);
                        let xla_op = match maybe_xla_op {
                            Ok(x) => x,
                            Err(_) => panic!("Failed on multiplication node."),
                        };
                        let xla_id = xla_op_slotmap.insert(xla_op);
                        unda_xla_map.insert(*dependent_op, xla_id);
                        unda_op_queue.push_back(*dependent_op);
                    }

                    Operation::Add(node1, node2) => {
                        let maybe_xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                            .add_(&xla_op_slotmap[unda_xla_map[&node2]]);
                        let xla_op = match maybe_xla_op {
                            Ok(x) => x,
                            Err(_) => panic!("Failed on addition node."),
                        };
                        let xla_id = xla_op_slotmap.insert(xla_op);
                        unda_xla_map.insert(*dependent_op, xla_id);
                        unda_op_queue.push_back(*dependent_op);
                    }
                }
            }
        }

        let xla_computation = match xla_op_slotmap[unda_xla_map[&a.into()]].build() {
            Ok(c) => c,
            Err(_) => panic!("Internal XLA build error")
        };

        match xla_computation.compile(client) {
            Ok(e) => e,
            Err(_) => panic!("XLA internal compile error!")
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
