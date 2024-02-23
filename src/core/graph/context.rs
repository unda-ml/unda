use self::operation::{
    ConstantBinding, Node, NodeIdentifier, Operation, Parameter, ParameterBinding,
};
use self::shape::Shape;
use super::*;
use crate::core::error::Error::*;
use crate::core::error::Result;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use xla::FromRawBytes;

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

    pub fn scalar<T: xla::ArrayElement + xla::NativeType>(
        &mut self,
        value: T,
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let value = match xla::Literal::scalar(value).convert(dtype.primitive_type()) {
            Ok(v) => v,
            Err(e) => return Err(XlaError { err: e }),
        };
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::new(),
            operation: Operation::Constant(ConstantBinding { value: value }),
            dtype: dtype,
        });
        self.const_indices.push(node_id);
        Ok(node_id)
    }

    pub fn vector<T: xla::ArrayElement + xla::NativeType, const N: usize>(
        &mut self,
        values: [T; N],
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let value = match xla::Literal::vec1(&values).convert(dtype.primitive_type()) {
            Ok(v) => v,
            Err(e) => return Err(XlaError { err: e }),
        };
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::of(N as u16),
            operation: Operation::Constant(ConstantBinding { value: value }),
            dtype: T::TY,
        });
        self.const_indices.push(node_id);
        Ok(node_id)
    }

    pub fn matrix<T: xla::ArrayElement + xla::NativeType, const N: usize, const M: usize>(
        &mut self,
        values: [[T; M]; N],
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let vec = values
            .into_iter()
            .flat_map(|f| f.into_iter())
            .collect::<Vec<T>>();
        let slice = vec.as_slice();
        let value = match xla::Literal::vec1(slice).convert(dtype.primitive_type()) {
            Ok(v) => v,
            Err(e) => return Err(XlaError { err: e }),
        };
        let reshaped = match value.reshape(&[N as i64, M as i64]) {
            Ok(r) => r,
            Err(e) => return Err(XlaError { err: e }),
        };
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::of(N as u16).by(M as u16),
            operation: Operation::Constant(ConstantBinding { value: reshaped }),
            dtype: T::TY,
        });
        self.const_indices.push(node_id);
        Ok(node_id)
    }

    pub fn const_from_npy<T: AsRef<Path>>(&mut self, path: T) -> Result<NodeIdentifier> {
        match xla::Literal::read_npy(path, &()) {
            xla::Result::Err(e) => Err(XlaError { err: e }),
            xla::Result::Ok(l) => match l.shape() {
                Err(e) => Err(XlaError { err: e }),
                Ok(s) => match l.ty() {
                    Err(e) => Err(XlaError { err: e }),
                    Ok(t) => {
                        let s = Shape::from_xla_shape(s)?;
                        let node_id = self.nodes.insert(Node {
                            callsite: callsite!(1),
                            shape: s,
                            operation: Operation::Constant(ConstantBinding { value: l }),
                            dtype: t,
                        });
                        self.const_indices.push(node_id);
                        Ok(node_id)
                    }
                },
            },
        }
    }

    pub fn reshape_const(
        &mut self,
        const_id: NodeIdentifier,
        new_shape: &[u16],
    ) -> Result<NodeIdentifier> {
        let value = match &self.nodes[const_id].operation {
            Operation::Constant(b) => &b.value,
            _ => {
                return Err(GraphError {
                    msg: "Tried to call reshape_const on non-constant node.".to_string(),
                })
            }
        };
        let i64_vec = new_shape
            .into_iter()
            .map(|d| *d as i64)
            .collect::<Vec<i64>>();
        let i64_slice = i64_vec.as_slice();
        let new_value = match value.reshape(&i64_slice) {
            Ok(nv) => nv,
            Err(e) => return Err(XlaError { err: e }),
        };
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape {
                sizes: SmallVec::from_slice(new_shape),
            },
            operation: Operation::Constant(ConstantBinding { value: new_value }),
            dtype: self.nodes[const_id].dtype,
        });
        self.const_indices.push(node_id);
        Ok(node_id)
    }

    pub fn typecast_const(
        &mut self,
        const_id: NodeIdentifier,
        new_type: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let value = match &self.nodes[const_id].operation {
            Operation::Constant(b) => &b.value,
            _ => {
                return Err(GraphError {
                    msg: "Tried to call typecast_const on non-constant node.".to_string(),
                })
            }
        };
        let new_value = match value.convert(new_type.primitive_type()) {
            Ok(nv) => nv,
            Err(e) => return Err(XlaError { err: e }),
        };
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[const_id].shape.clone(),
            operation: Operation::Constant(ConstantBinding { value: new_value }),
            dtype: new_type,
        });
        self.const_indices.push(node_id);
        Ok(node_id)
    }

    pub fn parameter<S: AsRef<str>>(
        &mut self,
        name: S,
        shape: &[u16],
        dtype: xla::ElementType,
    ) -> Result<Parameter> {
        let name = name.as_ref().to_string();
        for node_id in self.param_indices.iter() {
            match &self.nodes[*node_id].operation {
                Operation::Parameter(binding) => {
                    if binding.name == name {
                        return Err(NameError { name });
                    }
                }
                _ => unreachable!(),
            };
        }
        let param = Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                // TODO: proper shape here
                // It makes sense to pass the shape of the parameter to its constructor
                shape: Shape {
                    sizes: SmallVec::from_slice(shape),
                },
                operation: Operation::Parameter(ParameterBinding { name }),
                dtype: dtype,
            }),
        };
        self.param_indices.push(param.node);
        Ok(param)
    }

    pub fn diff(&mut self, node: NodeIdentifier, with_respect_to: Parameter) -> NodeIdentifier {
        self.nodes.insert(Node {
            callsite: callsite!(1),
            // what should go here?
            shape: Shape::new(),
            operation: Operation::Diff(node, with_respect_to),
            dtype: self.nodes[with_respect_to.node].dtype,
        })
    }

    // TODO: use trait aliases for `Into<NodeIdentifier> + Copy`
    // when they get stablized: https://github.com/rust-lang/rust/issues/41517
    pub fn add<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let node_a = &self.nodes[a.into()];
        let node_b = &self.nodes[b.into()];

        if node_a.dtype != node_b.dtype {
            Err(TypeError {
                type1: node_a.dtype,
                type2: node_b.dtype,
            })
        } else {
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::Add(a.into(), b.into()),
                dtype: node_a.dtype,
            };
            // TODO: special case adding const zero
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ShapeError {
                    shape1: node_a.shape.clone(),
                    shape2: node_b.shape.clone(),
                })
            } else {
                Ok(self.nodes.insert(node))
            }
        }
    }

    pub fn mul<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let node_a = &self.nodes[a.into()];
        let node_b = &self.nodes[b.into()];

        if node_a.dtype != node_b.dtype {
            Err(TypeError {
                type1: node_a.dtype,
                type2: node_b.dtype,
            })
        } else {
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::Mul(a.into(), b.into()),
                dtype: node_a.dtype,
            };
            // TODO: special case adding const zero
            // TODO: support broadcastable shapes
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ShapeError {
                    shape1: node_a.shape.clone(),
                    shape2: node_b.shape.clone(),
                })
            } else {
                Ok(self.nodes.insert(node))
            }
        }
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
                let outer_dtype = outer_node.dtype.primitive_type();
                match outer_node.operation.clone() {
                    Operation::Constant(_) => {
                        // derivative of a constant with respect to anything is 0
                        self.nodes[input.into()].operation = Operation::Constant(ConstantBinding {
                            value: xla::Literal::create_from_shape(outer_dtype, &[]),
                        });
                        self.nodes[input.into()].shape = Shape::scalar();
                        true
                    }
                    Operation::Parameter(_) => {
                        // derivative of a parameter with respect to itself is one, and otherwise zero
                        self.nodes[input.into()].operation = Operation::Constant(ConstantBinding {
                            value: match xla::Literal::scalar(
                                (outer == outer_param.into()) as u32 as f32,
                            )
                            .convert(outer_dtype)
                            {
                                Ok(x) => x,
                                Err(e) => panic!("Error converting type of scalar literal."),
                            },
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
                            dtype: self.nodes[a].dtype,
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Diff(b, outer_param),
                            dtype: self.nodes[b].dtype,
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
                            dtype: self.nodes[a].dtype,
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Diff(b, outer_param),
                            dtype: self.nodes[b].dtype,
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
                            dtype: self.nodes[a].dtype,
                        };
                        let prod_b_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input.into()].callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Mul(a, diff_b),
                            dtype: self.nodes[b].dtype,
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

                if node_a.is_const() && node_b.is_const() {
                    //TODO: Do replacement
                }
                false
            }
            Operation::Mul(a, b) => {
                let node_a = &self.nodes[a];
                let node_b = &self.nodes[b];

                if node_a.is_const() && node_b.is_const() {
                    //TODO: Do replacement
                }
                false
            }
            _ =>
            //TODO: Not fully sure if const folding needs to happen when the
            //operation isn't addition or multiplication, returnign false
            //if the operation isn't either of these for now, but definitely
            //let me know if this should be other behavior
            {
                false
            }
        };
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
            return 0;
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

    fn get_dependent_nodes<A: Into<NodeIdentifier> + Copy>(
        &self,
        a: A,
        dep_nodes: &mut HashMap<NodeIdentifier, Vec<NodeIdentifier>>,
    ) -> Result<()> {
        let this_node_id = a.into();
        let input_node = &self.nodes[this_node_id];
        match input_node.operation {
            Operation::Constant(_) => Ok(()),
            Operation::Parameter(_) => Ok(()),
            Operation::Diff(_, _) => Err(GraphError {
                msg: "Found Diff Node during XLA conversion!".to_string(),
            }),
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
                self.get_dependent_nodes(node1, dep_nodes)?;
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
                self.get_dependent_nodes(node1, dep_nodes)?;
                self.get_dependent_nodes(node2, dep_nodes)
            }
        }
    }

    pub fn compile<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        name: &str,
        client: &xla::PjRtClient,
    ) -> Result<xla::PjRtLoadedExecutable> {
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
        let builder = xla::XlaBuilder::new(name);
        // Get the bottom-up dependencies of the compute graph
        let mut dependent_nodes: HashMap<NodeIdentifier, Vec<NodeIdentifier>> = HashMap::new();
        self.get_dependent_nodes(a, &mut dependent_nodes)?;

        // Prepare to loop through the unda compute graph and construct the XLA compute graph
        let mut xla_op_slotmap: SlotMap<NodeIdentifier, xla::XlaOp> = SlotMap::with_key();
        let mut unda_op_queue: VecDeque<NodeIdentifier> = VecDeque::new();
        let mut unda_xla_map: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        let mut covered_ops: HashSet<NodeIdentifier> = HashSet::new();

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
                node.dtype,
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
        for unda_id in self.const_indices.iter() {
            let node = &self.nodes[*unda_id];

            let const_val: &xla::Literal = match &node.operation {
                Operation::Constant(const_binding) => &const_binding.value,
                _ => panic!("Constant indices pointed to a non-constant node!"),
            };

            let maybe_xla_const = builder.constant_literal(const_val);
            let xla_const = match maybe_xla_const {
                Ok(c) => c,
                Err(e) => return Err(XlaError { err: e }),
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

            match dependent_nodes.get(&unda_id) {
                None => continue,
                Some(dependent_ops) => {
                    for dependent_op in dependent_ops {
                        if !covered_ops.contains(dependent_op) {
                            match self.nodes[*dependent_op].operation {
                                Operation::Parameter(_) => {
                                    return Err(GraphError {
                                        msg: "Parameter found as dependent node!".to_string(),
                                    })
                                }
                                Operation::Constant(_) => {
                                    return Err(GraphError {
                                        msg: "Constant found as dependent node!".to_string(),
                                    })
                                }
                                Operation::Diff(_, _) => {
                                    return Err(GraphError {
                                        msg: "Diff node found during XLA conversion!".to_string(),
                                    })
                                }

                                Operation::Mul(node1, node2) => {
                                    if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                                        && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                                    {
                                        let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                            .mul_(&xla_op_slotmap[unda_xla_map[&node2]]).map_err(|err| XlaError{err})?;
                                        let xla_id = xla_op_slotmap.insert(xla_op);
                                        unda_xla_map.insert(*dependent_op, xla_id);
                                        unda_op_queue.push_back(*dependent_op);
                                        covered_ops.insert(*dependent_op);
                                    }
                                }

                                Operation::Add(node1, node2) => {
                                    if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                                        && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                                    {
                                        let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                            .add_(&xla_op_slotmap[unda_xla_map[&node2]]).map_err(|err| XlaError{err})?;
                                        let xla_id = xla_op_slotmap.insert(xla_op);
                                        unda_xla_map.insert(*dependent_op, xla_id);
                                        unda_op_queue.push_back(*dependent_op);
                                        covered_ops.insert(*dependent_op);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let xla_computation = match xla_op_slotmap[unda_xla_map[&a.into()]].build() {
            Ok(c) => c,
            Err(_) => panic!("XLA internal build error"),
        };

        match xla_computation.compile(client) {
            Ok(e) => Ok(e),
            Err(e) => Err(XlaError { err: e }),
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

#[cfg(test)]
mod tests {
    use xla::FromRawBytes;
    #[test]
    fn test_mul_add_scalar_consts_and_params() {
        let mut ctx = super::Context::new();

        let three = ctx.scalar(3, xla::ElementType::F32).expect("three");

        let x = ctx.parameter("x", &[], xla::ElementType::F32).expect("x");
        let y = ctx.parameter("y", &[], xla::ElementType::F32).expect("y");

        let product = ctx.mul(x, three).expect("product");
        let sum = ctx.add(product, y).expect("sum");

        // output XLA
        // client must be exposed to the user, it is very nice to control device, memory fraction, and pre-allocation
        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let name = "test";
        let executable = ctx.compile(sum, &name, &client).expect("executable");

        let x_input = xla::Literal::scalar(2f32);
        let y_input = xla::Literal::scalar(3f32);
        // args are just provided in the order they are defined, would be nice to pass a dict or something
        // a pjrtbuffer is just an array slice on some device
        // but im not sure why its a nested vector instead of just one vector
        let device_result = executable.execute(&[x_input, y_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let rust_result = host_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);

        assert_eq!(rust_result[0], 9f32);
    }

    #[test]
    fn test_vector_matrix_bf16() {
        let mut ctx = super::Context::new();

        let foo = ctx.vector([1, 2, 3], xla::ElementType::Bf16).expect("foo");
        let bar = ctx
            .matrix([[4, 5, 6], [7, 8, 9], [10, 11, 12]], xla::ElementType::Bf16)
            .expect("bar");

        let baz = ctx.reshape_const(foo, &[1, 3]).expect("baz");
        let barbaz = ctx.mul(bar, baz).expect("barbaz");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let executable = ctx.compile(barbaz, "test", &client).expect("executable");

        let device_result = executable.execute::<xla::Literal>(&[]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let f32_result = host_result
            .convert(xla::ElementType::F32.primitive_type())
            .expect("f32 conversion");
        let rust_result = f32_result.to_vec::<f32>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[4f32, 10f32, 18f32, 7f32, 16f32, 27f32, 10f32, 22f32, 36f32]);
    }

    #[test]
    fn test_npy_loading() {
        let mut ctx = super::Context::new();

        let my_const = ctx.const_from_npy("test.npy").expect("my_const");
        println!("{}", ctx.nodes[my_const].dtype);
        let my_param = ctx
            .parameter("my_param", &[2, 2], xla::ElementType::S64)
            .expect("my_param");

        let sum = ctx.add(my_const, my_param).expect("sum");

        let client = xla::PjRtClient::gpu(0.7, false).expect("client");
        let executable = ctx.compile(sum, "test", &client).expect("executable");

        let my_param_input = xla::Literal::read_npy("test.npy", &()).expect("my_param_input");

        let device_result = executable.execute(&[my_param_input]).expect("execute");
        let host_result = device_result[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let rust_result = host_result.to_vec::<i64>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result.as_slice(), &[0, 2, 2, 0]);
    }
}
