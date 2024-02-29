use super::*;
use serde_json::de;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(thiserror::Error, Debug)]
pub enum CompileError {
    #[error("Found Diff Node during XLA conversion {0}")]
    DiffNode(Callsite),

    #[error("Found an unused Parameter in the compute graph {0}")]
    UnusedParameter(Callsite),

    #[error("Unable to compile a context that does not return")]
    NoReturn,

    #[error("XLA error: {0}")]
    Xla(#[from] xla::Error),
}

impl Context {

    pub fn compile<A: Into<NodeIdentifier> + Copy, const N: usize>(
        &mut self,
        name: &str,
        returns: [A; N],
        client: &xla::PjRtClient,
    ) -> Result<xla::PjRtLoadedExecutable> {
        // TODO: gate debug mode behind a feature flag

        if returns.is_empty() {
            Err(CompileError::NoReturn)?;
        }

        for a in returns.iter() {
            self.foldconsts(*a, usize::MAX)?;
        }
        //while self.foldconsts(a, 1)? {
        //    println!("{}", self.to_string(a));
        //}

        for a in returns.iter() {
            self.extract_subterms(*a, usize::MAX)?;
        }
        //while self.extract_subterms(a, 1)? {
        //    println!("{}", self.to_string(a));
        //}

        // Prepare to loop through the unda compute graph and construct the XLA compute graph
        let mut xla_op_slotmap: SlotMap<NodeIdentifier, xla::XlaOp> = SlotMap::with_key();
        let mut unda_op_queue: VecDeque<NodeIdentifier> = VecDeque::new();
        let mut unda_xla_map: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        let mut covered_ops: HashSet<NodeIdentifier> = HashSet::new();

        let builder = xla::XlaBuilder::new(name);

        // declare parameters with the XLA builder
        for (i, unda_id) in self.parameters.iter().enumerate() {
            let node = &self.nodes[*unda_id];

            let shape = node
                .shape
                .sizes
                .iter()
                .map(|d| *d as i64)
                .collect::<SmallVec<[i64; 4]>>();

            let param_name: &String = match &node.operation {
                Operation::Parameter(param_binding) => &param_binding.name,
                _ => unreachable!("Parameter indices pointed to a non-parameter node!"),
            };

            let xla_param =
                builder.parameter(i as i64, node.dtype, shape.as_slice(), &param_name)?;

            let xla_id = xla_op_slotmap.insert(xla_param);
            unda_xla_map.insert(*unda_id, xla_id);
            unda_op_queue.push_back(*unda_id);
        }

        // Initialize constants for the XLA builder
        for unda_id in self.constants.iter() {
            let node = &self.nodes[*unda_id];

            let const_val = match &node.operation {
                Operation::Constant(const_binding) => &const_binding.value,
                _ => unreachable!("Constant indices pointed to a non-constant node!"),
            };

            let xla_const = builder.constant_literal(const_val)?;
            let xla_id = xla_op_slotmap.insert(xla_const);
            unda_xla_map.insert(*unda_id, xla_id);
            unda_op_queue.push_back(*unda_id);
        }

        // using the set of bottom up dependencies we constructed
        // loop through a queue of all operations in the context
        // and add them to the XLA context
        while unda_op_queue.len() > 0 {
            let unda_id = unda_op_queue.pop_front().unwrap();

            let Some(dependent_ops) = self.dependent_nodes.get(&unda_id) else {
                continue;
            };

            for dependent_op in dependent_ops {
                if covered_ops.contains(dependent_op) {
                    continue;
                }
                let node = &self.nodes[*dependent_op];
                match node.operation {
                    Operation::Parameter(_) => {
                        unreachable!("Parameters can't depend on other nodes")
                    }
                    Operation::Constant(_) => unreachable!("Constants can't depend on other nodes"),
                    Operation::StopGradient(node) => {
                        let xla_id = unda_xla_map[&node];
                        unda_xla_map.insert(*dependent_op, xla_id);
                        unda_op_queue.push_back(*dependent_op);
                        covered_ops.insert(*dependent_op);
                    }

                    Operation::Mul(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .mul_(&xla_op_slotmap[unda_xla_map[&node2]])?;
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
                                .add_(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Equal(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .eq(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::NotEqual(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .ne(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::LessThan(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .lt(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::GreaterThan(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .gt(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::LessThanEq(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .le(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::GreaterThanEq(node1, node2) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node1])
                            && xla_op_slotmap.contains_key(unda_xla_map[&node1])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node1]]
                                .ge(&xla_op_slotmap[unda_xla_map[&node2]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::Select {
                        pred,
                        on_true,
                        on_false,
                    } => {
                        if unda_xla_map.contains_key(&pred)
                            && unda_xla_map.contains_key(&on_true)
                            && unda_xla_map.contains_key(&on_false)
                            && xla_op_slotmap.contains_key(unda_xla_map[&pred])
                            && xla_op_slotmap.contains_key(unda_xla_map[&on_true])
                            && xla_op_slotmap.contains_key(unda_xla_map[&on_false])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&pred]].select(
                                &xla_op_slotmap[unda_xla_map[&on_true]],
                                &xla_op_slotmap[unda_xla_map[&on_false]],
                            )?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::TypeCast(node, ty) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].convert(ty.primitive_type())?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::SliceInDim{ node, start, stop, stride, dim } => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].slice_in_dim(start, stop, stride, dim)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ZerosLike(node) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].zeros_like()?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ReduceMax { node, dim, keepdims } => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].reduce_max(&[dim], keepdims)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                }
            }
        }

        let xla_return_vec: Vec<&xla::XlaOp> = returns
            .into_iter()
            .map(|i| &xla_op_slotmap[unda_xla_map[&i.into()]])
            .collect();
        let xla_return_tuple = builder.tuple(&xla_return_vec.as_slice())?;

        let xla_computation = xla_return_tuple.build()?;

        Ok(xla_computation.compile(client)?)
    }
}
