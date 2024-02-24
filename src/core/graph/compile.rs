use super::*;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(thiserror::Error, Debug)]
pub enum CompileError {
    #[error("Found Diff Node during XLA conversion {0}")]
    DiffNode(Callsite),

    #[error("Found an unused Parameter in the compute graph {0}")]
    UnusedParameter(Callsite),

    #[error("XLA error: {0}")]
    Xla(#[from] xla::Error),
}

impl Context {
    fn get_dependent_nodes<A: Into<NodeIdentifier> + Copy>(
        &self,
        a: A,
        dep_nodes: &mut HashMap<NodeIdentifier, Vec<NodeIdentifier>>,
        constants: &mut HashSet<NodeIdentifier>,
        parameters: &mut HashSet<NodeIdentifier>,
    ) -> Result<()> {
        let this_node_id = a.into();
        let input_node = &self.nodes[this_node_id];
        match input_node.operation {
            Operation::Constant(_) => {
                constants.insert(this_node_id);
                Ok(())
            }
            Operation::Parameter(_) => {
                parameters.insert(this_node_id);
                Ok(())
            }
            Operation::Diff(_, _) => Err(CompileError::DiffNode(input_node.callsite.clone()))?,
            Operation::Mul(node1, node2) | Operation::Add(node1, node2) => {
                dep_nodes
                    .entry(node1)
                    .or_insert(Vec::new())
                    .push(this_node_id);
                dep_nodes
                    .entry(node2)
                    .or_insert(Vec::new())
                    .push(this_node_id);
                self.get_dependent_nodes(node1, dep_nodes, constants, parameters)?;
                self.get_dependent_nodes(node2, dep_nodes, constants, parameters)
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
        while self.autodiff(a, 1)? {
            println!("{}", self.to_string(a));
        }

        //self.foldconsts(a, usize::MAX);
        while self.foldconsts(a, 1)? {
            println!("{}", self.to_string(a));
        }

        //self.extract_subterms(a, usize::MAX);
        while self.extract_subterms(a, 1)? {
            println!("{}", self.to_string(a));
        }

        let builder = xla::XlaBuilder::new(name);
        // Get the bottom-up dependencies of the compute graph
        let mut dependent_nodes = HashMap::new();
        let mut constants = HashSet::new();
        let mut parameters = HashSet::new();
        self.get_dependent_nodes(a, &mut dependent_nodes, &mut constants, &mut parameters)?;

        // Prepare to loop through the unda compute graph and construct the XLA compute graph
        let mut xla_op_slotmap: SlotMap<NodeIdentifier, xla::XlaOp> = SlotMap::with_key();
        let mut unda_op_queue: VecDeque<NodeIdentifier> = VecDeque::new();
        let mut unda_xla_map: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        let mut covered_ops: HashSet<NodeIdentifier> = HashSet::new();

        // declare parameters with the XLA builder
        for (i, unda_id) in self.param_indices.iter().enumerate() {
            let node = &self.nodes[*unda_id];

            if !parameters.contains(unda_id) {
                // Unused parameter found
            }

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
        // >1 dimensions not yet supported for constants
        for unda_id in constants.iter() {
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

            let Some(dependent_ops) = dependent_nodes.get(&unda_id) else {
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
                    Operation::Diff(_, _) => Err(CompileError::DiffNode(node.callsite.clone()))?,

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
                }
            }
        }

        let xla_computation = xla_op_slotmap[unda_xla_map[&a.into()]].build()?;

        Ok(xla_computation.compile(client)?)
    }
}
