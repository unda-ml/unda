use super::*;
use slotmap::SlotMap;
use smallvec::SmallVec;
use xla::XlaOp;
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
    pub fn build<const N: usize>(
        &mut self,
        name: &str,
        returns: [NodeIdentifier; N],
    ) -> Result<xla::XlaComputation> {
        // TODO: gate debug mode behind a feature flag

        if returns.is_empty() {
            Err(CompileError::NoReturn)?;
        }

        for a in returns.iter() {
            self.fold_consts(*a, usize::MAX)?;
        }

        //while self.foldconsts(a, 1)? {
        //    println!("{}", self.to_string(a));
        //}
        self.extract_subterms(&returns, usize::MAX)?;
        /*for a in returns.iter() {
            self.extract_subterms(*a, usize::MAX)?;
        }*/
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
                Operation::Parameter(param_binding) => param_binding,
                _ => unreachable!("Parameter indices pointed to a non-parameter node!"),
            };

            let xla_param =
                builder.parameter(i as i64, node.dtype, shape.as_slice(), param_name)?;

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
        while !unda_op_queue.is_empty() {
            let unda_id = unda_op_queue.pop_front().unwrap();

            let Some(dependent_ops) = self.dependent_nodes.get(&unda_id) else {
                continue;
            };

            for dependent_op in dependent_ops {
                if covered_ops.contains(dependent_op) {
                    continue;
                }
                let this_node = &self.nodes[*dependent_op];
                //TODO: Clone here is not great, we could & the node operation
                //or come up with a better way of storing the Vec<i64> that Transpose
                //uses(that's what causes the borrow checker error if we dont clone)
                match this_node.operation.clone() {
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

                    Operation::Mul(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .mul_(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::MatMul(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .matmul(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Div(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .div_(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::RngNormal(mu, sigma, shape) => {
                        if unda_xla_map.contains_key(&mu)
                            && unda_xla_map.contains_key(&sigma)
                            && xla_op_slotmap.contains_key(unda_xla_map[&mu])
                            && xla_op_slotmap.contains_key(unda_xla_map[&sigma])
                        {
                            let dtype = self.nodes[mu].dtype;
                            let xla_op = XlaOp::rng_normal(&xla_op_slotmap[unda_xla_map[&mu]],
                                               &xla_op_slotmap[unda_xla_map[&sigma]], &shape.to_array_shape(dtype))?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::RngUniform(min, max, shape) => {
                        if unda_xla_map.contains_key(&min)
                            && unda_xla_map.contains_key(&max)
                            && xla_op_slotmap.contains_key(unda_xla_map[&min])
                            && xla_op_slotmap.contains_key(unda_xla_map[&max])
                        {
                            let dtype = self.nodes[min].dtype;
                            let xla_op = XlaOp::rng_uniform(&xla_op_slotmap[unda_xla_map[&min]],
                                               &xla_op_slotmap[unda_xla_map[&max]], &shape.to_array_shape(dtype))?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::Pow(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .pow(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Transpose(a, perm_index) => {
                        if unda_xla_map.contains_key(&a)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]].transpose(&perm_index)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Add(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .add_(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Sub(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .sub_(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Neg(node) => {
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node]].neg()?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Exp(a) => {
                        if unda_xla_map.contains_key(&a)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]].exp()?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Log(a) => {
                        if unda_xla_map.contains_key(&a)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]].log()?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::Equal(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .eq(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::NotEqual(a, b) => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .ne(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::LessThan(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .lt(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::GreaterThan(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .gt(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::LessThanEq(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .le(&xla_op_slotmap[unda_xla_map[&b]])?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }

                    Operation::GreaterThanEq(a, b) => {
                        if unda_xla_map.contains_key(&a)
                            && unda_xla_map.contains_key(&b)
                            && xla_op_slotmap.contains_key(unda_xla_map[&a])
                            && xla_op_slotmap.contains_key(unda_xla_map[&b])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&a]]
                                .ge(&xla_op_slotmap[unda_xla_map[&b]])?;
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
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].convert(ty.primitive_type())?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::Reshape(node) => {
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node]].reshape(
                                this_node
                                    .shape
                                    .sizes
                                    .iter()
                                    .map(|s| *s as i64)
                                    .collect::<Vec<i64>>()
                                    .as_slice(),
                            )?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::SliceInDim {
                        node,
                        start,
                        stop,
                        stride,
                        dim,
                    } => {
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node]]
                                .slice_in_dim(start, stop, stride, dim)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::TileInDim { node, n_tiles, dim } => {
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let node_op = &xla_op_slotmap[unda_xla_map[&node]];
                            let mut copies: Vec<xla::XlaOp> = Vec::new();
                            for _ in 1..n_tiles {
                                copies.push(node_op.copy()?);
                            }
                            let xla_op = node_op.concat_in_dim(copies.as_slice(), dim)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ZerosLike(node) => {
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let xla_op = xla_op_slotmap[unda_xla_map[&node]].zeros_like()?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::OneHot(node) => {
                        if unda_xla_map.contains_key(&node)
                            && xla_op_slotmap.contains_key(unda_xla_map[&node])
                        {
                            let n_classes = this_node.shape.sizes[1];
                            let dtype = this_node.dtype;
                            let xla_op = xla_op_slotmap[unda_xla_map[&node]]
                                .one_hot(n_classes as i64, dtype)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ReduceMax { node, dim } => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].reduce_max(&[dim], false)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ReduceSum { node, dim } => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].reduce_sum(&[dim], false)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ReduceMean { node, dim } => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].reduce_mean(&[dim], false)?;
                            let xla_id = xla_op_slotmap.insert(xla_op);
                            unda_xla_map.insert(*dependent_op, xla_id);
                            unda_op_queue.push_back(*dependent_op);
                            covered_ops.insert(*dependent_op);
                        }
                    }
                    Operation::ReduceArgmax { node, dim } => {
                        if xla_op_slotmap.contains_key(unda_xla_map[&node]) {
                            let xla_op =
                                xla_op_slotmap[unda_xla_map[&node]].reduce_argmax(dim, false)?;
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
            .map(|i| &xla_op_slotmap[unda_xla_map[&i]])
            .collect();
        let xla_return_tuple = builder.tuple(xla_return_vec.as_slice())?;

        let xla_computation = xla_return_tuple.build()?;

        Ok(xla_computation)
    }

    pub fn compile<const N: usize>(&mut self, name: &str, returns: [NodeIdentifier; N], client: &xla::PjRtClient) -> Result<xla::PjRtLoadedExecutable> {
        let comp = self.build(name, returns)?;
        match comp.compile(client) {
            Ok(exe) => Ok(exe),
            Err(err) => Err(ContextError::Xla(err)),
        }
    }
}
