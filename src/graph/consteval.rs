use std::collections::HashSet;

use super::*;

impl Context {
    fn collect_deps(&self, node: NodeIdentifier) -> Vec<NodeIdentifier> {
        if self.dependent_nodes.contains_key(&node) {
            self.dependent_nodes[&node].to_vec()
        } else {
            vec![]
        }
    }

    pub(crate) fn replace_index(
        &mut self,
        to_remove: NodeIdentifier,
        rep_with: NodeIdentifier,
    ) -> Result<bool> {
        let mut changed = false;

        let deps = self.collect_deps(to_remove);

        for dep_node in deps {
            //Again, clone here is pretty bad
            match self.nodes[dep_node].operation.clone() {
                Operation::Add(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::Add(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Add(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::Add(a, rep_with);
                        changed = true;
                    }
                }
                Operation::RngUniform(a, b, shape)
                    | Operation::RngNormal(a, b, shape) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::RngUniform(rep_with, b, shape);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::RngUniform(a, rep_with, shape);
                        changed = true;
                    } 
                }
                Operation::Pow(a, b) => {
                    if a == to_remove && a == b {
                        self.nodes[dep_node].operation = Operation::Pow(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Pow(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::Pow(a, rep_with);
                        changed = true;
                    }
                }
                Operation::Sub(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::Sub(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Sub(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::Sub(a, rep_with);
                        changed = true;
                    }
                }
                Operation::Mul(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::Mul(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Mul(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::Mul(a, rep_with);
                        changed = true;
                    }
                }
                Operation::MatMul(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::MatMul(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::MatMul(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::MatMul(a, rep_with);
                        changed = true;
                    }
                }
                Operation::Div(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::Div(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Div(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::Div(a, rep_with);
                        changed = true;
                    }
                }
                Operation::GreaterThan(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::GreaterThan(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::GreaterThan(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::GreaterThan(a, rep_with);
                        changed = true;
                    }
                }

                Operation::GreaterThanEq(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation =
                            Operation::GreaterThanEq(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::GreaterThanEq(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::GreaterThanEq(a, rep_with);
                        changed = true;
                    }
                }
                Operation::Equal(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::Equal(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Equal(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::Equal(a, rep_with);
                        changed = true;
                    }
                }
                Operation::NotEqual(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::NotEqual(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::NotEqual(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::NotEqual(a, rep_with);
                        changed = true;
                    }
                }
                Operation::LessThan(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::LessThan(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::LessThan(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::LessThan(a, rep_with);
                        changed = true;
                    }
                }

                Operation::LessThanEq(a, b) => {
                    if to_remove == a && a == b {
                        self.nodes[dep_node].operation = Operation::LessThanEq(rep_with, rep_with);
                        changed = true;
                    } else if a == to_remove {
                        self.nodes[dep_node].operation = Operation::LessThanEq(rep_with, b);
                        changed = true;
                    } else if b == to_remove {
                        self.nodes[dep_node].operation = Operation::LessThanEq(a, rep_with);
                        changed = true;
                    }
                }
                Operation::Constant(_) | Operation::Parameter(_) => {
                    unreachable!("Constants or Parameters cannot depend on nodes");
                }
                Operation::StopGradient(a) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::StopGradient(rep_with);
                        changed = true;
                    }
                }
                Operation::Neg(a) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Neg(rep_with);
                        changed = true;
                    }
                }
                Operation::Exp(a) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Exp(rep_with);
                        changed = true;
                    }
                }

                Operation::Log(a) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Log(rep_with);
                        changed = true;
                    }
                }

                Operation::ZerosLike(a) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::ZerosLike(rep_with);
                        changed = true;
                    }
                }
                Operation::OneHot(node) => {
                    if node == to_remove {
                        self.nodes[dep_node].operation = Operation::OneHot(rep_with);
                        changed = true;
                    }
                }
                Operation::TypeCast(node, t) => {
                    if node == to_remove {
                        self.nodes[dep_node].operation = Operation::TypeCast(rep_with, t);
                        changed = true;
                    }
                }
                Operation::Reshape(node) => {
                    if node == to_remove {
                        self.nodes[dep_node].operation = Operation::Reshape(rep_with);
                        changed = true;
                    }
                }
                Operation::Select {
                    pred,
                    on_false,
                    on_true,
                } => {
                    if pred == to_remove {
                        if pred == on_true {
                            self.nodes[dep_node].operation = Operation::Select {
                                pred: rep_with,
                                on_true: rep_with,
                                on_false,
                            }
                        } else if pred == on_false {
                            self.nodes[dep_node].operation = Operation::Select {
                                pred: rep_with,
                                on_true,
                                on_false: rep_with,
                            }
                        } else {
                            self.nodes[dep_node].operation = Operation::Select {
                                pred: rep_with,
                                on_true,
                                on_false,
                            }
                        }
                        changed = true;
                    } else if on_true == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::Select {
                            pred,
                            on_true: rep_with,
                            on_false,
                        }
                    } else if on_false == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::Select {
                            pred,
                            on_true,
                            on_false: rep_with,
                        }
                    }
                }
                Operation::ReduceMax { node, dim } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::ReduceMax {
                            node: rep_with,
                            dim,
                        }
                    }
                }
                Operation::ReduceArgmax {
                    node,
                    dim,
                } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::ReduceArgmax {
                            node: rep_with,
                            dim,
                        }
                    }
                }
                Operation::ReduceSum { node, dim } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::ReduceSum {
                            node: rep_with,
                            dim,
                        }
                    }
                }
                Operation::ReduceMean { node, dim } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::ReduceMean {
                            node: rep_with,
                            dim,
                        }
                    }
                }
                Operation::Transpose(a, dim) => {
                    if a == to_remove {
                        self.nodes[dep_node].operation = Operation::Transpose(rep_with, dim.clone());
                        changed = true;
                    }
                }
                Operation::SliceInDim {
                    node,
                    start,
                    stop,
                    stride,
                    dim,
                } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::SliceInDim {
                            node: rep_with,
                            start,
                            stop,
                            stride,
                            dim,
                        }
                    }
                }
                Operation::TileInDim { node, n_tiles, dim } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::TileInDim {
                            node: rep_with,
                            n_tiles,
                            dim,
                        }
                    }
                }
            }
        }

        //iterate through all dependent node vecs that contain the to_remove node, and replace them
        //with rep_with node
        
        for (id, deps) in self.dependent_nodes.clone().iter().filter(|(_, nodes)| nodes.contains(&to_remove)) {
            let deps_without = deps.clone().into_iter().filter(|node| node != &to_remove).collect::<Vec<NodeIdentifier>>();

            /*if !deps_without.contains(&rep_with) {
                deps_without.push(rep_with);
            }*/

            self.dependent_nodes.insert(*id, deps_without);
        }

        if self.dependent_nodes.contains_key(&to_remove) {
            let mut new_deps = self.dependent_nodes[&to_remove].clone();
            new_deps.extend(self.dependent_nodes.get(&rep_with).unwrap_or(&vec![]).iter());
            self.dependent_nodes.insert(rep_with, new_deps);
        }


        /*if let Some(deps) = self.dependent_nodes.remove(&to_remove) {
            let mut prev_deps = self.dependent_nodes.get(&rep_with).unwrap_or(&Vec::new()).clone();
            prev_deps.extend(deps.iter());

            self.dependent_nodes.insert(rep_with, prev_deps);
        }*/


        if let Some((idx, _)) = self.parameters.iter().enumerate().find(|(_, node)| *node == &to_remove) {
            self.parameters.remove(idx);
        }

        if let Some((idx, _)) = self.constants.iter().enumerate().find(|(_, node)| *node == &to_remove) {
            self.constants.remove(idx);
        }

        self.dependent_nodes.remove(&to_remove);
        self.nodes.remove(to_remove);

        Ok(changed)
    }

    // Utility function for handling products with tiled and reshaped constants
    // TODO: This should be made to handle arbitrary depth
    fn replace_tiled_const(&mut self, a: NodeIdentifier, b: NodeIdentifier, top_level_node: NodeIdentifier) -> Result<bool> {
        if let Operation::TileInDim { node, n_tiles: _, dim: _ } = self.nodes[a].operation {
            if self.nodes[node].is_one()? {
                let tiled_b = self.tile_to_shape(b, self.nodes[top_level_node].shape.clone())?;
                self.replace_index(top_level_node, tiled_b)?;
                Ok(true)
            } else {
                self.replace_tiled_const(node, b, top_level_node)
            }
        } else if let Operation::Reshape(node) = self.nodes[a].operation {
            if self.nodes[node].is_one()? {
                let tiled_b = self.tile_to_shape(b, self.nodes[top_level_node].shape.clone())?;
                self.replace_index(top_level_node, tiled_b)?;
                Ok(true)
            } else {
                self.replace_tiled_const(node, b, top_level_node)
            }
        }
        else {
            Ok(false)
        }
    }

    /// Folds constants in place by replacing any node whose both inputs are Constant
    /// with a Constant of the result of the operation. All existing references to
    /// the old node will still point to it once its replaced, and this process is
    /// repeated until there are no more nodes whose inputs are all constants.
    pub(crate) fn fold_consts(
        &mut self,
        input: NodeIdentifier,
        modification_limit: usize,
    ) -> Result<bool> {
        if modification_limit == 0 {
            return Ok(true);
        }

        let mut modifications: usize = 0;
        let mut changed = false;

        let mut to_visit: Vec<NodeIdentifier> = vec![input];
        let mut visitied: HashSet<NodeIdentifier> = HashSet::new();

        while let Some(node_id) = to_visit.pop() {
            if visitied.contains(&node_id) || modifications >= modification_limit || !self.nodes.contains_key(node_id) {
                continue;
            }
            match self.nodes[node_id].operation {
                Operation::Add(a, b) | Operation::Sub(a, b) => {
                    if self.nodes[a].is_zero()? {
                        self.replace_index(node_id, b)?;
                        modifications += 1;
                        changed = true;
                    } else if self.nodes[b].is_zero()? {
                        self.replace_index(node_id, a)?;
                        modifications += 1;
                        changed = true;
                    }
                    //Enqueue the dependent nodes to check both of them for constant
                    //mul/adding

                    //TODO: Once we create a new Node based on the constant propegation,
                    //use insert_with_key to 'replace existant node'
                    if self.nodes.get(a).unwrap().is_const().is_none() {
                        to_visit.push(a);
                    }
                    if self.nodes.get(b).unwrap().is_const().is_none() {
                        to_visit.push(b);
                    }
                }
                Operation::Mul(a, b) => {
                    if self.nodes[a].is_zero()? {
                        self.replace_index(node_id, a)?;
                        modifications += 1;
                        changed = true;
                    }
                    if self.nodes[a].is_one()? {
                        self.replace_index(node_id, b)?;
                        modifications += 1;
                        changed = true;
                    }
                    if self.nodes[b].is_zero()? {
                        self.replace_index(node_id, b)?;
                        modifications += 1;
                        changed = true;
                    }
                    if self.nodes[b].is_one()? {
                        self.replace_index(node_id, a)?;
                        modifications += 1;
                        changed = true;
                    }
                    if self.replace_tiled_const(a, b, node_id)? {
                        modifications += 1;
                        changed = true;
                    };
                    if self.replace_tiled_const(b, a, node_id)? {
                        modifications += 1;
                        changed = true;
                    };
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }
                    if self.nodes[b].is_const().is_none() {
                        to_visit.push(b);
                    }
                }
                Operation::Neg(a) => {
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }
                }
                Operation::Exp(a) => {
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }
                }
                Operation::Log(a) => {
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }
                }
                Operation::Transpose(a, _) => {
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }
                }
                Operation::GreaterThan(a, b)
                | Operation::GreaterThanEq(a, b)
                | Operation::LessThan(a, b)
                | Operation::LessThanEq(a, b)
                | Operation::Equal(a, b)
                | Operation::NotEqual(a, b)
                | Operation::Div(a, b)
                | Operation::Pow(a, b)
                | Operation::RngUniform(a, b, _)
                | Operation::RngNormal(a, b, _)
                | Operation::MatMul(a, b) => {
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }

                    if self.nodes[b].is_const().is_none() {
                        to_visit.push(b);
                    }
                }
                Operation::StopGradient(a)
                | Operation::TypeCast(a, _)
                | Operation::Reshape(a)
                | Operation::ZerosLike(a)
                | Operation::OneHot(a) => {
                    if self.nodes[a].is_const().is_none() {
                        to_visit.push(a);
                    }
                }
                Operation::Select {
                    pred,
                    on_true,
                    on_false,
                } => {
                    if self.nodes[pred].is_const().is_none() {
                        to_visit.push(pred)
                    }
                    if self.nodes[on_true].is_const().is_none() {
                        to_visit.push(on_true)
                    }
                    if self.nodes[on_false].is_const().is_none() {
                        to_visit.push(on_false)
                    }
                }
                Operation::SliceInDim {
                    node,
                    start: _,
                    stop: _,
                    stride: _,
                    dim: _,
                } => {
                    if self.nodes[node].is_const().is_none() {
                        to_visit.push(node);
                    }
                }
                Operation::TileInDim { node, n_tiles: _, dim: _ } => {
                    if self.nodes[node].is_const().is_none() {
                        to_visit.push(node);
                    }
                }
                Operation::ReduceMax { node, dim: _ }
                | Operation::ReduceSum { node, dim: _ }
                | Operation::ReduceMean { node, dim:_  }
                | Operation:: ReduceArgmax { node, dim: _ } => {
                    if self.nodes[node].is_const().is_none() {
                        to_visit.push(node);
                    }
                }
                Operation::Constant(_) | Operation::Parameter(_) => {}
            }
            visitied.insert(node_id);
        }

        Ok(changed)
    }
}
