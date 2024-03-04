use std::collections::HashSet;

use super::*;

impl Context {

    fn collect_deps<A: Into<NodeIdentifier> + Copy>(
        &self,
        node: A,
        ) -> Vec<NodeIdentifier> {
        self.dependent_nodes[&node.into()].iter()
            .map(|node| node.clone())
            .collect::<Vec<NodeIdentifier>>()
    }

    fn replace_index<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        to_remove: A,
        rep_with: A
        ) -> Result<bool> {
        let mut changed = false;
        
        let deps = self.collect_deps(to_remove);

        for dep_node in deps {
            match self.nodes[dep_node].operation {
                Operation::Add(a, b) 
                    | Operation::Sub(a, b)
                    | Operation::Mul(a, b)
                    | Operation::GreaterThan(a, b)
                    | Operation::GreaterThanEq(a, b)
                    | Operation::Equal(a, b)
                    | Operation::NotEqual(a, b)
                    | Operation::LessThan(a, b)
                    | Operation::LessThanEq(a, b)  => {
                        
                    if a == to_remove.into() {
                        match self.nodes[dep_node].operation {
                            Operation::Add(_, _) => self.nodes[dep_node].operation = Operation::Add(rep_with.into(), b),
                            Operation::Sub(_, _) => self.nodes[dep_node].operation = Operation::Sub(rep_with.into(), b),
                            Operation::Mul(_, _) => self.nodes[dep_node].operation = Operation::Mul(rep_with.into(), b),
                            Operation::GreaterThan(_, _) => self.nodes[dep_node].operation = Operation::GreaterThan(rep_with.into(), b),
                            Operation::GreaterThanEq(_, _) => self.nodes[dep_node].operation = Operation::GreaterThanEq(rep_with.into(), b),
                            Operation::Equal(_, _) => self.nodes[dep_node].operation = Operation::Equal(rep_with.into(), b),
                            Operation::NotEqual(_, _) => self.nodes[dep_node].operation = Operation::NotEqual(rep_with.into(), b),
                            Operation::LessThan(_, _) => self.nodes[dep_node].operation = Operation::LessThan(rep_with.into(), b),
                            Operation::LessThanEq(_, _) => self.nodes[dep_node].operation = Operation::LessThanEq(rep_with.into(), b),
                            _ => unreachable!("Only add, sub, mul, gt, gte, e, ne, lt and lte can be reached here")
                        }
                        changed = true;
                    } else if b == to_remove.into() {
                        match self.nodes[dep_node].operation {
                            Operation::Add(_, _) => self.nodes[dep_node].operation = Operation::Add(a, rep_with.into()),
                            Operation::Sub(_, _) => self.nodes[dep_node].operation = Operation::Sub(a, rep_with.into()),
                            Operation::Mul(_, _) => self.nodes[dep_node].operation = Operation::Mul(a, rep_with.into()),
                            Operation::GreaterThan(_, _) => self.nodes[dep_node].operation = Operation::GreaterThan(a, rep_with.into()),
                            Operation::GreaterThanEq(_, _) => self.nodes[dep_node].operation = Operation::GreaterThanEq(a, rep_with.into()),
                            Operation::Equal(_, _) => self.nodes[dep_node].operation = Operation::Equal(a, rep_with.into()),
                            Operation::NotEqual(_, _) => self.nodes[dep_node].operation = Operation::NotEqual(a, rep_with.into()),
                            Operation::LessThan(_, _) => self.nodes[dep_node].operation = Operation::LessThan(a, rep_with.into()),
                            Operation::LessThanEq(_, _) => self.nodes[dep_node].operation = Operation::LessThanEq(a, rep_with.into()),
                            _ => unreachable!("Only add, sub, mul, gt, gte, e, ne, lt and lte can be reached here")
                        }

                        changed = true;
                    }
                }, 
                Operation::Constant(_) 
                    | Operation::Parameter(_) => {
                    unreachable!("Constants or Parameters cannot depend on nodes");
                },
                Operation::StopGradient(a) 
                    | Operation::Neg(a)
                    | Operation::ZerosLike(a) => {
                    if a == to_remove.into() {
                        match self.nodes[dep_node].operation {
                            Operation::Neg(_) => self.nodes[dep_node].operation = Operation::Neg(rep_with.into()),
                            Operation::ZerosLike(_) => self.nodes[dep_node].operation = Operation::ZerosLike(rep_with.into()),
                            Operation::StopGradient(_) => self.nodes[dep_node].operation = Operation::StopGradient(rep_with.into()),
                            _ => unreachable!("Only Neg, StopGradient and ZerosLike get this far")
                        }
                        changed = true;
                    }
                },
                Operation::TypeCast(a, t) => {
                    changed = true;
                    self.nodes[dep_node].operation = Operation::TypeCast(rep_with.into(), t)
                },
                Operation::Select { pred, on_true, on_false } => {
                    if pred == to_remove.into() {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::Select { pred: rep_with.into(), on_true, on_false }
                    } else if on_true == to_remove.into() {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::Select { pred, on_true: rep_with.into(), on_false }
                    } else if on_false == to_remove.into() {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::Select { pred, on_true, on_false: rep_with.into() }
                    }
                },
                Operation::ReduceMax { node, dim, keepdims } => {
                    changed = true;
                    self.nodes[dep_node].operation = Operation::ReduceMax { node: rep_with.into(), dim, keepdims }
                },
                Operation::SliceInDim { node, start, stop, stride, dim } => {
                    changed = true;
                    self.nodes[dep_node].operation = Operation::SliceInDim { node: rep_with.into(), start, stop, stride, dim }
                }
            }
        }
        Ok(changed)
    }

    /// Folds constants in place by replacing any node whose both inputs are Constant
    /// with a Constant of the result of the operation. All existing references to
    /// the old node will still point to it once its replaced, and this process is
    /// repeated until there are no more nodes whose inputs are all constants.
    pub(crate) fn fold_consts<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        input: A,
        modification_limit: usize,
        ) -> Result<bool> {
        if modification_limit == 0 {
            return Ok(true);
        }

        let mut modifications: usize = 0;
        let mut changed = false;

        let mut to_visit: Vec<NodeIdentifier> = vec![input.into()];
        let mut visitied: HashSet<NodeIdentifier> = HashSet::new();

        while let Some(node_id) = to_visit.pop() {
            if visitied.contains(&node_id.into()) || modifications >= modification_limit {
                continue;
            }
            match self.nodes[node_id].operation {
                Operation::Add(a, b) | Operation::Sub(a, b) | Operation::Mul(a, b) => {
                    if self.nodes.contains_key(a.into()) && self.nodes.contains_key(b.into()) {

                        match self.nodes[node_id].operation {
                            Operation::Add(_, _) | Operation::Sub(_, _) => {
                                if self.nodes[a].is_zero()? {
                                    self.replace_index(node_id, b)?;
                                    modifications += 1;
                                    changed = true;

                                } else if self.nodes[b].is_zero()? {
                                    self.replace_index(node_id, a)?;
                                    modifications += 1;
                                    changed = true;
                                }
                            },
                            Operation::Mul(_, _) => {
                                if self.nodes[a].is_zero()? {
                                    self.replace_index(node_id, a)?;
                                    modifications += 1;
                                    changed = true;
                                } else if let Some(literal) = self.nodes[a].is_const() {
                                    //Check for mul by 1

                                } else if self.nodes[b].is_zero()?{
                                    self.replace_index(node_id, b)?;
                                    modifications += 1;
                                    changed = true;
                                } else if let Some(literal) = self.nodes[b].is_const() {
                                    //Check for mul by 1
                                    
                                }
                            },
                            _ => {
                                unreachable!("Cannot fold parameters of a node that isn't mul, add or sub for now")
                            }

                        };

                        modifications += 1;
                        //Enqueue the dependent nodes to check both of them for constant
                        //mul/adding

                        //TODO: Once we create a new Node based on the constant propegation,
                        //use insert_with_key to 'replace existant node'
                        if self.nodes.get(a.into()).unwrap().is_const().is_none() {
                            to_visit.push(a.into());
                        } 
                        if self.nodes.get(b.into()).unwrap().is_const().is_none() {
                            to_visit.push(b.into());
                        }
                    }
                },
                Operation::GreaterThan(a, b)
                    | Operation::GreaterThanEq(a, b)
                    | Operation::LessThan(a, b)
                    | Operation::LessThanEq(a, b)
                    | Operation::Equal(a, b)
                    => {

                        if let Some(node) = self.nodes.get(a) {
                            if node.is_const().is_none() {
                                to_visit.push(a);
                            }
                        }

                        if let Some(node) = self.nodes.get(b) {
                            if node.is_const().is_none() {
                                to_visit.push(b);
                            }
                        }
                    },
                _ => {}
            }
            visitied.insert(node_id.into());
        }

        Ok(changed)
    }
}
