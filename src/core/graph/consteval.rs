use std::collections::HashSet;

use xla::ElementType;

use super::*;

impl Context {
    fn collect_deps(&self, node: NodeIdentifier) -> Vec<NodeIdentifier> {
        self.dependent_nodes[&node]
            .iter()
            .map(|node| node.clone())
            .collect::<Vec<NodeIdentifier>>()
    }

    fn replace_index(
        &mut self,
        to_remove: NodeIdentifier,
        rep_with: NodeIdentifier,
    ) -> Result<bool> {
        let mut changed = false;

        let deps = self.collect_deps(to_remove);

        for dep_node in deps {
            match self.nodes[dep_node].operation {
                Operation::Add(a, b) => {
                    if a == b {
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
                Operation::Sub(a, b) => {
                    if a == b {
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
                    if a == b {
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
                Operation::GreaterThan(a, b) => {
                    if a == b {
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
                    if a == b {
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
                    if a == b {
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
                    if a == b {
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
                    if a == b {
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
                    if a == b {
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
                Operation::TypeCast(_, t) => {
                    changed = true;
                    self.nodes[dep_node].operation = Operation::TypeCast(rep_with, t)
                }
                Operation::Select {
                    pred,
                    on_true,
                    on_false,
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
                Operation::ReduceMax {
                    node,
                    dim,
                    keepdims,
                } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::ReduceMax {
                            node: rep_with,
                            dim,
                            keepdims,
                        }
                    }
                }
                Operation::ReduceArgmax {
                    node,
                    dim,
                    keepdims,
                } => {
                    if node == to_remove {
                        changed = true;
                        self.nodes[dep_node].operation = Operation::ReduceArgmax {
                            node: rep_with,
                            dim,
                            keepdims,
                        }
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
            }
        }
        Ok(changed)
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
            if visitied.contains(&node_id) || modifications >= modification_limit {
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
                    if let Some(literal) = self.nodes[a].is_const() {
                        //Check for mul by 1
                        let floating_literal: Vec<f32> =
                            literal.convert(xla::PrimitiveType::F32)?.to_vec()?;
                        let mut all_one = true;
                        floating_literal.iter().for_each(|elem| {
                            if *elem != 1f32 {
                                all_one = false;
                            }
                        });
                        if all_one {
                            //a is all ones, replace node_id with a
                            self.replace_index(node_id, b)?;
                            modifications += 1;
                            changed = true;
                        }
                    }
                    if self.nodes[b].is_zero()? {
                        self.replace_index(node_id, b)?;
                        modifications += 1;
                        changed = true;
                    }
                    if let Some(literal) = self.nodes[b].is_const() {
                        //Check for mul by 1
                        let floating_literal: Vec<f32> =
                            literal.convert(xla::PrimitiveType::F32)?.to_vec()?;
                        let mut all_one = true;
                        floating_literal.iter().for_each(|elem| {
                            if *elem != 1f32 {
                                all_one = false;
                            }
                        });
                        if all_one {
                            //b is all ones, replace node_id with a
                            self.replace_index(node_id, a)?;
                            modifications += 1;
                            changed = true;
                        }
                    }
                    if let None = self.nodes[a].is_const() {
                        to_visit.push(a);
                    }
                    if let None = self.nodes[b].is_const() {
                        to_visit.push(b);
                    }
                }
                Operation::Neg(a) => {
                    if let None = self.nodes[a].is_const() {
                        to_visit.push(a);
                    }
                }
                Operation::GreaterThan(a, b)
                | Operation::GreaterThanEq(a, b)
                | Operation::LessThan(a, b)
                | Operation::LessThanEq(a, b)
                | Operation::Equal(a, b)
                | Operation::NotEqual(a, b) => {
                    if let None = self.nodes[a].is_const() {
                        to_visit.push(a);
                    }

                    if let None = self.nodes[b].is_const() {
                        to_visit.push(b);
                    }
                }
                Operation::StopGradient(a)
                | Operation::TypeCast(a, _)
                | Operation::ZerosLike(a)
                | Operation::OneHot(a) => {
                    if let None = self.nodes[a].is_const() {
                        to_visit.push(a);
                    }
                }
                Operation::Select {
                    pred,
                    on_true,
                    on_false,
                } => {
                    if let None = self.nodes[pred].is_const() {
                        to_visit.push(pred)
                    }
                    if let None = self.nodes[on_true].is_const() {
                        to_visit.push(on_true)
                    }
                    if let None = self.nodes[on_false].is_const() {
                        to_visit.push(on_false)
                    }
                }
                Operation::SliceInDim {
                    node,
                    start,
                    stop,
                    stride,
                    dim,
                } => {
                    if let None = self.nodes[node].is_const() {
                        to_visit.push(node);
                    }
                }
                Operation::ReduceMax {
                    node,
                    dim,
                    keepdims,
                } => {
                    if let None = self.nodes[node].is_const() {
                        to_visit.push(node);
                    }
                }
                Operation::Constant(_) | Operation::Parameter(_) => {}
                Operation::ReduceArgmax {
                    node,
                    dim,
                    keepdims,
                } => {
                    if let None = self.nodes[node].is_const() {
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
