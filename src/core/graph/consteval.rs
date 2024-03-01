use std::collections::HashSet;

use xla::Literal;

use super::*;

impl Context {

    fn collect_deps<A: Into<NodeIdentifier> + Copy>(
        &self,
        curr_node: A,
        prev_dep: A
        ) -> Vec<NodeIdentifier> {
        self.dependent_nodes[&curr_node.into()].iter()
            .filter(|node| node != &&prev_dep.into())
            .map(|node| node.clone())
            .collect::<Vec<NodeIdentifier>>()
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
                                    self.nodes[node_id] = self.nodes[b].clone();

                                    let mut deps = self.collect_deps(node_id, b);
                                    let index = self.dependent_nodes[&node_id].iter().enumerate().find(|n| n.1 == &b).unwrap().0;
                                    self.dependent_nodes.get_mut(&b).unwrap().remove(index);
                                    self.dependent_nodes.get_mut(&b).unwrap().append(&mut deps);

                                    modifications += 1;
                                    changed = true;

                                } else if self.nodes[b].is_zero()? {
                                    self.nodes[node_id] = self.nodes[a].clone();

                                    let mut deps = self.collect_deps(node_id, a);
                                    let index = self.dependent_nodes[&node_id].iter().enumerate().find(|n| n.1 == &b).unwrap().0;
                                    self.dependent_nodes.get_mut(&b).unwrap().remove(index);
                                    self.dependent_nodes.get_mut(&a).unwrap().append(&mut deps);
                                    
                                    modifications += 1;
                                    changed = true;
                                }
                            },
                            Operation::Mul(_, _) => {

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
