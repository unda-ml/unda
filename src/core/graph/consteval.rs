use std::collections::HashSet;

use xla::Literal;

use super::*;

impl Context {
    /// Folds constants in place by replacing any node whose both inputs are Constant
    /// with a Constant of the result of the operation. All existing references to
    /// the old node will still point to it once its replaced, and this process is
    /// repeated until there are no more nodes whose inputs are all constants.
    pub(crate) fn foldconsts<A: Into<NodeIdentifier> + Copy>(
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
            
            if let Some(node) = self.nodes.get(node_id) {
                match node.operation {
                    Operation::Add(a, b) | Operation::Sub(a, b) | Operation::Mul(a, b) => {
                        if let (Some(a_node), Some(b_node)) = (self.nodes.get(a.into()), self.nodes.get(b.into())) {
                            
                            //TODO, if add and one of the nodes is zero convert current node to be
                            //just the node that isn't zero
                            //
                            //If mul and one of the nodes is zero, zero out the new node
                            //If mul and one of the nodes is one, set the node to just be
                            //the non-one node
                            //
                            //Increment modifications only if these cases are met
                            modifications += 1;
                            //Enqueue the dependent nodes to check both of them for constant
                            //mul/adding
                            
                            //TODO: Once we create a new Node based on the constant propegation,
                            //use insert_with_key to 'replace existant node'
                            if a_node.is_const().is_none() {
                                to_visit.push(a.into());
                            } 
                            if b_node.is_const().is_none() {
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
            }
            
            visitied.insert(node_id.into());
        }

        Ok(changed)
    }
}
