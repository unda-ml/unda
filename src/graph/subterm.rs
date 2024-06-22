use super::*;
use std::collections::{HashMap, HashSet};

impl Context {
    /// Traverses graph context, building a hashmap of Node -> NodeIdentifier pairs
    /// If a duplicate Node is found, we can reference the other NodeIdentifier with
    /// the already existant node instead of having duplicates
    /// make sure to update entry for the modified node, as the hash will change.
    /// do not include callsite when calculating the hash.
    pub(crate) fn extract_subterms(
        &mut self,
        outputs: &[NodeIdentifier],
        modification_limit: usize,
    ) -> Result<bool> {
        if modification_limit == 0 {
            return Ok(true);
        }
        let mut node_map: HashMap<Node, NodeIdentifier> = HashMap::new();

        let mut modifications = 0;
        let mut changed = false;

        let mut to_visit: Vec<NodeIdentifier> = outputs.to_vec();
        let mut visited: HashSet<NodeIdentifier> = HashSet::new();

        while let Some(node_id) = to_visit.pop() {
            if visited.contains(&node_id) || modifications >= modification_limit || !self.nodes.contains_key(node_id) {
                continue;
            }
 
            if node_map.contains_key(&self.nodes[node_id]) && node_map[&self.nodes[node_id]] != node_id {
                self.replace_index(node_id, node_map[&self.nodes[node_id]])?;
                modifications += 1;
                changed = true;
            } else {
                visited.insert(node_id);
                //Add operation nodes to the queue
                match self.nodes[node_id].operation {
                    Operation::Add(a, b) 
                        | Operation::Sub(a, b) 
                        | Operation::Mul(a, b)
                        | Operation::Div(a, b)
                        | Operation::NotEqual(a, b)
                        | Operation::Equal(a, b)
                        | Operation::LessThan(a, b)
                        | Operation::GreaterThan(a, b)
                        | Operation::GreaterThanEq(a, b)
                        | Operation::LessThanEq(a, b)
                        | Operation::MatMul(a, b)
                        | Operation::RngNormal(a, b, _)
                        | Operation::RngUniform(a, b, _)
                        | Operation::Pow(a, b) => {
                            to_visit.push(a);
                            to_visit.push(b);
                        }
                    Operation::Neg(a) 
                        | Operation::StopGradient(a)
                        | Operation::Log(a)
                        | Operation::Exp(a)
                        | Operation::TypeCast(a, _) 
                        | Operation::Transpose(a, _) 
                        | Operation::SliceInDim { node: a, start: _, stop: _, stride: _, dim: _ } 
                    | Operation::TileInDim { node: a, n_tiles: _, dim: _ }
                    | Operation::Reshape(a)
                        | Operation::ZerosLike(a) => {
                            to_visit.push(a);
                        }
                    Operation::ReduceMax { node, dim: _ }
                    | Operation::ReduceMean { node, dim: _ }
                    | Operation::ReduceArgmax { node, dim: _ }
                    | Operation::ReduceSum { node, dim: _ } => {
                        to_visit.push(node);
                    }
                    Operation::Select { pred, on_true, on_false } => {
                        to_visit.push(pred);
                        to_visit.push(on_true);
                        to_visit.push(on_false);
                    }
                    Operation::OneHot(node) => to_visit.push(node),
                    Operation::Constant(_) | Operation::Parameter(_) => {}
                }
                node_map.insert(self.nodes[node_id].clone(), node_id);
            }
            
        }

        //Recursive recall if we changed something and modifications are still available
        match changed {
            false => Ok(false),
            true => Ok(
                changed || self.extract_subterms(outputs, modification_limit - modifications)?
            ),
        }
    }
}
