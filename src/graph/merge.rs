use std::collections::HashMap;

use super::NodeIdentifier;
use super::Operation;
use super::Result;

use super::Context;


impl Context {
    pub fn merge_graphs(&mut self, other: &Context, desired_remaps: &[NodeIdentifier]) -> Result<Vec<NodeIdentifier>> {

        let mut old_to_new: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        let mut addition_queue = other.inputs();

        while let Some(old_node) = addition_queue.pop() {
            let new_id = self.nodes.insert(other.nodes[old_node].clone());

            match self.nodes[new_id].operation {
                Operation::Constant(_) => self.constants.push(new_id),
                Operation::Parameter(_) => self.parameters.push(new_id),
                _ => (),
            }

            if let Some(deps) = other.dependent_nodes.clone().get(&old_node) {
                for node in deps {
                    addition_queue.insert(0, *node);
                }
            }

            old_to_new.insert(old_node, new_id);
        }

        for (old_node, old_deps) in other.dependent_nodes.clone() {
            let new_node = old_to_new[&old_node];
            let new_deps = old_deps.iter().map(|old| old_to_new[old]).collect::<Vec<NodeIdentifier>>();

            self.dependent_nodes.insert(new_node, new_deps);
        }

        for (old, new) in old_to_new.iter() {
            self.replace_index(*old, *new)?;
        }

        let mut new_remaps = vec![];

        for old in desired_remaps {
            new_remaps.push(old_to_new[old])
        }


        Ok(new_remaps)
    }

    pub fn find_and_replace_params(&mut self, param_reps: &[(&str, &[NodeIdentifier])]) -> Result<()> {
        for (param_name, rep_with) in param_reps {
            let params_with_name: Vec<NodeIdentifier> = self.nodes.clone().into_iter().filter(|(_, node)| {
                match node.operation.clone() {
                    Operation::Parameter(name) => name.contains(param_name),
                    _ => false
                }
            }).map(|(id, _)| id).collect();

            if params_with_name.len() != rep_with.len() {
                return Err(super::ContextError::IncorrectOutputSizeError(rep_with.len(), params_with_name.len()));
            }

            for i in 0..params_with_name.len() {
                self.replace_index(params_with_name[i], rep_with[i])?;
                            }
        }

        Ok(())
    }

    fn inputs(&self) -> Vec<NodeIdentifier> {
        let mut res = self.parameters.clone();
        res.extend(self.constants.iter());

        res
    }
}
