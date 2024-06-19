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

            if let Some(deps) = other.dependent_nodes.clone().get(&old_node) {
                for node in deps {
                    addition_queue.insert(0, *node);
                }
            }

            old_to_new.insert(old_node, new_id);
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
        /*let mut inputs = vec![];
        let mut queue = outputs.to_vec();

        while let Some(current_node) = queue.pop() {
            todo!()
        }

        Ok(inputs)*/
        self.nodes.clone().into_iter().filter(|(_, node)| {
            match node.operation {
                Operation::Constant(_) | Operation::Parameter(_) => true,
                _ => false
            }
        }).map(|(id, _)| id).collect::<Vec<NodeIdentifier>>()
    }
}
