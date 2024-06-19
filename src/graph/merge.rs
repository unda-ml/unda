use std::collections::HashMap;

use super::Node;
use super::NodeIdentifier;
use super::Operation;
use super::Result;

use super::Context;

impl Context {
    pub fn merge_graphs(&mut self, other: &Context, other_outputs: &[NodeIdentifier], desired_remaps: &[NodeIdentifier]) -> Result<Vec<NodeIdentifier>> {

        let mut addition_queue = other.inputs(other_outputs)?;

        let mut old_to_new: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        

        while let Some(old_node) = addition_queue.pop() {
            
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

    fn inputs(&self, outputs: &[NodeIdentifier]) -> Result<Vec<NodeIdentifier>> {
        let mut inputs = vec![];
        let mut queue = outputs.to_vec();

        while let Some(current_node) = queue.pop() {
            
        }

        Ok(inputs)
    }
}
