use super::NodeIdentifier;
use super::Result;

use super::Context;

impl Context {
    pub fn merge_graphs(&mut self, other: &Context) -> Result<()> {

        let other_slotmap = other.nodes.clone();
        for (key, node) in other_slotmap {
            self.nodes[key] = node;
        }

        Ok(())
    }

    pub fn find_and_replace_params(&mut self, param_reps: &[(&str, &[NodeIdentifier])]) -> Result<()> {
        Ok(())
    }
}
