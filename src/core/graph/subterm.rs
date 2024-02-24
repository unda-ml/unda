use super::*;
use std::collections::HashMap;

impl Context {
    /// Traverses graph context, building a hashmap of Node -> NodeIdentifier pairs
    /// If a duplicate Node is found, we can reference the other NodeIdentifier with
    /// the already existant node instead of having duplicates
    /// make sure to update entry for the modified node, as the hash will change.
    /// do not include callsite when calculating the hash.
    pub(crate) fn extract_subterms<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        _input: A,
        modification_limit: usize,
    ) -> Result<bool> {
        if modification_limit == 0 {
            return Ok(true);
        }
        let mut _node_map: HashMap<String, NodeIdentifier> = HashMap::new();
        for (mut _identifier, _node) in self.nodes.iter_mut() {
            //TODO: Build a HashMap out of all nodes, check if a node already 'exists'
            //If node exists, remove all references to its NodeIdentifier and replace with the
            //prexisting NodeIdentifier
        }
        Ok(false)
    }
}
