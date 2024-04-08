use crate::core::graph::{Result, NodeIdentifier};

pub enum Initializer {
    He,
    Xavier,
    Default
}

impl Initializer {
    pub fn initialize(&self, on_node: NodeIdentifier) -> Result<NodeIdentifier> {
        //TODO Need XLA's random number generator functions as ops
        Ok(on_node)
    }
}
