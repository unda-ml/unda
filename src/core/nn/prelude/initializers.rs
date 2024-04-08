use crate::core::graph::{Result, NodeIdentifier};

pub enum Initializer {
    He,
    Xavier,
    Default
}

impl Initializer {
    pub fn initialize(&self, on_node: NodeIdentifier, n: usize) -> Result<NodeIdentifier> {
        match self {
            Initializer::He => {},
            Initializer::Xavier => {},
            Initializer::Default => {}
        }
        Ok(on_node)
    }
}
