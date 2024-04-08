use crate::core::graph::{Context, NodeIdentifier, Result};

pub enum Activation {
    ReLU,
    LeakyReLU(f32),
    Tanh,
    Sigmoid,
    Softmax
}

impl Activation {
    pub fn apply(&self, last_node: NodeIdentifier, ctx: &mut Context) -> Result<NodeIdentifier> {
        match self {
            Activation::ReLU => ctx.relu(last_node),
            Activation::LeakyReLU(alpha) => ctx.leaky_relu(last_node, *alpha),
            Activation::Tanh => ctx.tanh(last_node),
            Activation::Sigmoid => ctx.sigmoid(last_node),
            Activation::Softmax => ctx.softmax(last_node)
        }
    }
}
