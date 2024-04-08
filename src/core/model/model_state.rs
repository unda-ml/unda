use crate::core::{graph::{Context, Result, NodeIdentifier}, neural_net::prelude::initializers::Initializers};

use super::model_builder::ModelBuilder;

#[allow(dead_code)]
pub struct Model{
    model_ctx: Context,
    initializer: Initializers,

    curr_node: Option<NodeIdentifier>,
    weight_bias_pairs: Vec<(NodeIdentifier, NodeIdentifier)>
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        Self { 
            model_ctx: Context::new(),
            initializer: Initializers::Default,
            curr_node: None,
            weight_bias_pairs: vec![]
        }
    }
    pub fn set_initializer(&mut self, new_init: Initializers) {
        self.initializer = new_init;
    }
    pub fn compile(&mut self) -> Self {
        todo!();
    }
    pub fn dense(&mut self, out_size: u32, name: &str) -> Result<()> {
        if let Some(node) = self.curr_node {
            //Append dense layer onto end of current context
            let (out, (weights_curr, bias_curr)) = ModelBuilder::dense(&mut self.model_ctx, node, out_size, name)?;
            self.curr_node = Some(out);
            self.weight_bias_pairs.push((weights_curr, bias_curr));

            //TODO create backwards pass here too? Potentially.

        } else {
            //Create initial dense layer with input params
            todo!();
        }

        Ok(())
    }
}
