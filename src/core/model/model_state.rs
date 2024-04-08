use xla::ElementType;

use crate::core::{graph::{Context, Result, NodeIdentifier}, nn::prelude::{initializers::Initializer, activations::Activation}};

use super::model_builder::ModelBuilder;

#[allow(dead_code)]
pub struct Model{
    model_ctx: Context,
    initializer: Initializer,

    curr_node: Option<NodeIdentifier>,
    loss: Option<NodeIdentifier>,
    learning_rate: NodeIdentifier,
    weight_bias_pairs: Vec<(NodeIdentifier, NodeIdentifier)>,
}

impl Default for Model {
    fn default() -> Self {
        Self::new(0.01).expect("Error constructing model")
    }
}

impl Model {
    //TODO: Maybe allow specifying a dtype as well,
    //could become the default dtype from there as well
    pub fn new(learning_rate: f32) -> Result<Self> {
        let mut ctx = Context::new();
        let learn_rate = ctx.scalar(learning_rate, ElementType::F32)?;

        Ok(Self { 
            model_ctx: ctx,
            initializer: Initializer::Default,
            curr_node: None,
            loss: None,
            learning_rate: learn_rate,
            weight_bias_pairs: vec![]
        })
    }
    pub fn set_initializer(&mut self, new_init: Initializer) {
        self.initializer = new_init;
    }
    pub fn compile(&mut self) -> Self {
        todo!();
    }
    pub fn dense(&mut self, out_size: u32, activation: Activation) -> Result<()> {
        if let Some(node) = self.curr_node {
            //Append dense layer onto end of current context
            let mut name = "dense_".to_owned();
            name.push_str(&(self.weight_bias_pairs.len() + 1).to_string());

            let (out, (weights_curr, bias_curr)) = ModelBuilder::dense(&mut self.model_ctx, 
                                                                       node, out_size, &name)?;
            self.weight_bias_pairs.push((weights_curr, bias_curr));
            let activation_applied = activation.apply(out, &mut self.model_ctx)?;

            self.curr_node = Some(activation_applied);

        } else {
            //Create initial dense layer with input params
            todo!();
        }
        Ok(())
    }
    pub fn diff(&mut self) -> Result<()> {
        for (weight, bias) in self.weight_bias_pairs.iter().rev() {
            //Collect gradients of weights and biases
        }
        Ok(())
    }
}
