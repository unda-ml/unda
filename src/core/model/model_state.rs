use xla::ElementType;

use crate::core::{graph::{Context, Result, NodeIdentifier, ContextError}, nn::prelude::{initializers::Initializer, activations::Activation}};

use super::model_builder::ModelBuilder;

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
        Self::new(0.01)
    }
}

impl Model {
    //TODO: Maybe allow specifying a dtype as well,
    //could become the default dtype from there as well
    pub fn new(learning_rate: f32) -> Self {
        let mut ctx = Context::new();

        //Not sure if we want outward facing results unless its directly a user problem
        //such as calling layer constructors incorrectly
        let learn_rate = ctx.scalar(learning_rate, ElementType::F32)
            .expect("Error constructing model with compute graph");

        Self { 
            model_ctx: ctx,
            initializer: Initializer::Default,
            curr_node: None,
            loss: None,
            learning_rate: learn_rate,
            weight_bias_pairs: vec![]
        }
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
                                                                       node, out_size,
                                                                       &self.initializer, &name)?;
            self.weight_bias_pairs.push((weights_curr, bias_curr));
            let activation_applied = activation.apply(out, &mut self.model_ctx)?;

            self.curr_node = Some(activation_applied);
            Ok(())
        } else {
            //No input params have been set yet, error(at least I think this is valid behavior)
            Err(ContextError::InvalidLayerConstructionError("Dense".to_owned()))
        }
    }
    pub fn diff(&mut self) -> Result<()> {
        if let Some(loss) = self.loss {
            for (weight, bias) in self.weight_bias_pairs.iter().rev() {
                //Collect gradients of weights and biases
                let weight_grad = self.model_ctx.diff(loss, *weight)?;
                let bias_grad = self.model_ctx.diff(loss, *bias)?;

                let weight_update = self.model_ctx.mul(weight_grad, self.learning_rate)?;
                let bias_update = self.model_ctx.mul(bias_grad, self.learning_rate)?;

                //TODO store weight bias updates in context 
            }
            Ok(())
        } else {
            Err(ContextError::InvalidDiffError())
        }
    }
}
