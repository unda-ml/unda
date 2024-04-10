use xla::{ElementType, PjRtClient};

use crate::core::{graph::{Context, Result, NodeIdentifier, ContextError}, nn::prelude::{initializers::Initializer, activations::Activation, optimizers::Optimizer}};

use super::model_builder::ModelBuilder;

pub struct Model{
    pub(crate) model_ctx: Context,
    pub(crate) client: PjRtClient,
    pub(crate) initializer: Initializer,
    pub(crate) dtype: ElementType,

    pub(crate) optimizer: Optimizer,
    pub(crate) learning_rate: NodeIdentifier,
    curr_node: Option<NodeIdentifier>,
    loss: Option<NodeIdentifier>,
    weight_bias_pairs: Vec<(NodeIdentifier, NodeIdentifier)>,
}

impl Default for Model {
    fn default() -> Self {
        Self::new(0.01, ElementType::F32)
    }
}

pub enum ClientType {
    GPU(f64),
    CPU
}

impl ClientType {
    pub(crate) fn to_client(&self) -> xla::Result<PjRtClient> {
        match self {
            ClientType::GPU(mem_frac) => PjRtClient::gpu(*mem_frac, false),
            ClientType::CPU => PjRtClient::cpu()
        }
    }
}

impl Model {
    pub fn new(learning_rate: f32, dtype: ElementType) -> Self {
        let mut ctx = Context::new();

        //Not sure if we want outward facing results unless its directly a user problem
        //such as calling layer constructors incorrectly
        let learn_rate = ctx.scalar(learning_rate, dtype)
            .expect("Error constructing model with compute graph");

        Self { 
            model_ctx: ctx,
            initializer: Initializer::Default,
            curr_node: None,
            loss: None,
            client: ClientType::CPU.to_client().expect("Error initializing CPU client"),
            dtype,
            optimizer: Optimizer::SGD,
            learning_rate: learn_rate,
            weight_bias_pairs: vec![]
        }
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
    pub fn diff(&mut self) -> Result<Vec<(NodeIdentifier, NodeIdentifier)>> {
        if let Some(loss) = self.loss {

            let mut res = Vec::new();

            for (weight, bias) in self.weight_bias_pairs.iter() {
                //Collect gradients of weights and biases
                let (weight_new, bias_new) = self.optimizer.apply(&mut self.model_ctx,
                                                                  *weight,
                                                                  *bias,
                                                                  loss,
                                                                  self.learning_rate)?;
                res.push((weight_new, bias_new));
            }
            Ok(res)
        } else {
            Err(ContextError::InvalidDiffError())
        }
    }
}
