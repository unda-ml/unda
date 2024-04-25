use xla::{ElementType, PjRtBuffer, PjRtClient, XlaComputation};

use super::loaded_model::{LoadedEvalModel, LoadedGradientModel, LoadedInferenceModel};
use crate::core::{
    graph::{Context, ContextError, NodeIdentifier, Result},
    nn::prelude::{activations::Activation, initializers::Initializer, optimizers::Optimizer},
};

pub struct Model<P> {
    // forward computation of the network without loss
    pub(crate) network: Context,
    // wraps the node identifiers for the parameters of the network
    // will be buffers at execution
    pub(crate) params: P,
    // list of input nodes
    // will be literals not buffers at executation
    pub(crate) inputs: Vec<NodeIdentifier>,
    // list of output nodes
    // will be buffers at execution
    pub(crate) outputs: Vec<NodeIdentifier>,

    // separate context which takes parameters, outputs, and targets
    pub(crate) compute_metrics: Context,
    // additional inputs to compute_metrics as the targets of the supervised learning algorithm
    pub(crate) targets: Vec<NodeIdentifier>,
    // index into compute_metrics context to find differentiable loss function
    pub(crate) loss: NodeIdentifier,
    // points to additional metrics like accuracy
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,

    // executes the network context without evaluating metrics
    pub(crate) inference_computation: xla::XlaComputation,
    // executes the network and training metrics
    pub(crate) evalutation_computation: xla::XlaComputation,
    // executes the network and training metrics and returns derivatives of the parameters
    pub(crate) gradient_computation: xla::XlaComputation,
}

impl<P: From<Vec<NodeIdentifier>> + Into<Vec<NodeIdentifier>>> Model<P> {
    // this function should
    // build the inference_computation from the network context
    // fuse the network and compute_metrics contexts and build the evaluation_computation
    // further augment the context to return derivatives of all params and then build the gradient_computation
    pub fn new(
        network: Context,
        params: P,
        inputs: Vec<NodeIdentifier>,
        outputs: Vec<NodeIdentifier>,
        compute_metrics: Context,
        targets: Vec<NodeIdentifier>,
        loss: NodeIdentifier,
        auxiliary_metrics: Vec<NodeIdentifier>,
    ) {
        panic!("Not yet implemented");
    }

    pub fn compile_inference(
        &self,
        client: xla::PjRtClient,
    ) -> Result<LoadedInferenceModel> {
        panic!("Not yet implemented")
    }
    pub fn compile_evaluation(
        &self,
        client: xla::PjRtClient,
    ) -> Result<LoadedEvalModel> {
        panic!("Not yet implemented")
    }
    pub fn compile_gradient(
        &self,
        client: xla::PjRtClient,
    ) -> Result<LoadedGradientModel> {
        panic!("Not yet implemented")
    }
}
