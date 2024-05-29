use xla::{ElementType, PjRtBuffer, PjRtClient, XlaComputation};

use crate::{
    graph::{Context, ContextError, NodeIdentifier, Result},
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

    // executes the network context without Evaluationuating metrics
    pub(crate) inference_computation: xla::XlaComputation,
    // executes the network and training metrics
    pub(crate) Evaluationutation_computation: xla::XlaComputation,
    // executes the network and training metrics and returns derivatives of the parameters
    pub(crate) Training_computation: xla::XlaComputation,
}

impl<P: From<Vec<NodeIdentifier>> + Into<Vec<NodeIdentifier>>> Model<P> {
    // this function should
    // build the inference_computation from the network context
    // fuse the network and compute_metrics contexts and build the Evaluationuation_computation
    // further augment the context to return derivatives of all params and then build the Training_computation
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
    ) -> Result<SupervisedInferenceExecutable> {
        panic!("Not yet implemented")
    }
    pub fn compile_evaluationn(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedEvaluationExecutable> {
        panic!("Not yet implemented")
    }
    pub fn compile_training(
        &self,
        client: xla::PjRtClient,
    ) -> Result<LoadedTrainingModel> {
        panic!("Not yet implemented")
    }
}


pub struct SupervisedInferenceExecutable {
    pub(crate) executable: xla::PjRtLoadedExecutable,
    pub(crate) n_params: usize,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
}

impl SupervisedInferenceExecutable {
    pub fn run<P: Into<Vec<xla::PjRtBuffer>> + From<Vec<xla::PjRtBuffer>>>(
        &self,
        parameters: P,
        inputs: Vec<xla::Literal>,
    ) -> Result<(
        // network outputs
        Vec<PjRtBuffer>,
    )> {
        let param_vec = parameters.into();
        let n_params = param_vec.len();
        //let buffer_inputs = inputs
        //    .iter()
        //    .map(|x| self.executable.client().buffer_from_host_literal(None, x))
        //    .collect::<Vec<xla::PjRtBuffer>>();
        panic!("Not yet implemented")
    }
}

pub struct SupervisedEvaluationExecutable {
    pub(crate) executable: xla::PjRtLoadedExecutable,
    pub(crate) n_params: usize,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) n_metrics: usize,
}

impl SupervisedEvaluationExecutable {
    pub fn run<P: Into<Vec<xla::PjRtBuffer>> + From<Vec<xla::PjRtBuffer>>>(
        &self,
        parameters: P,
        inputs: Vec<xla::Literal>,
        targets: Vec<xla::Literal>
    ) -> Result<(
        // network outputs
        Vec<PjRtBuffer>,
        // metrics
        Vec<PjRtBuffer>
    )> {
        let param_vec = parameters.into();
        let n_params = param_vec.len();
        //let buffer_inputs = inputs
        //    .iter()
        //    .map(|x| self.executable.client().buffer_from_host_literal(None, x))
        //    .collect::<Vec<xla::PjRtBuffer>>();
        panic!("Not yet implemented")
    }
}

pub struct LoadedTrainingModel {
    pub(crate) executable: xla::PjRtLoadedExecutable,
    pub(crate) n_params: usize,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) n_metrics: usize,
}

impl LoadedTrainingModel {
    pub fn run<P: Into<Vec<xla::PjRtBuffer>> + From<Vec<xla::PjRtBuffer>>>(
        &self,
        parameters: P,
        inputs: Vec<xla::Literal>,
        targets: Vec<xla::Literal>
    ) -> Result<(
        // network outputs
        Vec<PjRtBuffer>,
        // metrics
        Vec<PjRtBuffer>,
        // Trainings,
        P
    )> {
        let param_vec = parameters.into();
        let n_params = param_vec.len();
        //let buffer_inputs = inputs
        //    .iter()
        //    .map(|x| self.executable.client().buffer_from_host_literal(None, x))
        //    .collect::<Vec<xla::PjRtBuffer>>();
        panic!("Not yet implemented")
    }
}
