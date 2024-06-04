use xla::{Literal, PjRtBuffer};

use crate::graph::{Context, ContextError, Node, NodeIdentifier, Result};

pub struct SupervisedModel<
    const P: usize,
    const I: usize,
    const O: usize,
    const T: usize,
    const M: usize,
> {
    // forward computation of the network without loss
    pub(crate) network: Context,
    // wraps the node identifiers for the parameters of the network
    // will be buffers at execution
    pub(crate) params: [NodeIdentifier; P],
    // list of input nodes
    // will be literals not buffers at executation
    pub(crate) inputs: [NodeIdentifier; I],
    // list of output nodes
    // will be buffers at execution
    pub(crate) outputs: [NodeIdentifier; O],

    // separate context which takes parameters, outputs, and targets
    pub(crate) compute_metrics: Context,
    // additional inputs to compute_metrics as the targets of the supervised learning algorithm
    pub(crate) targets: [NodeIdentifier; T],
    // index into compute_metrics context to find differentiable loss function
    pub(crate) loss: NodeIdentifier,
    // points to additional metrics like accuracy
    pub(crate) auxiliary_metrics: [NodeIdentifier; M],

    // executes the network context without Evaluationuating metrics
    pub(crate) inference_computation: xla::XlaComputation,
    // executes the network and gradient metrics
    pub(crate) evaluation_computation: xla::XlaComputation,
    // executes the network and gradient metrics and returns derivatives of the parameters
    pub(crate) gradient_computation: xla::XlaComputation,
}

impl<const P: usize, const I: usize, const O: usize, const T: usize, const M: usize>
    SupervisedModel<P, I, O, T, M>
{
    // this function should
    // build the inference_computation from the network context
    // fuse the network and compute_metrics contexts and build the evaluation_computation
    // further augment the context to return derivatives of all params and then build the gradient_computation
    pub fn new(
        network: Context,
        params: [NodeIdentifier; P],
        inputs: [NodeIdentifier; I],
        outputs: [NodeIdentifier; O],
        compute_metrics: Context,
        targets: [NodeIdentifier; T],
        loss: NodeIdentifier,
        auxiliary_metrics: [NodeIdentifier; M],
    ) {
        panic!("Not yet implemented");
    }

    pub fn compile_inference(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedInferenceExecutable<P, I, O>> {
        panic!("Not yet implemented")
    }
    pub fn compile_evaluation(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedEvaluationExecutable<P, I, O, T, M>> {
        panic!("Not yet implemented")
    }
    pub fn compile_gradient(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedGradientExecutable<P, I, O, T, M>> {
        panic!("Not yet implemented")
    }
}

pub struct SupervisedInferenceExecutable<const P: usize, const I: usize, const O: usize> {
    pub(crate) executable: xla::PjRtLoadedExecutable,
}

impl<const P: usize, const I: usize, const O: usize> SupervisedInferenceExecutable<P, I, O> {
    pub fn run(
        &self,
        parameters: [PjRtBuffer; P],
        inputs: [Literal; I],
    ) -> Result<(
        // network outputs
        [PjRtBuffer; O],
    )> {
        panic!("Not yet implemented")
    }
}

pub struct SupervisedEvaluationExecutable<
    const P: usize,
    const I: usize,
    const O: usize,
    const T: usize,
    const M: usize,
> {
    pub(crate) executable: xla::PjRtLoadedExecutable,
}

impl<const P: usize, const I: usize, const O: usize, const T: usize, const M: usize>
    SupervisedEvaluationExecutable<P, I, O, T, M>
{
    pub fn run(
        &self,
        parameters: [PjRtBuffer; P],
        inputs: [Literal; I],
        targets: [Literal; T],
    ) -> Result<(
        // network outputs
        [PjRtBuffer; O],
        // loss
        PjRtBuffer,
        // auxiliary metrics
        [PjRtBuffer; M],
    )> {
        panic!("Not yet implemented")
    }
}

pub struct SupervisedGradientExecutable<
    const P: usize,
    const I: usize,
    const O: usize,
    const T: usize,
    const M: usize,
> {
    pub(crate) executable: xla::PjRtLoadedExecutable,
}

impl<const P: usize, const I: usize, const O: usize, const T: usize, const M: usize>
    SupervisedGradientExecutable<P, I, O, T, M>
{
    pub fn run(
        &self,
        parameters: [PjRtBuffer; P],
        inputs: [Literal; I],
        targets: [Literal; T],
    ) -> Result<(
        // network outputs
        [PjRtBuffer; O],
        // loss
        PjRtBuffer,
        // auxiliary metrics
        [PjRtBuffer; M],
        // gradients
        [PjRtBuffer; P],
    )> {
        panic!("Not yet implemented")
    }
}
