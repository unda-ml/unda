use xla::{ElementType, PjRtBuffer, PjRtClient, XlaComputation};

use crate::{
    graph::{Context, ContextError, NodeIdentifier, Result},
};

pub struct LoadedInferenceModel {
    pub(crate) executable: xla::PjRtLoadedExecutable,
    pub(crate) n_params: usize,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
}

impl LoadedInferenceModel {
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

pub struct LoadedEvalModel {
    pub(crate) executable: xla::PjRtLoadedExecutable,
    pub(crate) n_params: usize,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) n_metrics: usize,
}

impl LoadedEvalModel {
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

pub struct LoadedGradientModel {
    pub(crate) executable: xla::PjRtLoadedExecutable,
    pub(crate) n_params: usize,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) n_metrics: usize,
}

impl LoadedGradientModel {
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
        // gradients,
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
