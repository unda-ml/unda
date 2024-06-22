use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::ContextError;

pub trait Optimizer<const P: usize, const S: usize, U> {
    fn get_step(&self) -> &Context;
    fn get_old_params(&self) -> [NodeIdentifier; P];
    fn get_gradients(&self) -> [NodeIdentifier; P];
    fn get_new_params(&self) -> [NodeIdentifier; P];
    fn get_old_state(&self) -> [NodeIdentifier; S];
    fn get_new_state(&self) -> [NodeIdentifier; S];
    fn initialize(user_params: U, model_params: &[NodeIdentifier; P], model: &Context) -> Self;
}

pub struct SGD<const P: usize> {
    step: Context,
    old_params: [NodeIdentifier; P],
    grads: [NodeIdentifier; P],
    new_params: [NodeIdentifier; P],
    pub learning_rate: f32,
}

impl<const P: usize> Optimizer<P, 0, f32> for SGD<P> {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn get_old_params(&self) -> [NodeIdentifier; P] {
        self.old_params
    }
    fn get_gradients(&self) -> [NodeIdentifier; P] {
        self.grads
    }
    fn get_new_params(&self) -> [NodeIdentifier; P] {
        self.new_params
    }
    fn get_old_state(&self) -> [NodeIdentifier; 0] {
        []
    }
    fn get_new_state(&self) -> [NodeIdentifier; 0] {
        []
    }
    fn initialize(learning_rate: f32, model_params: &[NodeIdentifier; P], model: &Context) -> SGD<P> {
        let build = || {
            let mut step = Context::new();

            let dtype = model.nodes[model_params[0]].dtype;
            let lr = step.scalar(learning_rate, dtype)?;

            let mut old_params = [model_params[0]; P];
            let mut grads = [model_params[0]; P];
            let mut new_params = [model_params[0]; P];

            for (i, node_id) in model_params.iter().enumerate() {
                let model_param = &model.nodes[*node_id];
                let old_param = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                let grad = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                let mul = step.mul(lr, grad)?;
                let new_param = step.sub(old_param, mul)?;
                old_params[i] = old_param;
                grads[i] = grad;
                new_params[i] = new_param;
            }

            Ok::<SGD<P>, ContextError>(SGD {
                step,
                old_params,
                grads,
                new_params,
                learning_rate,
            })
        };

        build().expect("Failed to initialize SGD")
    }
}
