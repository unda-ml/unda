use crate::core::{graph::Context, neural_net::prelude::initializers::Initializers};

#[allow(dead_code)]
pub struct Model{
    model_ctx: Context,
    initializer: Initializers
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
            initializer: Initializers::Default
        }
    }
    pub fn set_initializer(&mut self, new_init: Initializers) {
        self.initializer = new_init;
    }
    pub fn compile(&mut self) -> Self {
        todo!();
    }
}
