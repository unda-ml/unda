use crate::core::graph::Context;

#[allow(dead_code)]
pub struct Model{
    model_ctx: Context
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        todo!();
    }
    pub fn compile(&mut self) -> Self {
        todo!();
    }
}
