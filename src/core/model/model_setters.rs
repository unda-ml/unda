use xla::ElementType;

use crate::core::{nn::prelude::{initializers::Initializer, optimizers::Optimizer}, graph::Result};

use super::model_state::{Model, ClientType};

impl Model {
    pub fn set_initializer(&mut self, new_init: Initializer) {
        self.initializer = new_init;
    }

    pub fn set_client(&mut self, client: ClientType) {
        self.client = client.to_client().expect("Error setting client type");
    }

    pub fn set_learning_rate(&mut self, rate: f64, dtype: ElementType) -> Result<()> {
        self.dtype = dtype;
        self.learning_rate = self.model_ctx.scalar(rate, dtype)?;
        Ok(())
    }

    pub fn set_optimizer(&mut self, optimizer: Optimizer) {
        self.optimizer = optimizer;
    }

}
