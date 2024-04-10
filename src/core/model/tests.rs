#[cfg(test)]
mod tests {
    use crate::core::{model::model_state::Model, nn::prelude::activations::Activation};

    #[test]
    fn model_panics_on_dense_before_param_init() {
        let mut model = Model::default();
        
        assert!(model.dense(10, Activation::ReLU).is_err());
    }
}
