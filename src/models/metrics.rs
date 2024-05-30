use crate::graph::{Context, dtypes::check_int_type, NodeIdentifier, Result};

impl Context {

    // assumes dense_predictions is rank 2 with dimension 0 being batch and dimension 1 being predictions
    // assumes sparse_label_vector is rank 1 i64 of class labels
    pub fn accuracy(
        &mut self,
        dense_predictions: NodeIdentifier,
        sparse_label_vector: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let converted_labels = match check_int_type(self.nodes[sparse_label_vector].dtype) {
            Ok(_) => self.type_cast(sparse_label_vector, xla::ElementType::S64),
            Err(e) => return Err(e),
        };
        let sparse_predictions = self.reduce_argmax(dense_predictions, 1, false)?;
        let compare = self.eq(sparse_predictions, converted_labels)?;
        let converted = self.type_cast(compare, xla::ElementType::F32);
        self.reduce_mean(converted, 0, false)
    }

}