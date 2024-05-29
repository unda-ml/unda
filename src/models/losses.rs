use crate::graph::{
    callsite::callsite, dtypes::check_real_type, Context, ContextError, NodeIdentifier, Result,
};

impl Context {
    pub fn mean_cross_entropy(
        &mut self,
        prediction_probabilities: NodeIdentifier,
        one_hot_labels: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let dtype = check_real_type(self.nodes[prediction_probabilities].dtype)?;
        if dtype != self.nodes[one_hot_labels].dtype {
            return Err(ContextError::IncompatibleOperandTypes(
                dtype,
                self.nodes[one_hot_labels].dtype,
                callsite!(1),
            ));
        }

        let eps = self.scalar(1e-8, dtype)?;
        // prevent logarithm of zero
        let offset = self.add(prediction_probabilities, eps)?;
        let log = self.log(offset)?;
        let neglog = self.neg(log);
        let mul = self.mul(one_hot_labels, neglog)?;
        let sum = self.reduce_sum(mul, 1, false)?;
        self.reduce_mean(sum, 0, false)
    }
}
