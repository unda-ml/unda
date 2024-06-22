use crate::graph::{
    dtypes::check_fp_type, Context, NodeIdentifier, Result,
};

impl Context {

    pub fn relu(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        let const_zero = self.scalar(0, a_dtype)?;
        self.maximum(const_zero, a)
    }

    pub fn leaky_relu(&mut self, a: NodeIdentifier, alpha: f32) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        //TODO: force dtype to be floating point or else this just becomes normal relu
        let const_small = self.scalar(alpha, a_dtype)?;
        let small_x = self.mul(a, const_small)?;

        self.maximum(small_x, a)
    }

    pub fn sigmoid(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        let one = self.scalar(1, a_dtype)?;
        let neg_x = self.neg(a);
        let exp_x = self.exp(neg_x)?;

        let one_p_exp_x = self.add(one, exp_x)?;

        self.div(one, one_p_exp_x)
    }

    pub fn tanh(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        let two = self.scalar(2, a_dtype)?;
        let one = self.scalar(1, a_dtype)?;

        let two_a = self.mul(two, a)?;
        let sigmoid_a_2 = self.sigmoid(two_a)?;

        let two_sigmoid = self.mul(two, sigmoid_a_2)?;
        self.sub(two_sigmoid, one)
    }

    pub fn softmax(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let dtype = check_fp_type(self.nodes[a].dtype)?;

        let max = self.reduce_max(a, 0, true)?;
        let stop_grad = self.stop_gradient(max);
        let unnormalized = self.sub(a, stop_grad)?;
        let unnormalized_exp = self.exp(unnormalized)?;

        let sum = self.reduce_sum(unnormalized_exp, 0, true)?;
        let eps = self.scalar(1e-8, dtype)?;
        // prevent division by 0
        let sum = self.add(sum, eps)?;

        self.div(unnormalized_exp, sum)
    }

}
