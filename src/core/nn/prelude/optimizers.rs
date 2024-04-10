use crate::core::graph::{Context, NodeIdentifier, Result};

pub enum Optimizer {
    Adam,
    SGD,
    RMSProp
}

impl Optimizer {
    pub fn apply(&self, ctx: &mut Context, weights: NodeIdentifier, bias: NodeIdentifier, loss: NodeIdentifier, lr: NodeIdentifier) -> Result<(NodeIdentifier, NodeIdentifier)> {

        let weight_grad = ctx.diff(loss, weights)?;
        let bias_grad = ctx.diff(loss, bias)?;

        match self {
            Optimizer::SGD => {
                let weight_update = ctx.mul(weight_grad, lr)?;
                let bias_update = ctx.mul(bias_grad, lr)?;

                let weight_new = ctx.sub(weights, weight_update)?;
                let bias_new = ctx.sub(bias, bias_update)?;
                
                Ok((weight_new, bias_new))
            }
            Optimizer::Adam => {
                todo!()
            }
            Optimizer::RMSProp => {
                todo!()
            }
        }
    }
}
