use crate::core::graph::{Result, NodeIdentifier, Context};

pub enum Initializer {
    He,
    Xavier,
    Default
}

impl Initializer {
    pub fn initialize(&self, ctx: &mut Context, on_node: NodeIdentifier, n: usize) -> Result<NodeIdentifier> {
        match self {
            Initializer::He => {
                let dep_size = ctx.nodes[on_node].shape.sizes.to_owned();
                let dep_dtype = ctx.nodes[on_node].dtype;
                
                let mu = ctx.scalar(0, dep_dtype)?;
                
                let n_node = ctx.scalar(n as u32, dep_dtype)?;
                let two = ctx.scalar(2, dep_dtype)?;

                let half = ctx.scalar(0.5, dep_dtype)?;

                let two_over_n = ctx.div(two, n_node)?;
                let sigma = ctx.pow(two_over_n, half)?;

                let rand_vals = ctx.rng_normal(mu, sigma, dep_size.as_slice())?;

                ctx.add(on_node, rand_vals)

            },
            Initializer::Xavier => {
                let dep_size = ctx.nodes[on_node].shape.sizes.to_owned();
                let dep_dtype = ctx.nodes[on_node].dtype;
                
                let one = ctx.scalar(1, dep_dtype)?;
                let half = ctx.scalar(0.5, dep_dtype)?;
                let n_node = ctx.scalar(n as u32, dep_dtype)?;

                let sqrt_n = ctx.pow(n_node, half)?;

                let max = ctx.div(one, sqrt_n)?;
                let min = ctx.neg(max);

                let rand_vals = ctx.rng_uniform(min, max, dep_size.as_slice())?;

                ctx.add(on_node, rand_vals)

            },
            Initializer::Default => {
                let dep_size = ctx.nodes[on_node].shape.sizes.to_owned();
                let dep_dtype = ctx.nodes[on_node].dtype;
                
                let min = ctx.scalar(-1, dep_dtype)?;
                let max = ctx.scalar(1, dep_dtype)?;

                let rand_vals = ctx.rng_uniform(min, max, dep_size.as_slice())?;

                ctx.add(on_node, rand_vals)
            }
        }
    }
}
