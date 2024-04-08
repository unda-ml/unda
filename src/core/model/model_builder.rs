use crate::core::graph::{NodeIdentifier, Context, Shape, Result};

pub struct ModelBuilder;

impl ModelBuilder {
    pub fn dense(
        model: &mut Context,
        input_node: NodeIdentifier,
        out_size: u32,
        name: &str) -> Result<(NodeIdentifier, (NodeIdentifier, NodeIdentifier))> {
    
        let shape = model.nodes[input_node].shape.clone();
        let last_dim = shape.sizes[shape.ndims() - 1];
        let dtype = model.nodes[input_node].dtype;

        let weights_shape = Shape::from([last_dim, out_size]);
        let mut weights_name = name.to_owned();
        weights_name.push_str("_weights");
        let weights = model.parameter(weights_name, weights_shape, dtype)?;
        
        let mut bias_shape = Shape::new();
        for _ in 0..(shape.ndims() - 1) {
            bias_shape.sizes.push(1u32);
        }
        bias_shape.sizes.push(out_size);
        let mut bias_name = name.to_owned();
        bias_name.push_str("_bias");
        let bias = model.parameter(bias_name, bias_shape, dtype)?;

        let matmul_node = model.matmul(input_node, weights)?;
        let dense_node = model.add(matmul_node, bias)?;

        Ok((dense_node, (weights, bias)))
    }
}