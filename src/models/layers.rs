use crate::{
    graph::{Context, NodeIdentifier, Result, Shape},
    tree::Tree,
};

use super::initializers::Initializer;

pub struct ConvParams<T> {
    kernel: Box<T>,
    bias: Box<T>,
}

impl<T> Tree<T, ()> for ConvParams<T> {
    fn flatten(self) -> (Vec<Box<T>>, ()) {
        (vec![self.kernel, self.bias], ())
    }
    fn unflatten(flat: &mut Vec<Box<T>>, _: ()) -> Self {
        let kernel = match flat.pop() {
            None => panic!("Tried to unflatten empty vector into ConvParams!"),
            Some(w) => w,
        };
        let bias = match flat.pop() {
            None => panic!("Tried to unflatten vector of length 1 in to ConvParams"),
            Some(b) => b,
        };
        ConvParams { kernel, bias }
    }
}

impl Context {
    pub fn dense<IW: Initializer, IB: Initializer>(
        &mut self,
        input_node: NodeIdentifier,
        out_size: u32,
        kernel_initializer: IW,
        bias_initializer: IB,
        name: &str,
    ) -> Result<(NodeIdentifier, ConvParams<NodeIdentifier>, ConvParams<xla::Literal>)> {
        panic!("Not implemented!")
        /*
        let shape = self.nodes[input_node].shape.clone();
        let last_dim = shape.sizes[shape.ndims() - 1];
        let dtype = self.nodes[input_node].dtype;

        let kernel_shape = Shape::from([last_dim, out_size]);
        let mut kernel_name = name.to_owned();
        kernel_name.push_str("_kernel");
        let kernel = self.parameter(kernel_name, kernel_shape, dtype)?;
        let kernel_init = initializer.initialize(
            self,
            kernel,
            self.nodes[input_node].shape.sizes[1] as usize,
        )?;

        let mut bias_shape = Shape::new();
        for _ in 0..(shape.ndims() - 1) {
            bias_shape.sizes.push(1u32);
        }
        bias_shape.sizes.push(out_size);
        let mut bias_name = name.to_owned();
        bias_name.push_str("_bias");
        let bias = self.parameter(bias_name, bias_shape, dtype)?;

        let matmul_node = self.matmul(input_node, kernel_init)?;
        let dense_node = self.add(matmul_node, bias)?;

        Ok((dense_node, (kernel_init, bias)))
        */
    }
}
