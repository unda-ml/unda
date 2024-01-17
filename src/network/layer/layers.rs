use crate::network::{input::Input, matrix::Matrix, activations::Activations, network::Network};
use rayon::prelude::ParallelIterator;
use serde::{Serialize, Deserialize};

use super::{dense::Dense};

#[typetag::serde]
pub trait Layer{
    fn forward(&self, inputs: &Box<dyn Input>) -> Box<dyn Input> {
        Box::new(Matrix::new_empty(0,0))
    }
    fn backward(&mut self, gradients: Box<dyn Input>, errors: Box<dyn Input>, data: Box<dyn Input>) -> Box<dyn Input>; 
    fn get_data(&self) -> Box<dyn Input>;
    fn set_data(&mut self, data: &Box<dyn Input>); 
    fn get_activation(&self) -> Option<Activations> {
        None
    }
    ///Input shape the layer takes in
    fn shape(&self) -> (usize,usize,usize);
    fn get_loss(&self) -> f32;
    fn update_gradient(&self) -> Box<dyn Input>;
}

#[derive(Serialize, Deserialize, Clone)]
pub enum LayerTypes{
    ///DENSE: Nodes, Activation Function, Learning Rate
    DENSE(usize, Activations, f32),
    //NETWORK(Vec<LayerTypes>, usize),
    //CONV: Kernel Size, stride, Learning Rate
    //CONV((usize, usize), usize, f32),    
}

impl LayerTypes{
    pub fn to_layer(&self, prev_cols: usize, seed: &Option<String>, input_size: usize) -> Box<dyn Layer> {
        return match self {
            LayerTypes::DENSE(rows, activation, learning) => Box::new(Dense::new(rows.clone(), prev_cols, activation.clone(), learning.clone(), seed, input_size)),
            /*LayerTypes::NETWORK(layers, batch_size) => {
                let mut new_net: Network = Network::new(*batch_size);
                layers.iter().for_each(|layer| {
                    new_net.add_layer(layer.clone());
                });
                new_net.compile();
                Box::new(new_net)
            },*/
            //LayerTypes::CONV(shape, stride, learning) => Box::new()
        };
    }
    pub fn get_size(&self) -> usize{
        return match self{
            LayerTypes::DENSE(rows, _, _) => *rows,
            _ => 0
        }
    }
}
