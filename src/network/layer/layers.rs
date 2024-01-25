use std::ops::Range;

use crate::network::{input::Input, matrix::Matrix, activations::Activations};

use rand::RngCore;
use serde::{Serialize, Deserialize};

use super::{dense::Dense, pair::GradientPair};

#[typetag::serde]
pub trait Layer: Send + Sync{
    ///Propegates the data forward through 1 layer and returns
    ///the data created from that respective layer as a dynamic Input object
    ///
    ///Ex: A Dense layer will take in a 1 dimensional vector and return out a 1 dimensional vector
    ///A convolutional layer will return a 3 dimensional matrix of all filter applications
    ///
    ///Both of these types implement the Input trait
    fn forward(&self, _inputs: &Box<dyn Input>) -> Box<dyn Input> {
        Box::new(Matrix::new_empty(0,0))
    }
    ///Propegates the neural network backwards once, updating internal weights and biases
    ///accodrding to a derived gradient and obtaining a new metric for loss
    fn backward(&mut self, gradients: Box<dyn Input>, errors: Box<dyn Input>, data: Box<dyn Input>) -> Box<dyn Input>; 
    fn avg_gradient(&self, gradients: Vec<&Box<dyn Input>>) -> Box<dyn Input>;
    ///Updates the model's current weights and biases according to a paired gradient in the format
    ///(Bias Gradient, Weight Gradient)
    fn update_gradients(&mut self, gradient_pair: (&Box<dyn Input>, &Box<dyn Input>), clip: Option<Range<f32>>);//, noise: &f32);
    ///Returns the data currently found at this layer. The data is dependent on the current
    ///iteration and input data going through
    fn get_data(&self) -> Box<dyn Input>;
    ///Sets the current data at this layer
    fn set_data(&mut self, data: &Box<dyn Input>); 
    ///Generates the error found at this layer given the error of the previous layer. Can be used
    ///to recursively travel backwards through the error
    fn update_errors(&self, errors: Box<dyn Input>) -> Box<dyn Input>;
    ///Develops the gradient for this layer in the GradientPair (*Bias Gradient*, *Weight Gradient*)
    ///Utilizes the current data found at the layer(passed in because this is used with
    ///asynchronous updating) and the data found at the previous layer to derive the new gradients
    fn get_gradients(&self, data: &Box<dyn Input>, data_at: &Box<dyn Input>, errors: &Box<dyn Input>) -> GradientPair; 
    ///Returns the activation function found at this layer, currently all implemented layers have
    ///an activation function, so the Option return type is really for future safety
    fn get_activation(&self) -> Option<Activations> {
        None
    }
    ///Input shape the layer takes in
    fn shape(&self) -> (usize,usize,usize);
    ///The current loss found at this layer specifically
    fn get_loss(&self) -> f32;
    ///Returns the unmodified gradient of this layer, to be used on the next layer
    fn update_gradient(&self) -> Box<dyn Input>;
    fn get_weights(&self) -> Box<dyn Input>;
    fn get_biases(&self) -> Box<dyn Input>;
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
    pub fn to_layer(&self, prev_cols: usize, rand: &mut Box<dyn RngCore>) -> Box<dyn Layer> {
        return match self {
            LayerTypes::DENSE(rows, activation, learning) => Box::new(Dense::new(rows.clone(), prev_cols, activation.clone(), learning.clone(), rand)),
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
        }
    }
}
