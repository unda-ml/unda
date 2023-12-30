use serde::{Serialize, Deserialize};

use crate::network::{matrix::Matrix, activations::Activations, input::Input};

use super::layers::Layer;



#[derive(Serialize, Deserialize)]
pub struct Convolutional{
    filter_weights: Vec<Matrix>,
    filter_biases: Vec<f32>,
    data: Vec<Matrix>,
    stride: usize,
    filters: usize,
    shape: (usize, usize),

    activation_fn: Activations,
    learning_rate: f32
}

impl Convolutional{
    pub fn new(filters: usize, kernel_size: (usize, usize), stride: usize, activation_fn: Activations, learning_rate: f32) -> Convolutional {
        let mut res = Convolutional{
            filter_weights: vec![Matrix::new_random(kernel_size.0, kernel_size.1); filters],
            filter_biases: vec![0.0; filters],
            data: vec![Matrix::new_empty(kernel_size.0, kernel_size.1); filters],
            stride,
            filters,
            shape: kernel_size,
            activation_fn,
            learning_rate
        };

        res
    }
}

/*#[typetag::serde]
impl Layer for Convolutional{
    fn forward(&mut self,inputs: &Box<dyn Input>) -> Box<dyn Input> {

    }
    fn backward(&mut self,parsed:Box<dyn Input> ,errors:Box<dyn Input> ,data:Box<dyn Input>) -> Box<dyn Input> {

    }
    fn get_activation(&self) -> Option<Activations> {
        Some(self.activation_fn)
    }
    fn shape(&self) -> (usize,usize,usize) {
        (self.shape.0, self.shape.1, 0)
    }
    fn get_cols(&self) -> usize {
        self.shape.1
    }
    fn get_rows(&self) -> usize {
        self.shape.0
    }
    fn get_data(&self) -> Box<dyn crate::network::input::Input> {
        Box::new(self.data[0])
    }
    fn get_bias(&self) -> Matrix {
        Matrix::from(self.filter_biases.to_param_2d())
    }
    fn get_weights(&self) -> Matrix {
        self.filter_weights[0].clone()
    }
    fn set_bias(&mut self,new_bias:Matrix) {
        
    }
    fn set_weights(&mut self,new_weights:Matrix) {
        
    }
    fn get_loss(&self) -> f32 {
        0.0
    }
    fn update_gradient(&self) -> Box<dyn Input> {
        Box::new(self.data[0])
    }
}*/
