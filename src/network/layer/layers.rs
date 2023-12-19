use crate::network::{input::Input, matrix::Matrix, activations::Activations};

use super::dense::Dense;

pub trait Layer{
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        Matrix::new_random(0,0) 
    }
    fn backward(&mut self, inputs: &Matrix, gradients: &Matrix, errors: &Matrix, layer_prev: &Matrix, layer_prev_bias: &Matrix) -> (Matrix, Matrix, Matrix, Matrix); 
    fn get_cols(&self) -> usize {0}
    fn get_rows(&self) -> usize {0}
    fn get_weights(&self) -> Matrix;
    fn set_weights(&mut self, new_weights: Matrix);
    fn get_bias(&self) -> Matrix;
    fn set_bias(&mut self, new_bias: Matrix);
    fn get_activation(&self) -> Option<Activations> {
        None
    }
}

pub enum LayerTypes{
    //DENSE: Nodes, Activation Function, Learning Rate
    DENSE(usize, Activations, f32),
    
}

impl LayerTypes{
    pub fn to_layer(&self, prev_cols: usize) -> Box<dyn Layer> {
        return match self {
            LayerTypes::DENSE(rows, activation, learning) => Box::new(Dense::new(rows.clone(), prev_cols, activation.clone(), learning.clone())),
        };
    }
    pub fn get_size(&self) -> usize{
        return match self{
            LayerTypes::DENSE(rows, _, _) => *rows,
            _ => 0
        }
    }
}
