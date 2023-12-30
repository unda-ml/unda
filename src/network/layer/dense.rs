use crate::network::{matrix::Matrix, activations::{Activation, Activations}, input::Input};

use super::layers::Layer;
use serde::{Deserialize, Serialize};

///A Dense Neural Network Layer of a model, containing just nodes, weights, biases and an
///activation function
///Implements the Layer trait
#[derive(Serialize, Deserialize)]
pub struct Dense{
    pub weights: Matrix,   
    pub biases: Matrix,
    pub data: Matrix,
    loss: f32,

    pub activation_fn: Activations,
    learning_rate: f32,

    beta1: f32,
    beta2: f32,
    epsilon: f32,
    time: usize,

    m_weights: Matrix,
    v_weights: Matrix,
    m_biases: Matrix,
    v_biases: Matrix
}

impl Dense{
    pub fn new(layers: usize, layer_cols_before: usize, activation: Activations, learning_rate: f32) -> Dense{
        let mut res = Dense { 
            loss: 1.0,
            weights: Matrix::new_random(layer_cols_before, layers),
            biases: Matrix::new_random(layer_cols_before, 1),

            m_weights: Matrix::new_empty(layer_cols_before, layers),
            v_weights: Matrix::new_empty(layer_cols_before, layers),

            m_biases: Matrix::new_empty(layer_cols_before, 1),
            v_biases: Matrix::new_empty(layer_cols_before, 1),

            data: Matrix::new_random(0, 0),
            activation_fn: activation,
            learning_rate,
            beta1: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
            time: 0
        };
        (res.beta1, res.beta2) = res.get_betas();
        res.epsilon = res.get_epsilon();

        res
    }
    fn get_betas(&self) -> (f32, f32){
        (0.9, 0.999)
    }
    fn get_epsilon(&self) -> f32{
        1e-10
    }
}

#[typetag::serde]
impl Layer for Dense{
    fn get_data(&self) -> Box<dyn Input>{
        Box::new(self.data.clone())
    }
    ///Moves the DNN forward through the weights and biases of this current layer
    ///Maps an activation function and then returns the resultant Matrix
    fn forward(&mut self, inputs: &Box<dyn Input>) -> Box<dyn Input> {
        self.data = (self.weights.clone() * &Matrix::from(inputs.to_param_2d()).transpose() + &self.biases)
            .map(self.activation_fn.get_function().function);

        Box::new(self.data.clone().transpose())
    }
    ///Does Back Propegation according to simple Dense network rules
    ///Finds the error of the previous layer and returns what the updated weights and biases should
    ///be in that layer, updates the gradients and errors to move backwards once
    fn backward(&mut self, gradients: Box<dyn Input>, errors: Box<dyn Input>, data: Box<dyn Input>) -> Box<dyn Input> {
        let mut gradients_mat = Matrix::from(gradients.to_param_2d());
        let mut errors_mat = Matrix::from(errors.to_param_2d());
        let data_mat = Matrix::from(data.to_param_2d());

        gradients_mat = gradients_mat.dot_multiply(&errors_mat) * self.learning_rate;
        errors_mat = self.weights.clone().transpose() * &errors_mat;

        self.loss = 0.0;
        errors_mat.to_param().iter().for_each(|error| {
            self.loss += error.powi(2);
        });

        self.loss = self.loss / errors_mat.to_param().len() as f32;

        self.time += 1;

        let weight_gradient = gradients_mat.clone() * &(data_mat.clone().transpose());

        self.m_weights = self.m_weights.clone() * self.beta1 + &(weight_gradient.clone() * (1.0 - self.beta1));
        self.v_weights = self.v_weights.clone() * self.beta2 + &((weight_gradient^2) * (1.0 - self.beta2));

        self.m_biases = self.m_biases.clone() * self.beta1 + &(gradients_mat.clone() * (1.0 - self.beta1));
        self.v_biases = self.v_biases.clone() * self.beta2 + &((gradients_mat.clone()^2) * (1.0 - self.beta2));

        let m_weights_hat = self.m_weights.clone() / (1.0 - self.beta1.powi(self.time as i32));
        let v_weights_hat = self.v_weights.clone() / (1.0 - self.beta2.powi(self.time as i32));

        let m_bias_hat = self.m_biases.clone() / (1.0 - self.beta1.powi(self.time as i32));
        let v_bias_hat = self.v_biases.clone() / (1.0 - self.beta2.powi(self.time as i32));

        let weights_update = m_weights_hat.clone() / &(v_weights_hat.sqrt() + self.epsilon);
        let bias_update = m_bias_hat.clone() / &(v_bias_hat.sqrt() + self.epsilon);

        self.biases = self.biases.clone() + &bias_update;
        self.weights = self.weights.clone() + &weights_update;

        Box::new(errors_mat)
    }

    fn update_gradient(&self) -> Box<dyn Input> {
        Box::new(self.data.clone().map(self.activation_fn.get_function().derivative))
    }

    /*fn backward(&mut self, inputs: &Matrix, gradients: &Matrix, errors: &Matrix, layer_prev: &Matrix, layer_prev_bias: &Matrix) -> (Matrix, Matrix, Matrix, Matrix){
        let mut gradients_mat = gradients.clone().dot_multiply(&errors).map(&|x| x * self.learning_rate);
        let new_layer_prev = layer_prev.clone() + &(gradients_mat.clone() * &self.data.clone().transpose());
        let new_biases = layer_prev_bias.clone() + &gradients_mat.clone();
        
        let errors_mat = layer_prev.clone().transpose() * errors;

        //set error of layer, should have something to do with possibly the MSE of errors_mat,
        //which we could call .to_param() on and iterate through like we do in the network accuracy
        //fn
        self.loss = 0.0;
        errors_mat.to_param().iter().for_each(|error| {
            self.loss += error.powi(2);
        });

        self.loss = self.loss / errors_mat.to_param().len() as f32;

        gradients_mat = self.data.map(self.activation_fn.get_function().derivative);
        (new_biases.clone(), new_layer_prev.clone(), gradients_mat, errors_mat)
    }*/
    fn get_activation(&self) -> Option<Activations> {
        Some(self.activation_fn.clone())
    }
    fn shape(&self) -> (usize, usize, usize){
        (self.weights.rows, self.weights.columns, 0)
    }
    fn get_loss(&self) -> f32{
        self.loss
    }
}
