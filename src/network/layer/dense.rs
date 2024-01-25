use std::ops::Range;

use crate::network::{matrix::Matrix, activations::Activations, input::Input};

use super::{layers::Layer, distributions::Distributions, pair::GradientPair};
use rand::RngCore;
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
    pub fn new_ser(rows: usize, cols: usize, flat_weight: Vec<f32>, flat_bias: Vec<f32>) -> Dense {
        let weight_shape: Matrix = Matrix::from_sized(flat_weight, rows, cols);
        let bias_shape: Matrix = Matrix::from_sized(flat_bias, rows, 1);

        Dense {
            weights: weight_shape,
            biases: bias_shape,
            data: Matrix::new_empty(0, 0),
            loss: 1.0,
            activation_fn: Activations::SIGMOID,
            learning_rate: 0.01,
            beta1: 0.99,
            beta2: 0.99,
            epsilon: 1e-16,
            time: 1,
            m_weights: Matrix::new_empty(0, 0),
            v_weights: Matrix::new_empty(0, 0),
            m_biases: Matrix::new_empty(0, 0),
            v_biases: Matrix::new_empty(0, 0)
        }
    }
    pub fn new(layers: usize, layer_cols_before: usize, activation: Activations, learning_rate: f32, rng: &mut Box<dyn RngCore>) -> Dense{
        let distribution: Distributions = match activation{
            Activations::ELU(_) | Activations::RELU | Activations::LEAKYRELU | Activations::SOFTMAX => Distributions::He(layers),
            Activations::TANH | Activations::SIGMOID => Distributions::Xavier(layers, layer_cols_before),
        };
        let mut res = Dense { 
            loss: 1.0,
            weights: Matrix::new_random(layer_cols_before, layers, rng, &distribution),
            biases: Matrix::new_empty(layer_cols_before, 1),

            m_weights: Matrix::new_empty(layer_cols_before, layers),
            v_weights: Matrix::new_empty(layer_cols_before, layers),

            m_biases: Matrix::new_empty(layer_cols_before, 1),
            v_biases: Matrix::new_empty(layer_cols_before, 1),

            data: Matrix::new_empty(0, 0),
            activation_fn: activation,
            learning_rate,
            beta1: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
            time: 1
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
    fn update_gradients(&mut self, gradient_pair: (&Box<dyn Input>, &Box<dyn Input>), clip: Option<Range<f32>>) {//, noise: &f32) {
        let bias_gradient = Matrix::from(gradient_pair.0.to_param_2d()); //+ noise;
        let weight_gradient = Matrix::from(gradient_pair.1.to_param_2d()); //+ noise;


        self.time += 1;

        self.m_weights = self.m_weights.clone() * self.beta1 + &(weight_gradient.clone() * (1.0 - self.beta1));
        self.v_weights = self.v_weights.clone() * self.beta2 + &((weight_gradient^2) * (1.0 - self.beta2));

        self.m_biases = self.m_biases.clone() * self.beta1 + &(bias_gradient.clone() * (1.0 - self.beta1));
        self.v_biases = self.v_biases.clone() * self.beta2 + &((bias_gradient.clone()^2) * (1.0 - self.beta2));

        let m_weights_hat = self.m_weights.clone() / (1.0 - self.beta1.powi(self.time as i32));
        let v_weights_hat = self.v_weights.clone() / (1.0 - self.beta2.powi(self.time as i32));

        let m_bias_hat = self.m_biases.clone() / (1.0 - self.beta1.powi(self.time as i32));
        let v_bias_hat = self.v_biases.clone() / (1.0 - self.beta2.powi(self.time as i32));

        let mut weights_update = m_weights_hat.clone() / &(v_weights_hat.sqrt() + self.epsilon);
        let mut bias_update = m_bias_hat.clone() / &(v_bias_hat.sqrt() + self.epsilon);

        if let Some(clip_range) = clip{
            bias_update.clip(&clip_range);
            weights_update.clip(&clip_range);
        }
        self.biases = self.biases.clone() + &bias_update;
        self.weights = self.weights.clone() + &weights_update;
    }
    fn avg_gradient(&self, gradients: Vec<&Box<dyn Input>>) -> Box<dyn Input>{
        let len = gradients.len();
        let gradients_mat = gradients.into_iter()
            .map(|gradient| Matrix::from(gradient.to_param_2d()));
        let sum: Matrix = gradients_mat.sum();
        let avg = sum / len;
        Box::new(avg) 
    }
    fn get_gradients(&self, data: &Box<dyn Input>, data_at: &Box<dyn Input>, errors: &Box<dyn Input>) -> GradientPair {
        let gradient = self.activation_fn.apply_fn(Matrix::from(data.to_param_2d()));
        let errors_mat = Matrix::from(errors.to_param_2d());
        let mut gradients_mat = Matrix::from(gradient.to_param_2d());
        let mut data_mat = Matrix::from(data_at.to_param_2d());

        gradients_mat = gradients_mat.dot_multiply(&errors_mat) * self.learning_rate; 

        if gradients_mat.columns != data_mat.rows {
            data_mat = data_mat.transpose();
        }

        let weight_gradient = gradients_mat.clone() * &(data_mat.clone());

        GradientPair(Box::new(gradients_mat), Box::new(weight_gradient))
    }
    fn get_data(&self) -> Box<dyn Input>{
        Box::new(self.data.clone())
    }
    fn set_data(&mut self, data: &Box<dyn Input>) {
        self.data = Matrix::from(data.to_param_2d())
    }
    ///Moves the DNN forward through the weights and biases of this current layer
    ///Maps an activation function and then returns the resultant Matrix
    fn forward(&self, inputs: &Box<dyn Input>) -> Box<dyn Input> {
        let new_data = self.activation_fn.apply_fn(self.weights.clone() * &Matrix::from(inputs.to_param().to_param_2d()).transpose() + &self.biases);

        Box::new(new_data)
    }
    fn update_errors(&self, errors: Box<dyn Input>) -> Box<dyn Input> {
        let errors_mat = Matrix::from(errors.to_param_2d());
        Box::new(self.weights.clone().transpose() * &errors_mat)
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
        Box::new(self.activation_fn.apply_fn(self.data.clone()))
    }

    fn get_activation(&self) -> Option<Activations> {
        Some(self.activation_fn.clone())
    }
    fn shape(&self) -> (usize, usize, usize){
        (self.weights.columns, 1, 1)
    }
    fn get_loss(&self) -> f32{
        self.loss
    }
    fn get_weights(&self) -> Box<dyn Input> {
        self.weights.to_box()
    }
    fn get_biases(&self) -> Box<dyn Input>{
        self.biases.to_box()
    }
}
