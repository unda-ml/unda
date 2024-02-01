use std::ops::Range;

use rand::RngCore;
use serde::{Serialize, Deserialize};
use crate::core::data::{matrix::Matrix, input::Input, matrix3d::Matrix3D};
use super::{layers::Layer, methods::distributions::Distributions, methods::activations::Activations};
use crate::core::layer::methods::pair::GradientPair;

#[derive(Serialize, Deserialize, Clone)]
pub struct Convolutional{
    filter_weights: Matrix3D,
    filter_biases: Vec<f32>,
    data: Matrix3D,
    stride: usize,
    filters: usize,
    shape: (usize, usize),
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    loss: f32,
    
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    time: usize,

    m_weights: Matrix3D,
    v_weights: Matrix3D,
    m_biases: Vec<f32>,
    v_biases: Vec<f32>,

    activation_fn: Activations,
    learning_rate: f32
}

impl Convolutional{
    pub fn new(filters: usize, kernel_size: (usize, usize), input_shape: (usize, usize, usize), stride: usize, activation_fn: Activations, learning_rate: f32, rng: &mut Box<dyn RngCore>, input_size: usize) -> Convolutional {
        let distribution = match activation_fn {
            Activations::ELU(_) | Activations::TANH | Activations::SIGMOID | Activations::SOFTMAX => Distributions::Xavier(input_size, kernel_size.0 * kernel_size.1),
            Activations::RELU | Activations::LEAKYRELU => Distributions::He(input_size)
        };
        let mut res = Convolutional{
            filter_weights: Matrix3D::new_random(kernel_size.0, kernel_size.1, filters, rng, &distribution),
            m_weights: Matrix3D::new_empty(kernel_size.0, kernel_size.1, filters),
            v_weights: Matrix3D::new_empty(kernel_size.0, kernel_size.1, filters),

            filter_biases: vec![0.0; filters],
            m_biases: vec![0.0; filters],
            v_biases: vec![0.0; filters],

            data: Matrix3D::new_empty(0,0,0),
            stride,
            filters,
            input_shape,
            shape: kernel_size,
            activation_fn,
            learning_rate,
            output_shape: (0,0,0),
            loss: 1.0,

            beta1: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
            time: 1
        };

        let res_len = Convolutional::get_res_size(input_shape.0, kernel_size.0, 0, stride);
        let res_width = Convolutional::get_res_size(input_shape.1, kernel_size.1, 0, stride);
        
        res.data = Matrix3D::new_empty(res_len, res_width, filters);
        res.output_shape = (res_len, res_width, 1);

        (res.beta1, res.beta2) = res.get_betas();
        res.epsilon = res.get_epsilon();

        res
    }
    fn get_betas(&self) -> (f32, f32) {
        (0.9, 0.999)
    }
    fn get_epsilon(&self) -> f32{
        1e-10
    }
    pub fn convolute(&self, idx: usize, input: Matrix) -> Matrix {
        let kernel = self.filter_weights.get_slice(idx);
        let mut output = Matrix::new_empty(self.output_shape.0, self.output_shape.1);

        let mut x: usize;
        let mut y: usize = 0;

        for output_x in 0..output.columns {
            x = 0;
            for output_y in 0..output.rows {
                let sum = input.get_sub_matrix(x, y, kernel.rows, kernel.columns).dot_multiply(&kernel).sum();
                output.data[output_y][output_x] = sum;

                x += self.stride;
            }
            y += 1;
        }

        //println!("{}", output);
        //self.data.set_slice(idx, output);
        output
    }
    fn get_res_size(w: usize, k: usize, p: usize, s:usize) -> usize {
        (w - k + 2*p) / s + 1
    }
}

#[typetag::serde]
impl Layer for Convolutional {

    fn update_gradients(&mut self, _gradient_pair: (&Box<dyn Input>, &Box<dyn Input>), _clip: Option<Range<f32>>) {//, noise: &f32) {
        panic!("unfinished");
    }
    fn avg_gradient(&self, _gradients: Vec<&Box<dyn Input>>) -> Box<dyn Input>{
        panic!("unfinished");        
    }
    fn get_gradients(&self, _data: &Box<dyn Input>, _data_at: &Box<dyn Input>, _errors: &Box<dyn Input>) -> GradientPair { 
        panic!("unfinished");
    }
    fn update_errors(&self, _errors: Box<dyn Input>) -> Box<dyn Input>{
        panic!("unfinished");
    }
    fn forward(&self,inputs: &Box<dyn Input>) -> Box<dyn Input> {
        let input_mat = Matrix3D::from(inputs.to_param_3d());
        for i in 0..input_mat.layers {
            for j in 0..self.filters {
                self.convolute(j, input_mat.get_slice(i));
            }
        }
        let data = self.data.clone() + &self.filter_biases;
        Box::new(data.clone())
    }
    fn set_data(&mut self, data: &Box<dyn Input>) {
        self.data = Matrix3D::from(data.to_param_3d())
    }
    fn backward(&mut self,gradients:Box<dyn Input> ,errors:Box<dyn Input> ,data:Box<dyn Input>) -> Box<dyn Input> {
        let mut gradients_mat = Matrix3D::from(gradients.to_param_3d());
        let mut errors_mat = Matrix3D::from(errors.to_param_3d());
        let _data_mat = Matrix3D::from(data.to_param_3d());

        gradients_mat = gradients_mat.dot_multiply(&errors_mat) * self.learning_rate;
        errors_mat = self.filter_weights.clone().transpose() * &errors_mat;

        self.loss = 0.0;
        errors_mat.to_param().iter().for_each(|error| {
            self.loss += error.powi(2);
        });
        self.loss = self.loss / errors_mat.to_param().len() as f32;

        panic!();
    }
    fn get_data(&self) -> Box<dyn Input> {
        self.data.to_box()
    }
    fn shape(&self) -> (usize,usize,usize) {
        self.input_shape
    }
    fn get_loss(&self) -> f32 {
        self.loss
    }
    fn update_gradient(&self) -> Box<dyn Input> {
        //TODO
        self.data.to_box()
    }
    fn get_weights(&self) -> Box<dyn Input> {
        self.filter_weights.to_box()
    }
    fn get_biases(&self) -> Box<dyn Input>{
        self.filter_biases.to_box()
    }
}
