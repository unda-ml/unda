use std::f32::consts::E;
use serde::{Deserialize, Serialize};

use super::{matrix::Matrix, input::Input};

#[derive(Clone)]
pub struct Activation<'a>{
    pub function: &'a dyn Fn(f32) -> f32,
    pub derivative: &'a dyn Fn(f32) -> f32
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Activations{
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    SOFTMAX
}
impl Activations{
    fn get_function(&self) -> Option<Activation>{
        return match self{
            Activations::SIGMOID => Some(SIGMOID),
            Activations::TANH => Some(TANH),
            Activations::RELU => Some(RELU),
            Activations::LEAKYRELU => Some(LEAKY_RELU),
            _ => None
        };
    }
    pub fn apply_fn(&self, mut data: Matrix) -> Matrix {
        match self{
            Activations::SIGMOID | Activations::TANH | Activations::RELU | Activations::LEAKYRELU => {
                return data.map(self.get_function().unwrap().function);
            },
            Activations::SOFTMAX => { 
                let exp_logits: Vec<f32> = data.to_param()
                    .iter()
                    .map(|&x| x.exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                return Matrix::from_sized(exp_logits.iter().map(|x| x / sum_exp).collect::<Vec<f32>>(), data.rows, data.columns)
            }
        };
    }
    pub fn apply_derivative(&self, mut data: Matrix) -> Matrix {
        match self{
            Activations::SIGMOID | Activations::TANH | Activations::RELU | Activations::LEAKYRELU => {
                return data.map(self.get_function().unwrap().derivative);
            },
            Activations::SOFTMAX => { 
                let softmax_output = data.to_param()
                    .iter()
                    .zip(data.to_param().iter().map(|&x| 1.0 - x))
                    .map(|(s,ds)| s * ds)
                    .collect();
                return Matrix::from_sized(softmax_output, data.rows, data.columns);
            }
        };
    }
}


const SIGMOID: Activation = Activation {
    function: &|x| {
        let res = 1.0 / (1.0 + E.powf(-x));
        return res;
    },
    derivative: &|x| x * (1.0 - x)
};

const TANH: Activation = Activation {
    function: &|x| {
        let res = f32::tanh(x);
        return res;
    },
    derivative: &|x| 1.0 - f32::tanh(x).powf(2.0)
};

const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative: &|x| {
        if x.max(0.0) == x {
            return 1.0;
        }
        return 0.0;
    }
};

const LEAKY_RELU: Activation = Activation{
    function: &|x| {
        if x.max(0.0) == x{
            return x;
        }
        return 0.001 * x;
    },
    derivative: &|x| {
        if x.max(0.0) == x{
            return 1.0
        }
        return 0.001;
    }
};
