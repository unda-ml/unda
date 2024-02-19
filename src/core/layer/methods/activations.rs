use std::f32::consts::E;
use rand_distr::num_traits::Signed;
use serde::{Deserialize, Serialize};

use crate::core::data::{matrix::Matrix, input::Input};

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
    SOFTMAX,
    ELU(f32)
}
impl Activations{
    fn get_function(&self) -> Option<Activation>{
        match self{
            Activations::SIGMOID => Some(SIGMOID),
            Activations::TANH => Some(TANH),
            Activations::RELU => Some(RELU),
            Activations::LEAKYRELU => Some(LEAKY_RELU),
            _ => None
        }
    }
    pub fn apply_fn(&self, mut data: Matrix) -> Matrix {
        match self{
            Activations::SIGMOID | Activations::TANH | Activations::RELU | Activations::LEAKYRELU => {
                return data.map(self.get_function().unwrap().function);
            },
            Activations::SOFTMAX => { 
                let max = data.to_param().iter().fold(f32::NAN, |a, &b| a.max(b));
                let exp_logits: Vec<f32> = data.to_param()
                    .iter()
                    .map(|&x| {
                        (x - max).exp()
                    }).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                //println!("\n\n{:?}", data.to_param());
                let res = Matrix::from_sized(exp_logits.iter().map(|x| x / sum_exp).collect::<Vec<f32>>(), data.rows, data.columns);
                //println!("{}", res);
                res
            },
            Activations::ELU(alpha) => {
                let data_elu = data.to_param()
                    .iter()
                    .map(|&x| elu(*alpha, x)).collect();
                Matrix::from_sized(data_elu, data.rows, data.columns)
            }

        }
    }
    pub fn apply_derivative(&self, mut data: Matrix) -> Matrix {
        match self{
            Activations::SIGMOID | Activations::TANH | Activations::RELU | Activations::LEAKYRELU => {
                return data.map(self.get_function().unwrap().derivative);
            },
            Activations::SOFTMAX => { 
                //let softmax_output = data.to_param()
                    //.iter()
                    //.zip(data.to_param().iter().map(|&x| 1.0 - x))
                    //.map(|(s,ds)| s * ds)
                    //.collect();
                //println!("{:?}", data.to_param());
                
                //println!("{}", res);
                Matrix::from_sized(vec![1.0; data.rows * data.columns], data.rows, data.columns)
            },
            Activations::ELU(alpha) => {
                let data_elu = data.to_param()
                    .iter()
                    .map(|&x| d_elu(*alpha, x)).collect();
                Matrix::from_sized(data_elu, data.rows, data.columns)
            }
        }
    }
}


pub(super) const SIGMOID: Activation = Activation {
    function: &|x| {
        
        1.0 / (1.0 + E.powf(-x))
    },
    derivative: &|x| (1.0 / (1.0 + E.powf(-x))) * (1.0 - (1.0 / (1.0 + E.powf(-x))))
};

pub(super) const TANH: Activation = Activation {
    function: &|x| {
        
        f32::tanh(x)
    },
    derivative: &|x| 1.0 - f32::tanh(x).powf(2.0)
};

pub(super) const RELU: Activation = Activation {
    function: &|x| {
        
        x.max(0.0)
    },
    derivative: &|x| {
        !x.is_negative() as i32 as f32
    }
};

pub(super) const LEAKY_RELU: Activation = Activation{
    function: &|x| {
        if x > 0.0{
            return x;
        }
        0.001 * x
    },
    derivative: &|x| {
        if x.max(0.0) == x{
            return 1.0
        }
        0.001
    }
};

fn d_elu(alpha: f32, x: f32) -> f32 {
     if x > 0.0 {
        return 1.0;
    }
    alpha * E.powf(x)
   
}

fn elu(alpha: f32, x: f32) -> f32 {
    if x.max(0.0) == x{
        return x;
    }
    alpha * (E.powf(x) - 1.0)
}


