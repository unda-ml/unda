use std::f32::consts::E;
use serde::{Deserialize, Serialize};

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
    LEAKYRELU
}
impl Activations{
    pub fn get_function(&self) -> Activation{
        return match self{
            Activations::SIGMOID => SIGMOID,
            Activations::TANH => TANH,
            Activations::RELU => RELU,
            Activations::LEAKYRELU => LEAKY_RELU
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
