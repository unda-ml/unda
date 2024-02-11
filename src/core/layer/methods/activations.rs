use std::f32::{consts::E, NAN};
use rand_distr::num_traits::{Zero, Signed};
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
                return res;
            },
            Activations::ELU(alpha) => {
                let data_elu = data.to_param()
                    .iter()
                    .map(|&x| elu(*alpha, x)).collect();
                return Matrix::from_sized(data_elu, data.rows, data.columns);
            }

        };
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
                let res = Matrix::from_sized(vec![1.0; data.rows * data.columns], data.rows, data.columns);
                //println!("{}", res);
                return res;
            },
            Activations::ELU(alpha) => {
                let data_elu = data.to_param()
                    .iter()
                    .map(|&x| d_elu(*alpha, x)).collect();
                return Matrix::from_sized(data_elu, data.rows, data.columns);
            }
        };
    }
}


const SIGMOID: Activation = Activation {
    function: &|x| {
        let res = 1.0 / (1.0 + E.powf(-x));
        return res;
    },
    derivative: &|x| (1.0 / (1.0 + E.powf(-x))) * (1.0 - (1.0 / (1.0 + E.powf(-x))))
};

const TANH: Activation = Activation {
    function: &|x| {
        let res = f32::tanh(x);
        return res;
    },
    derivative: &|x| 1.0 - f32::tanh(x).powf(2.0)
};

const RELU: Activation = Activation {
    function: &|x| {
        let res = x.max(0.0);
        return res;
    },
    derivative: &|x| {
        !x.is_negative() as i32 as f32
    }
};

const LEAKY_RELU: Activation = Activation{
    function: &|x| {
        if x > 0.0{
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

fn d_elu(alpha: f32, x: f32) -> f32 {
     if x > 0.0 {
        return 1.0;
    }
    return alpha * E.powf(x);
   
}

fn elu(alpha: f32, x: f32) -> f32 {
    if x.max(0.0) == x{
        return x;
    }
    return alpha * (E.powf(x) - 1.0);
}

#[cfg(test)]
mod test {

    use crate::core::data::{matrix::Matrix, input::Input};

    use super::{SIGMOID, RELU, Activations};

    #[test]
    fn test_sigmoid() {
        let sigmoid_fn = SIGMOID.function;
        let res = sigmoid_fn(0.8);
        assert_eq!(res, 0.689974481128);
        let res = sigmoid_fn(1.0);
        assert_eq!(res, 0.73105857863);
        let res = sigmoid_fn(0.0);
        assert_eq!(res, 0.5);
    }
    #[test]
    fn test_sigmoid_der() {
        let sigmoid_der = SIGMOID.derivative;
        let res = sigmoid_der(0.8);
        assert_eq!(res, 0.21390969652);
        let res = sigmoid_der(1.0);
        assert_eq!(res, 0.196611933241);
        let res = sigmoid_der(0.0);
        assert_eq!(res, 0.25);
    }


    #[test]
    fn test_relu() {
        let relu_fn = RELU.function;
        let res = relu_fn(-100.0);
        assert_eq!(res, 0.0);
        let res = relu_fn(100.0);
        assert_eq!(res, 100.0);
        let res = relu_fn(0.6);
        assert_eq!(res, 0.6);
    }
    #[test]
    fn test_relu_der() {
        let relu_der = RELU.derivative;
        let res = relu_der(-1.0);
        assert_eq!(res, 0.0);
        let res = relu_der(5.0);
        assert_eq!(res, 1.0);
    }
    #[test]
    fn test_softmax(){
        let softmax = Activations::SOFTMAX;
        let input_mat = Matrix::from_sized(vec![0.7,0.8], 2, 1);
        let res = softmax.apply_fn(input_mat).to_param();

        assert_eq!(res, vec![0.4750208, 0.5249792]);
    }
}
