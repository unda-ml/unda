use std::ops::Range;

use rand::{Rng, prelude::*};

use rand_distr::{Normal, num_traits::Zero};

pub enum Distributions{
    Xavier(usize, usize),
    He(usize),
    Ranged(Range<f32>),
    Default
}

impl Distributions {
    pub fn sample(&self, rng: &mut Box<dyn RngCore>) -> f32 {
        let mut res = 0.0;
        while res.is_zero() {
            res = match self {
                Distributions::Xavier(n_inputs, n_outputs) => rng.gen_range(get_xavier_range(*n_inputs, *n_outputs)),
                Distributions::He(inputs) => { 
                    rng.sample(get_he_range(*inputs))
                }
                Distributions::Default => rng.gen_range(-0.05..0.05),
                Distributions::Ranged(range) => rng.gen_range(range.clone())
            };
        }
        res
    }
}

pub fn get_xavier_range(inputs: usize, layer_len: usize) -> Range<f32> {
    -(f32::sqrt(6.0) / ((inputs + layer_len) as f32).sqrt())..(f32::sqrt(6.0) / ((inputs + layer_len) as f32).sqrt())
}

pub fn get_he_range(inputs: usize) -> Normal<f32> {
    Normal::new(0.0, f32::sqrt(2.0 / inputs as f32)).unwrap()
}
