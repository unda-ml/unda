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
        let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();
        while res.is_zero() {
            res = match self {
                Distributions::Xavier(layer_size_prev, layer_size_curr) => {
                    rng.sample(normal) * 10.0 * get_xavier_range(*layer_size_prev, *layer_size_curr)
                },
                Distributions::He(layer_size_before) => { 
                    rng.sample(normal) * get_he_range(*layer_size_before)
                }
                Distributions::Default => rng.gen_range(-10.0..10.0),
                Distributions::Ranged(range) => rng.gen_range(range.clone())
            };
        }
        res
    }
}

pub fn get_xavier_range(inputs: usize, layer_len: usize) -> f32 {
    f32::sqrt(2.0 / (inputs + layer_len) as f32)
}

pub fn get_he_range(inputs: usize) -> f32 {
    f32::sqrt(2.0 / inputs as f32)
}
