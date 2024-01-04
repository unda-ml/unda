use std::ops::Range;

use rand::{Rng, prelude::*};

use rand_distr::Normal;

pub enum Distributions{
    Xavier(usize, usize),
    He(usize)
}

impl Distributions {
    pub fn sample(&self, rng: &mut Box<dyn RngCore>) -> f32 {
        let mut res = 0.0;
        while res == 0.0 {
            res = match self {
                Distributions::Xavier(inputs, layer_len) => rng.gen_range(get_xavier_range(*inputs, *layer_len)),
                Distributions::He(inputs) => { 
                    rng.sample(get_he_range(*inputs))
                }
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
