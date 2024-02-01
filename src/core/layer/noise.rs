use std::{ops::Range, f32::consts::PI};

pub fn gen_noise(n: f32, x: usize) -> Range<f32> {
    //let max = f32::abs(f32::log(-n * f32::atan(x as f32) + (n * PI)/2.0, 0.9));
    let max = 1.0 / (n * x as f32);
    -max..max
}
