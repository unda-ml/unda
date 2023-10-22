use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation<'a>{
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64
}

pub const SIGMOID: Activation = Activation {
    function: &|x| {
        let res = 1.0 / (1.0 + E.powf(-x));
        return res;
    },
    derivative: &|x| x * (1.0 - x)
};

pub const TANH: Activation = Activation {
    function: &|x| {
        let res = f64::tanh(x);
        return res;
    },
    derivative: &|x| 1.0 - f64::tanh(x).powf(2.0)
};

pub const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative: &|x| {
        if x.max(0.0) == x {
            return 1.0;
        }
        return 0.0;
    }
};
