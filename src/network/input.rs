pub trait Input{
    fn to_param(&self) -> Vec<f32>{
        vec![]
    }
}

impl Input for Vec<f32>{
    fn to_param(&self) -> Vec<f32>{
        self.clone()
    }
}
