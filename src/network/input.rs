use super::{matrix::Matrix};

pub trait Input{
    fn to_param(&self) -> Vec<f32>{
        vec![]
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>>{
        vec![]
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>>{
        vec![]
    }
}

impl Input for Vec<f32>{
    fn to_param(&self) -> Vec<f32>{
        self.clone()
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        vec![self.clone()]
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        vec![vec![self.clone()]]
    }
}

impl Input for Matrix {
    fn to_param(&self) -> Vec<f32> {
        self.data.clone().into_iter().flatten().collect::<Vec<f32>>()
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        self.data.clone()
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        vec![self.data.clone()]
    }
}
