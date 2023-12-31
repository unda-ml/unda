use super::{matrix::Matrix, matrix3d::Matrix3D};

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
    fn shape(&self) -> (usize, usize, usize){
        (0,0,0)
    }
    fn to_box(&self) -> Box<dyn Input>;
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
    fn shape(&self) -> (usize, usize, usize) {
        (self.len(), 1, 0)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param())
    }
}

impl Input for Vec<Vec<f32>> {
    fn to_param(&self) -> Vec<f32> {
        self.clone().into_iter().flatten().collect::<Vec<f32>>()
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        self.clone()
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        vec![self.clone()]
    }
    fn shape(&self) -> (usize, usize, usize) {
        (self.len(), self[0].len(), 0)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param_2d())
    }
}

impl Input for Vec<Vec<Vec<f32>>> {
    fn to_param(&self) -> Vec<f32> {
        self.clone().into_iter().flatten().flatten().collect::<Vec<f32>>()
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        self.clone().into_iter().flatten().collect::<Vec<Vec<f32>>>()
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        self.clone()
    }
    fn shape(&self) -> (usize, usize, usize) {
        (self[0][0].len(), self[0].len(), self.len())
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param_3d())
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
    fn shape(&self) -> (usize, usize, usize) {
        (self.rows, self.columns, 0)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param_2d())
    }
}

impl Input for Matrix3D {
    fn to_param(&self) -> Vec<f32> {
        self.data.clone().into_iter().flatten().flatten().collect::<Vec<f32>>()
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        self.data.clone().into_iter().flatten().collect::<Vec<Vec<f32>>>()
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        self.data.clone()
    }
    fn shape(&self) -> (usize, usize, usize) {
        self.shape()
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param_3d())
    }
}

impl Into<Box<dyn Input>> for Vec<f32> {
    fn into(self) -> Box<dyn Input> {
        Box::new(self)
    }
}
