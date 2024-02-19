use super::{matrix::Matrix, matrix3d::Matrix3D};
///Dynamic trait for input objects as they go in and travel through a neural network
///As long as your data type implements Input, it can be used through convolutional, dense,
///and any other layer type without the need for flattening layers or shape specifiers.
///
///Dense layers always use to_param for single input and to_param_2d to generate matrices through
///the network
///Convolutional layers always use to_param_2d for single inputs and to_param_3d for through-layer
///traveling
pub trait Input: Send + Sync{
    ///Flattens input data into a one dimensional context
    ///Most commonly used by dense layers and output layers
    fn to_param(&self) -> Vec<f32>;
    ///Flattens(or extends) data into a two dimensional context
    ///Flattens if data is a higher order, extends if its a lower order
    fn to_param_2d(&self) -> Vec<Vec<f32>>;
    ///Flattens(or extends) data into a two dimensional context
    ///Flattens if data is a higher order, extends if its a lower order
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>>;
    ///Returns the underlying shape the data has when it is not being morphed or shaped by the
    ///param methods
    fn shape(&self) -> (usize, usize, usize);
    ///Wrapper method for boxing an input up
    fn to_box(&self) -> Box<dyn Input>;
}

///Input implementation for 1 dimensional floating point vectors
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
        (self.len(), 1, 1)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param())
    }
}
///Input implementation for 2 dimensional floating point vectors(matrix data)
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
        (self.len(), self[0].len(), 1)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param_2d())
    }
}

///Input implementation for 3 dimensional floating point vectors(3d matrix data or can be viewed as
///convolutional outputs)
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
        (self.rows, self.columns, 1)
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
        (self.rows, self.columns, self.layers)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param_3d())
    }
}

impl From<Vec<f32>> for Box<dyn Input> {
    fn from(val: Vec<f32>) -> Self {
        Box::new(val)
    }
}
