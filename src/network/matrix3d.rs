use rand::{thread_rng, Rng};
use serde::{Serialize, Deserialize};

use std::ops;

use super::matrix::Matrix;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix3D {
    pub rows: usize,
    pub columns: usize,
    pub layers: usize,
    pub data: Vec<Vec<Vec<f32>>>
}

impl ops::Add<&Vec<f32>> for Matrix3D {
    type Output = Matrix3D;
    fn add(self, rhs: &Vec<f32>) -> Self::Output {
        if rhs.len() != self.layers {
            panic!("Vec of scalar values not same size as layers");
        }
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for k in 0..self.layers{
            for j in 0..self.rows{
                for i in 0..self.columns{
                    res.data[k][j][i] = self.data[k][j][i] + rhs[k];
                }
            }
        }
        res
    }
}

impl Matrix3D{
    pub fn new_empty(rows: usize, cols: usize, layers: usize) -> Matrix3D {
        Matrix3D { rows, columns: cols, layers, data: vec![vec![vec![0.0; cols]; rows]; layers] }
    }
    pub fn new_random(rows: usize, cols: usize, layers: usize) -> Matrix3D {
        let mut res = Matrix3D::new_empty(rows, cols, layers);
        let mut rng = thread_rng();

        for z in 0..layers {
            for y in 0..rows {
                for x in 0..cols {
                    res.data[z][y][x] = rng.gen::<f32>() * 2.0 - 1.0;
                }
            }
        }
        res
    }
    pub fn get_slice(&self, idx: usize) -> Matrix {
        if idx >= self.layers {
            panic!("Layer does not exist")
        }

        Matrix::from(self.data[idx].clone())
    }
    pub fn from(data: Vec<Vec<Vec<f32>>>) -> Matrix3D {
        Matrix3D { rows: data[0].len(), columns: data[0][0].len(), layers: data.len(), data }
    }
    pub fn set_slice(&mut self, idx: usize, slice: Matrix){
        if slice.rows != self.rows && slice.columns != self.columns {
            panic!("Slice you are trying to set is not size of the slice you are trying to input");
        }
        if idx >= self.layers {
            panic!("layer index is outside of layers present");
        }
        self.data[idx] = slice.data;
    }
}
