use rand::{thread_rng, Rng};
use ndarray::*;
use serde::{Serialize, Deserialize};
use std::ops;
use rayon::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix{
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Vec<f32>>
}

impl ops::Add<&Matrix> for Matrix{
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        /*if self.rows != other.rows || self.columns != other.columns{
            panic!("Error attempting to add two matrices with different dimensions");
        }

        let mut res = Matrix::new_empty(self.rows, self.columns);
        for i in 0..res.rows{
            for j in 0..res.columns{
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        res*/
        self.apply_elementwise(other, |a,b| a + b)
    }
}
impl ops::Sub<&Matrix> for Matrix{
    type Output = Matrix;
    
    fn sub(self, other: &Matrix) -> Matrix{
        /*if self.rows != other.rows || self.columns != other.columns {
            panic!("Error attemtping to subtract two matrices with different dimesnsions");
        }
        let mut res = Matrix::new_empty(self.rows, self.columns);
    
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        res*/

        self.apply_elementwise(other, |a,b| a - b)
    }
}

impl ops::Mul<&Matrix> for Matrix{
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix{
        if self.columns != other.rows{
            panic!("Matrix multiplication is in invalid format");
        }

        let mut res = Matrix::new_empty(self.rows, other.columns);
        res.data.iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(j, cell)| {
                *cell = self.data[i]
                    .iter()
                    .zip(other.data.iter().map(|r| &r[j]))
                    .map(|(a, b)| a * b)
                    .sum();
            });
        });
        res
    }
}

impl Matrix{

    fn new_empty(rows: usize, cols: usize) -> Matrix{
        Matrix{
            rows: rows,
            columns: cols,
            data: vec![vec![0.0; cols]; rows]
        }
    }

    pub fn new_random(rows: usize, cols: usize) -> Matrix{
        let mut rng = thread_rng();
        let mut res = Matrix::new_empty(rows, cols); 
        for row in 0..rows{
            for col in 0..cols{
                res.data[row][col] = rng.gen::<f32>() * 2.0 - 1.0;
            }
        }
        res
    }

    /*pub fn add(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Invalid matrix addition");
        }
        let mut res = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        res
    }*/

    fn as_ndarray(&self) -> Array2<f32> {
        Array2::from_shape_fn((self.rows, self.columns), |(i, j)| self.data[i][j])
    }

    fn apply_elementwise(&self, other: &Matrix, op: impl Fn(f32, f32) -> f32) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Mismatched matrix dimensions");
        }

        let self_arr = self.as_ndarray();
        let other_arr = other.as_ndarray();
        let result_arr = Array2::from_shape_fn((self.rows, self.columns), |(i, j)| {
            op(self_arr[[i, j]], other_arr[[i, j]])
        });

        Matrix::from_ndarray(&result_arr.view())
    }

    fn from_ndarray(arr: &ArrayView2<f32>) -> Matrix {
        let rows = arr.shape()[0];
        let columns = arr.shape()[1];
        let data: Vec<Vec<f32>> = (0..rows)
            .map(|i| (0..columns).map(|j| arr[[i, j]]).collect())
            .collect();
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn dot_multiply(&mut self, other: &Matrix) -> Matrix {
        self.apply_elementwise(other, |a, b| a * b)
    }
    
    pub fn from(data: Vec<Vec<f32>>) -> Matrix {
        Matrix{
            rows: data.len(),
            columns: data[0].len(),
            data
        }
    }
    pub fn map(&mut self, function: &dyn Fn(f32) -> f32) -> Matrix{
        Matrix::from((self.data).clone()
                     .into_iter()
                     .map(|row| row
                          .into_iter()
                          .map(|value| function(value))
                          .collect())
                     .collect())
    }
    pub fn transpose(&mut self) -> Matrix {
        let self_arr = self.as_ndarray();
        let result_arr = self_arr.t();

        Matrix::from_ndarray(&result_arr.view())
    }
}
