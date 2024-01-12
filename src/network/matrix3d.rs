use rand::{thread_rng, Rng, RngCore};
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use serde::{Serialize, Deserialize};

use std::ops;

use super::{matrix::Matrix, layer::distributions::Distributions};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix3D {
    pub rows: usize,
    pub columns: usize,
    pub layers: usize,
    pub data: Vec<Vec<Vec<f32>>>
}

impl ops::Mul<&Matrix3D> for Matrix3D {
    type Output = Matrix3D;
    fn mul(self, rhs: &Matrix3D) -> Self::Output {
        if self.layers != rhs.layers || self.columns != rhs.rows{
            panic!("Invalid 3D Matrix Dot Multiplication, mismatched dimensions:\nSelf:{}x{}x{}\nOther:{}x{}x{}",
                   self.rows,
                   self.columns,
                   self.layers,
                   rhs.rows,
                   rhs.columns,
                   rhs.layers)
        }
        let mut res = Matrix3D::new_empty(self.rows, rhs.columns, self.layers);

        for idx in 0..self.layers{
            for i in 0..self.rows {
                for j in 0..rhs.columns {
                    let mut sum: f32 = 0.0;
                    for k in 0..self.columns {
                        sum += self.data[idx][i][k] * rhs.data[idx][k][j];
                    }
                    res.data[idx][i][j] = sum;
                }
            }
        }
        res
    }
}

impl ops::Add<&Matrix3D> for Matrix3D {
    type Output = Matrix3D;
    fn add(self, rhs: &Matrix3D) -> Self::Output {
        self.compare(rhs);
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers {
            for j in 0..self.columns {
                for k in 0..self.rows {
                    res.data[i][j][k] = self.data[i][j][k] + rhs.data[i][j][k];
                }
            }
        }
        res
    }
}

///Does a per-point subtraction to a 3D Matrix
impl ops::Sub<&Matrix3D> for Matrix3D {
    type Output = Matrix3D;
    fn sub(self, rhs: &Matrix3D) -> Self::Output {
        self.compare(rhs);
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers {
            for j in 0..self.columns {
                for k in 0..self.rows {
                    res.data[i][j][k] = self.data[i][j][k] - rhs.data[i][j][k];
                }
            }
        }
        res
    }
}
///Does a scalar multiplication to all points of the 3D Matrix
impl ops::Mul<f32> for Matrix3D {
    type Output = Matrix3D;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers {
            for j in 0..self.columns {
                for k in 0..self.rows {
                    res.data[i][j][k] = self.data[i][j][k] * rhs;
                }
            }
        }
        res

    }
}

///Does a scalar power to a 3D Matrix
impl ops::BitXor<i32> for Matrix3D{
    type Output = Matrix3D;
    fn bitxor(self, rhs: i32) -> Self::Output {
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers {
            for j in 0..self.columns {
                for k in 0..self.rows {
                    res.data[i][j][k] = self.data[i][j][k].powi(rhs);
                }
            }
        }
        res
    }
}
///Performs scalar division for a 3D Matrix
impl ops::Div<f32> for Matrix3D {
    type Output = Matrix3D;
    fn div(self, rhs: f32) -> Self::Output {
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers {
            for j in 0..self.columns {
                for k in 0..self.rows {
                    res.data[i][j][k] = self.data[i][j][k] / rhs;
                }
            }
        }
        res
    }
}

///Does a scalar addition to a 3D Matrix
impl ops::Add<f32> for Matrix3D{
    type Output = Matrix3D;
    fn add(self, rhs: f32) -> Self::Output {
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers {
            for j in 0..self.columns {
                for k in 0..self.rows {
                    res.data[i][j][k] = self.data[i][j][k] + rhs;
                }
            }
        }
        res
    }
}

///Performs a per-point division based on two 3D Matrices with the same sizes
impl ops::Div<&Matrix3D> for Matrix3D {
    type Output = Matrix3D;
    fn div(self, rhs: &Matrix3D) -> Self::Output {
        self.compare(rhs);
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        for i in 0..self.layers{
            for j in 0..self.columns{
                for k in 0..self.rows{
                    res.data[i][j][k] = self.data[i][j][k] / rhs.data[i][j][k];
                }
            }
        }
        res
    }
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
    fn compare(&self, other: &Matrix3D) {
        if self.rows != other.rows || self.columns != other.columns || self.layers != other.layers {
            panic!("Invalid 3D Matrix Dot Multiplication, mismatched dimensions:\nSelf:{}x{}x{}\nOther:{}x{}x{}",
                   self.rows,
                   self.columns,
                   self.layers,
                   other.rows,
                   other.columns,
                   other.layers)
        }
    }
    pub fn dot_multiply(&mut self, other: &Matrix3D) -> Matrix3D{
        self.compare(other);        
        let mut res = Matrix3D::new_empty(self.rows, self.columns, self.layers);
        
        for i in 0..self.layers {
            for j in 0..self.rows {
                for k in 0..self.columns {
                    res.data[i][j][k] = self.data[i][j][k] * other.data[i][j][k];
                }
            }
        }

        res
    }
    pub fn transpose(&mut self) -> Matrix3D {
        let mut res = Matrix3D::new_empty(self.columns, self.rows, self.layers);
        for i in 0..self.layers{
            for j in 0..self.rows{
                for k in 0..self.columns{
                    res.data[i][k][j] = self.data[i][j][k];
                }
            }
        }
        res
    }
    pub fn map(&mut self, function: &dyn Fn(f32) -> f32) -> Matrix3D {
        Matrix3D::from(self.data.clone()
                       .into_iter()
                       .map(|layer| layer
                            .into_iter()
                            .map(|row| row
                                 .into_iter()
                                 .map(|value| function(value))
                                 .collect())
                            .collect())
                       .collect())
    }
    pub fn new_empty(rows: usize, cols: usize, layers: usize) -> Matrix3D {
        Matrix3D { rows, columns: cols, layers, data: vec![vec![vec![0.0; cols]; rows]; layers] }
    }
    pub fn new_random(rows: usize, cols: usize, layers: usize, seed: &Option<String>, distribution: &Distributions) -> Matrix3D {
        let mut res = Matrix3D::new_empty(rows, cols, layers);
        let mut rng: Box<dyn RngCore> = match seed {
            Some(seed_rng) => Box::new(Seeder::from(seed_rng).make_rng::<Pcg64>()),
            None => Box::new(thread_rng())
        };
        
        for z in 0..layers {
            for y in 0..rows {
                for x in 0..cols {
                    res.data[z][y][x] = distribution.sample(&mut rng);
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
