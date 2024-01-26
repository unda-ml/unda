use rand::prelude::*;
use std::ops::{AddAssign, SubAssign};
use std::{ops::{self, Range}, iter};
use serde::{Serialize, Deserialize};

use super::layer::distributions::Distributions;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Matrix{
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Vec<f32>>
}

impl std::fmt::Display for Matrix{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut resp: String = String::from("");
        for i in 0..self.rows{
            resp += "[";
            for j in 0..self.columns{
                if self.data[i][j] < 0.0 {
                    resp += &format!(" {:<03.3} ", self.data[i][j]);
                } else {
                    resp += &format!("  {:<03.3} ", self.data[i][j]);
                }
            }
            resp += "]\n";
        }
        write!(f, "{}", resp)
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        if self.rows != rhs.rows || self.columns != rhs.columns {
            panic!("Error attempting to add two matrices with different dimensions \nMatrix A: {} x {}\nMatrix B: {} x {}", 
                   self.rows, 
                   self.columns, 
                   rhs.rows,
                   rhs.columns);
        }
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        if self.rows != rhs.rows || self.columns != rhs.columns {
            panic!("Error attempting to add two matrices with different dimensions \nMatrix A: {} x {}\nMatrix B: {} x {}", 
                   self.rows, 
                   self.columns, 
                   rhs.rows,
                   rhs.columns);
        }
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
    }
}

impl ops::Add<&Matrix> for Matrix{
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns{
            panic!("Error attempting to add two matrices with different dimensions \nMatrix A: {} x {}\nMatrix B: {} x {}", self.rows, self.columns, other.rows, other.columns);
        }

        let mut res = Matrix::new_empty(self.rows, self.columns);
        for i in 0..res.rows{
            for j in 0..res.columns{
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        res
    }
}
impl ops::Sub<&Matrix> for Matrix{
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix{
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Error attempting to subtract two matrices with different dimensions \nMatrix A: {} x {}\nMatrix B: {} x {}", self.rows, self.columns, other.rows, other.columns);
        }
        let mut res = Matrix::new_empty(self.rows, self.columns);

        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        res
    }
}

impl ops::Mul<&Matrix> for Matrix{
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix{
        if self.columns != other.rows{
            panic!("Error attempting to multiply two matrices with different dimensions \nMatrix A: {} x {}\nMatrix B: {} x {}", self.rows, self.columns, other.rows, other.columns);
        }

        let mut res = Matrix::new_empty(self.rows, other.columns);                 

        for i in 0..self.rows{
            for j in 0..other.columns{
                let mut sum = 0.0;
                for k in 0..self.columns{
                    sum += self.data[i][k] * other.data[k][j];                    
                }
                res.data[i][j] = sum;
            }
        }
        res
    }
}

impl ops::Mul<f32> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] * rhs;
            }
        }
        res
    }
}

impl ops::BitXor<i32> for Matrix{
    type Output = Matrix;
    fn bitxor(self, rhs: i32) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j].powi(rhs);
            }
        }
        res
    }
}

impl ops::Div<f32> for Matrix {
    type Output = Matrix;
    fn div(self, rhs: f32) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] / rhs;
            }
        }
        res
    }
}

impl ops::Div<usize> for Matrix {
    type Output = Matrix;
    fn div(self, rhs: usize) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] / rhs as f32;
            }
        }
        res
    }
}

impl iter::Sum for Matrix {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter_peek = iter.peekable();
        let first_peek = iter_peek.peek().unwrap(); 
        let rows = first_peek.rows;
        let cols = first_peek.columns;

        iter_peek.fold(Matrix::new_empty(rows, cols), |curr, next|{
            curr + &next
        })
    }

}

impl ops::Add<f32> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: f32) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] + rhs;
            }
        }
        res
    }
}

impl ops::Add<&f32> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &f32) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] + rhs;
            }
        }
        res
    }
}

impl ops::Div<&Matrix> for Matrix{
    type Output = Matrix;
    fn div(self, rhs: &Matrix) -> Self::Output {
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] / rhs.data[i][j];
            }
        }
        res
    }
}

impl Matrix{
    pub fn clip(&mut self, clip_range: &Range<f32>){
        for i in 0..self.rows {
            for j in 0..self.columns {
                if self.data[i][j] < clip_range.start {
                    self.data[i][j] = clip_range.start;
                } else if self.data[i][j] > clip_range.end {
                    self.data[i][j] = clip_range.end;
                }
            }
        }
    }
    pub fn from_sized(data: Vec<f32>, rows: usize, cols: usize) -> Matrix {
        if rows * cols != data.len() {
            panic!("Size incompatible between data inputted and desired matrix size");
        }
        let mut res = Matrix::new_empty(rows, cols);
        let mut idx = 0;
        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = data[idx];
                idx += 1;
            }
        }
        res
    }
    pub fn sum(&self) -> f32 {
        let mut res: f32 = 0.0;
        for i in 0..self.rows{
            for j in 0..self.columns{
                res += self.data[i][j];
            }
        }
        res
    }
    pub fn get_sub_matrix(&self, x: usize, y: usize, rows: usize, cols: usize) -> Matrix {
        if x + cols > self.columns || y + rows > self.rows {
            panic!("Sub matrix cannot fit within matrix");
        }
        let mut res: Matrix = Matrix::new_empty(rows, cols);
        for i in 0..rows {
            for j in 0..cols{
                res.data[i][j] = self.data[i+y][j+x];
            }
        }
        res
    }
    pub fn sqrt(&self) -> Matrix{
        let mut res: Matrix = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j].sqrt();
            }
        }
        res
    }
    pub fn new_empty(rows: usize, cols: usize) -> Matrix{
        Matrix{
            rows: rows,
            columns: cols,
            data: vec![vec![0.0; cols]; rows]
        }
    }

    pub fn new_random(rows: usize, cols: usize, rng: &mut Box<dyn RngCore>, distribution: &Distributions) -> Matrix{
        let mut res = Matrix::new_empty(rows, cols); 
        for row in 0..rows{
            for col in 0..cols{
                res.data[row][col] = distribution.sample(rng);
            }
        }
        res
    }

    pub fn sample_noise(&self, noise: &Range<f32>, rng: &mut Box<dyn RngCore>) -> Matrix {
        let noise_dist: Distributions = Distributions::Ranged(noise.clone());

        let res = self.clone() + noise_dist.sample(rng);
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
    pub fn dot_multiply(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != self.columns{
            panic!("Invalid matrix dot multiplaction, mismatched dimensions:\n{}x{}\n{}x{}", 
                   self.rows, 
                   self.columns,
                   other.rows,
                   other.columns);
        }
        let mut res = Matrix::new_empty(self.rows, self.columns);

        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        res
    }
    /*pub fn subtract(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != self.columns{
            panic!("Invalid matrix subtraction, mismatched dimensions");
        }
        let mut res = Matrix::new_empty(self.rows, self.columns);
        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        res
    } */
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
        let mut res = Matrix::new_empty(self.columns, self.rows);

        for i in 0..self.rows{
            for j in 0..self.columns{
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn matrix_add() {
        let mat_a = Matrix::from_sized(vec![1.0,0.0,0.5,0.25], 2, 2);
        let mat_b = Matrix::from_sized(vec![0.0,0.0,0.5,0.5], 2, 2);

        assert_eq!(Matrix::from_sized(vec![1.0,0.0,1.0,0.75], 2, 2), mat_a + &mat_b);
    }
    #[test]
    #[should_panic]
    fn matrix_panic() {
        let mat_a = Matrix::from_sized(vec![1.0,0.0,0.5,0.25], 4, 1);
        let mat_b = Matrix::from_sized(vec![0.0,0.0,0.5,0.5], 2, 2);

        mat_a + &mat_b;
    }
    #[test]
    fn test_transpose() {
        let mat_a = Matrix::from_sized(vec![1.0,1.0,1.0,4.0], 4, 1);
        let mut mat_b = Matrix::from_sized(vec![1.0,1.0,1.0,4.0], 1, 4);

        assert_eq!(mat_a, mat_b.transpose());
    }
    #[test]
    fn test_transpose_ne(){
        let mat_a = Matrix::from_sized(vec![1.0,1.0,1.0,4.0], 4, 1);
        let mut mat_b = Matrix::from_sized(vec![4.0,1.0,1.0,1.0], 1, 4);

        assert_ne!(mat_a, mat_b.transpose());
    }
    #[test]
    fn test_pow() {
        let mat_a = Matrix::from_sized(vec![2.0,3.0,1.0,2.0,3.0,1.0], 2, 3);
        let res_test = Matrix::from_sized(vec![4.0,9.0,1.0,4.0,9.0,1.0], 2, 3);
        assert_eq!(res_test, mat_a ^ 2);
    }
    #[test]
    fn test_addassign(){
        let mut mat_a = Matrix::from_sized(vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0], 2, 3);
        mat_a += Matrix::from_sized(       vec![1.0, 0.0,-1.0, 5.0, 3.0,-1.0], 2, 3);

        assert_eq!(mat_a, Matrix::from_sized(vec![3.0, 3.0, 0.0, 7.0, 6.0, 0.0], 2, 3));
    }
    #[test]
    #[should_panic]
    fn test_invalid_mul() {
        let mat_a = Matrix::from_sized(vec![1.0,0.0,0.0,1.0], 2, 2);
        let mat_b = Matrix::from_sized(vec![1.0,0.0,0.0,1.0], 4, 1);

        mat_a * &mat_b;
    }
    #[test]
    fn test_mul(){
        let mat_a = Matrix::new_empty(2, 5);
        let mat_b = Matrix::new_empty(5, 10);

        mat_a * &mat_b;
    }
    #[test]
    fn test_mul_vals(){
        let mat_a = Matrix::from_sized(vec![2.5, 1.0, 3.0, 4.55, 8.9, -1.0], 2, 3);
        let mat_b = Matrix::from_sized(vec![0.66, 77.1, 10.5, 2.0,
                                             3.0, 9.75, 11.1, 18.0,
                                            15.0, 8.0, 1.9, 0.5], 3, 4);

        let res_mat = Matrix::from_sized(vec![49.65, 226.5, 43.05, 24.5, 14.702999, 429.58, 144.66501, 168.8], 2, 4);
        let res = mat_a * &mat_b;
        assert_eq!(res_mat, res);
    }
    #[test]
    fn test_mul_shape() {
        let mat_a = Matrix::new_empty(2, 5);
        let mat_b = Matrix::new_empty(5, 10);
        let res = mat_a * &mat_b;

        assert_eq!(res, Matrix::new_empty(2, 10));
    }
}

