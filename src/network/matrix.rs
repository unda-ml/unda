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

impl ops::Add<&Matrix> for &Matrix{
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
       self.apply_elementwise(other, |a,b| a + b)
    }
}
impl ops::Sub<&Matrix> for &Matrix{
    type Output = Matrix;
    
    fn sub(self, other: &Matrix) -> Matrix{
       self.apply_elementwise(other, |a,b| a - b)
    }
}

impl ops::Mul<&Matrix> for &Matrix{
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix{
        if self.columns != other.rows{
            panic!("Matrix multiplication is in invalid format");
        }
        //Do parallel matrix multiplication as to the Strassen algorithm
        self.par_multiply(other)
    }
}

impl Matrix{
    ///Parralelized matrix multiplication with the Strassen algorithm.
    ///Splits matrix into 4 chunks that are then subdivided even further recursively until each
    ///beginning matrix is less than 64 in length or width
    fn par_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.columns, other.rows);

        if self.rows <= 64 || self.columns <= 64 || other.columns <= 64 {
            // If the matrices are small enough, use the standard algorithm
            return self.standard_multiply(other);
        }

        // Split matrices into quadrants
        let a11 = self.submatrix(0, 0, self.rows / 2, self.columns / 2);
        let a12 = self.submatrix(0, self.columns / 2, self.rows / 2, self.columns / 2);
        let a21 = self.submatrix(self.rows / 2, 0, self.rows / 2, self.columns / 2);
        let a22 = self.submatrix(self.rows / 2, self.columns / 2, self.rows / 2, self.columns / 2);

        let b11 = other.submatrix(0, 0, other.rows / 2, other.columns / 2);
        let b12 = other.submatrix(0, other.columns / 2, other.rows / 2, other.columns / 2);
        let b21 = other.submatrix(other.rows / 2, 0, other.rows / 2, other.columns / 2);
        let b22 = other.submatrix(other.rows / 2, other.columns / 2, other.rows / 2, other.columns / 2);

        // Calculate 7 products recursively in parallel
        let p1 = rayon::join(|| a11.multiply(&(&b12 - &b22)), || a11.multiply(&(&b12 - &b22)));
        let p2 = rayon::join(|| (&a11 + &a12).multiply(&b22), || (&a11 + &a12).multiply(&b22));
        let p3 = rayon::join(|| (&a21 + &a22).multiply(&b11), || (&a21 + &a22).multiply(&b11));
        let p4 = rayon::join(|| a22.multiply(&(&b21 - &b11)), || a22.multiply(&(&b21 - &b11)));
        let p5 = rayon::join(
            || (&a11 + &a22).multiply(&(&b11 + &b22)),
            || (&a11 + &a22).multiply(&(&b11 + &b22)),
        );
        let p6 = rayon::join(
            || (&a12 - &a22).multiply(&(&b21 + &b22)),
            || (&a12 - &a22).multiply(&(&b21 + &b22)),
        );
        let p7 = rayon::join(
            || (&a11 - &a21).multiply(&(&b11 + &b12)),
            || (&a11 - &a21).multiply(&(&b11 + &b12)),
        );

        // Calculate result matrices
        let c11 = rayon::join(
            || &(&(&p5.0 + &p4.0) - &p2.0) + &p6.0,
            || &(&(&p5.1 + &p4.1) - &p2.1) + &p6.1,
        );
        let c12 = rayon::join(|| &p1.0 + &p2.0, || &p1.1 + &p2.1);
        let c21 = rayon::join(|| &p3.0 + &p4.0, || &p3.1 + &p4.1);
        let c22 = rayon::join(
            || &(&(&p5.0 + &p1.0) - &p3.0) - &p7.0,
            || &(&(&p5.1.clone() + &p1.1) - &p3.1) - &p7.1,
        );

        // Combine the quadrants into the result matrix
        let mut result = Matrix::new_empty(self.rows, other.columns);
        result.set_submatrix(0, 0, &c11.0);
        result.set_submatrix(0, other.columns / 2, &c12.0);
        result.set_submatrix(self.rows / 2, 0, &c21.0);
        result.set_submatrix(self.rows / 2, other.columns / 2, &c22.0);

        result
    }
    ///Unparrallel Strassen implementation for matrix multipication
    fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.columns, other.rows);

        if self.rows <= 64 || self.columns <= 64 || other.columns <= 64 {
            // If the matrices are small enough, use the standard algorithm
            return self.standard_multiply(other);
        }

        // Split matrices into quadrants
        let a11 = self.submatrix(0, 0, self.rows / 2, self.columns / 2);
        let a12 = self.submatrix(0, self.columns / 2, self.rows / 2, self.columns / 2);
        let a21 = self.submatrix(self.rows / 2, 0, self.rows / 2, self.columns / 2);
        let a22 = self.submatrix(self.rows / 2, self.columns / 2, self.rows / 2, self.columns / 2);

        let b11 = other.submatrix(0, 0, other.rows / 2, other.columns / 2);
        let b12 = other.submatrix(0, other.columns / 2, other.rows / 2, other.columns / 2);
        let b21 = other.submatrix(other.rows / 2, 0, other.rows / 2, other.columns / 2);
        let b22 = other.submatrix(other.rows / 2, other.columns / 2, other.rows / 2, other.columns / 2);

        // Calculate 7 products recursively
        let p1 = a11.multiply(&(&b12 - &b22));
        let p2 = (&a11 + &a12).multiply(&b22);
        let p3 = (&a21 + &a22).multiply(&b11);
        let p4 = a22.multiply(&(&b21 - &b11));
        let p5 = (&a11 + &a22).multiply(&(&b11 + &b22));
        let p6 = (&a12 - &a22).multiply(&(&b21 + &b22));
        let p7 = (&a11 - &a21).multiply(&(&b11 + &b12));

        // Calculate result matrices
        let c11 = &(&(&p5 + &p4) - &p2) + &p6;
        let c12 = &p1 + &p2;
        let c21 = &p3 + &p4;
        let c22 = &(&(&p5 + &p1) - &p3) - &p7;

        // Combine the quadrants into the result matrix
        let mut result = Matrix::new_empty(self.rows, other.columns);
        result.set_submatrix(0, 0, &c11);
        result.set_submatrix(0, other.columns / 2, &c12);
        result.set_submatrix(self.rows / 2, 0, &c21);
        result.set_submatrix(self.rows / 2, other.columns / 2, &c22);

        result
    }

    fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Matrix {
        let mut submatrix = Matrix::new_empty(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                submatrix.data[i][j] = self.data[row_start + i][col_start + j];
            }
        }
        submatrix
    }

    fn set_submatrix(&mut self, row_start: usize, col_start: usize, other: &Matrix) {
        for i in 0..other.rows {
            for j in 0..other.columns {
                self.data[row_start + i][col_start + j] = other.data[i][j];
            }
        }
    }

    fn standard_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.columns, other.rows);

        let mut result = Matrix::new_empty(self.rows, other.columns);

        for i in 0..self.rows {
            for j in 0..other.columns {
                for k in 0..self.columns {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }

        result
    }

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
