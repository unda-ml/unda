use rand::{thread_rng, Rng};
use std::ops;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    pub fn dot_multiply(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != self.columns{
            panic!("Invalid matrix multiplaction, mismatched dimensions");
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



