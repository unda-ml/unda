pub struct Matrix{
    pub cols: usize,
    pub rows: usize,
    pub data: Vec<f32>
}

impl ops::Add<&Matrix> for &Matrix{
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        if self.columns != other.columns || self.rows != other.rows{
            panic!("Matrix addition is not in proper format :(");
        }
        for i in 0..self.data{
            
        }
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
    }
}


impl Matrix{

}
