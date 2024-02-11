use crate::core::data::{input::Input, matrix::Matrix};

pub enum ErrorTypes{
    MeanAbsolute,
    MeanSquared,
    CategoricalCrossEntropy
}

impl ErrorTypes{
    pub fn get_error(&self, actual: &Box<dyn Input>, expected: &Box<dyn Input>, batch_size: usize) -> Box<dyn Input> {
        return match self {
            ErrorTypes::MeanAbsolute => {
                let actual_matrix = Matrix::from(actual.to_param_2d());
                let expected_matrix = Matrix::from(expected.to_param_2d());

                let res = (actual_matrix - &expected_matrix).transpose();

                Box::new(res)
            },
            ErrorTypes::MeanSquared => {
                let actual_matrix = Matrix::from(actual.to_param_2d());
                let expected_matrix = Matrix::from(expected.to_param_2d());

                let n = batch_size;

                let res = ((actual_matrix - &expected_matrix) ^ 2).transpose() / n;

                Box::new(res)
            },
            ErrorTypes::CategoricalCrossEntropy => {
                let expected_matrix = Matrix::from(expected.to_param_2d());
                let actual_matrix = Matrix::from(actual.to_param_2d());

                let mut res = actual_matrix - &expected_matrix;

                //Small little friend is the MVP of CategoricalCrossEntropy, he prevents doing a
                //log(0)
                /*let small_little_friend: f32 = 1e-9;
                actual_matrix += small_little_friend;
                actual_matrix.log();
                let mut res = expected_matrix.dot_multiply(&actual_matrix);
                println!("{}", -1.0 * res.sum());
                panic!();*/
                //println!("\n{:?}", actual.to_param());
                //println!("{:?}", expected.to_param());
                //println!("{:?}", res.to_param());
                let contains_nan = res.to_param().iter().map(|x| x.is_nan()).collect::<Vec<bool>>().contains(&true);
                if contains_nan {
                    panic!()
                }

                Box::new(res.transpose())
            }
        }
    }
}
