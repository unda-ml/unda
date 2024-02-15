pub mod activations;
pub mod distributions;
pub mod noise;
pub mod pair;
pub mod errors;

#[cfg(test)]
mod test {

    use crate::core::data::{matrix::Matrix, input::Input};

    use super::activations::{SIGMOID, RELU, Activations};

    #[test]
    fn test_sigmoid() {
        let sigmoid_fn = SIGMOID.function;
        let res = sigmoid_fn(0.8);
        assert_eq!(res, 0.689974481128);
        let res = sigmoid_fn(1.0);
        assert_eq!(res, 0.73105857863);
        let res = sigmoid_fn(0.0);
        assert_eq!(res, 0.5);
    }
    #[test]
    fn test_sigmoid_der() {
        let sigmoid_der = SIGMOID.derivative;
        let res = sigmoid_der(0.8);
        assert_eq!(res, 0.21390969652);
        let res = sigmoid_der(1.0);
        assert_eq!(res, 0.196611933241);
        let res = sigmoid_der(0.0);
        assert_eq!(res, 0.25);
    }


    #[test]
    fn test_relu() {
        let relu_fn = RELU.function;
        let res = relu_fn(-100.0);
        assert_eq!(res, 0.0);
        let res = relu_fn(100.0);
        assert_eq!(res, 100.0);
        let res = relu_fn(0.6);
        assert_eq!(res, 0.6);
    }
    #[test]
    fn test_relu_der() {
        let relu_der = RELU.derivative;
        let res = relu_der(-1.0);
        assert_eq!(res, 0.0);
        let res = relu_der(5.0);
        assert_eq!(res, 1.0);
    }
    #[test]
    fn test_softmax(){
        let softmax = Activations::SOFTMAX;
        let input_mat = Matrix::from_sized(vec![0.7,0.8], 2, 1);
        let res = softmax.apply_fn(input_mat).to_param();

        assert_eq!(res, vec![0.4750208, 0.5249792]);
    }
}
