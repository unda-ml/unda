pub mod dense;
pub mod conv;
pub mod layers;
pub mod methods;


#[cfg(test)]
mod test{
    use crate::core::{layer::layers::Layer, data::input::Input};

    use super::dense::Dense;

    #[test]
    fn dense_shape() {
        let new_dense = Dense::new_ser(10, 2, vec![0.0; 20], vec![0.0; 10]);
        assert_eq!(new_dense.shape(), (2, 1, 1));
    }
    #[test]
    fn forward_dense_shape(){
        let new_dense = Dense::new_ser(10, 2, vec![0.0; 20], vec![0.0; 10]);
        let input_test: Box<dyn Input> = Box::new(vec![1.0,1.0]);
        let input_res = new_dense.forward(&input_test);
        assert_eq!(input_res.shape(), (10, 1, 1));
    }
    #[test]
    #[should_panic]
    fn forward_dense_panic(){
        let new_dense = Dense::new_ser(5, 4, vec![0.0; 20], vec![0.0; 5]);
        let test: Box<dyn Input> = Box::new(vec![0.0; 5]);
        new_dense.forward(&test);
    }
    #[test]
    fn forward_dense_result(){
        let new_dense = Dense::new_ser(2,3, vec![1.0,1.0,2.0,5.0,0.75,6.1], vec![0.0; 2]);
        let test: Box<dyn Input> = Box::new(vec![0.5, 6.0, 3.0]);

        let res = new_dense.forward(&test);
        assert_eq!(res.to_param_2d(), vec![vec![0.9999963], vec![1.0]]);
    }

}
