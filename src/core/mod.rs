pub mod data;
pub mod graph;
pub mod layer;
pub mod network;
pub mod serialize;

#[cfg(test)]
mod test {
    use crate::core::{layer::layers::InputTypes, network::Sequential};

    #[test]
    fn check_set_input() {
        let mut net = Sequential::new(10);
        net.set_input(InputTypes::DENSE(10));
        net.set_input(InputTypes::DENSE(20));
        assert_eq!(net.layer_sizes[0], 20);
    }
}
