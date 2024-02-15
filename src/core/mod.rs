pub mod data;
pub mod graph;
pub mod layer;
pub mod network;
pub mod serialize;

#[cfg(test)]
mod test {
    use crate::core::{network::Network, layer::layers::InputTypes};

    #[test]
    fn check_set_input() {
        let mut net: Network = Network::new(10);
        net.set_input(InputTypes::DENSE(10));
        net.set_input(InputTypes::DENSE(20));
        assert_eq!(net.layer_sizes[0], 20);
    }
}
