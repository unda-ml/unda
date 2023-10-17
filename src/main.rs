use triton::network::{network::Network, activations};

fn main() {
    let mut new_net: Network = Network::new(vec![1,2,3,4,2], activations::SIGMOID, 0.1);
    let mut newer_net = Network::from(&new_net, 2, 4);

    println!("{:?}", newer_net.layers);
}
