use triton_grow::network::{network::Network, activations, modes::Mode};

fn main() {
    let mut inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    let mut outputs = vec![vec![0.0],vec![1.0],vec![1.0],vec![0.0]];
    let mut new_net: Network = Network::new(vec![2,3,1], activations::SIGMOID, 0.1);
    
    new_net = new_net.train_to_loss(inputs, outputs, 0.001, 100000, Mode::Avg, 0.001, 3, 10);
    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0])[0].round());
    println!("0 and 1: {:?}", new_net.feed_forward(&vec![0.0,1.0])[0].round());
    println!("1 and 1: {:?}", new_net.feed_forward(&vec![1.0,1.0])[0].round());
    println!("0 and 0: {:?}", new_net.feed_forward(&vec![0.0,0.0])[0].round());
    println!("Net network made: {:?}", new_net.layers);

}
