use triton_grow::network::{network::Network, activations::Activations, modes::Mode};

fn main() {
    let inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    let outputs = vec![vec![0.0],vec![1.0],vec![1.0],vec![0.0]];
    let mut new_net: Network = Network::new(vec![2,3,1], Activations::SIGMOID, 0.1);
    //let mut new_net: Network = Network::load("/root/source/rust/triton/save/net.json");
    
    new_net = new_net.train_to_loss(inputs, outputs, 0.00005, 50000, Mode::Avg, 0.1, 0.00001, 3, 10);
    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.feed_forward(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.feed_forward(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.feed_forward(&vec![0.0,0.0])[0]);
    println!("New network made: {:?}", new_net.layers);
    new_net.save("/home/braden/source/rust/triton/save/net.json");
}
