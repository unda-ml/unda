use triton_grow::network::{network::Network, activations, modes::Mode};

fn main() {
    let mut inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    let mut outputs = vec![vec![0.0],vec![1.0],vec![1.0],vec![0.0]];
    let mut new_net: Network = Network::new(vec![2,2,3,1], activations::SIGMOID, 0.1);
    
    new_net.train_to_loss(inputs.clone(), outputs.clone(), 0.005, 1000, Mode::Avg);
    println!("loss: {}", new_net.get_loss(inputs.clone(),outputs.clone(), &Mode::Avg));
    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0]));

}
