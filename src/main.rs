use triton_grow::network::{network::Network, activations::Activations, modes::Mode, layer::layers::LayerTypes};

fn main() {
    let inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0], vec![1.0,1.0]];
    let outputs: Vec<Vec<f32>> = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new();

    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.01));

    new_net.compile();

    new_net.fit(inputs, outputs, 100);

    //let mut new_net = Network::load("best_network.json");
    
    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.feed_forward(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.feed_forward(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.feed_forward(&vec![0.0,0.0])[0]);
    new_net.save("best_network.json");
}
