use triton_grow::network::{network::Network, activations::Activations, layer::{layers::LayerTypes, conv::Convolutional}, input::Input, matrix::Matrix};

fn main() {
    let inputs: Vec<Vec<f32>> = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0], vec![1.0,1.0]];
    let outputs: Vec<Vec<f32>> = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new(4);

    new_net.add_layer(LayerTypes::DENSE(2, Activations::RELU, 0.1));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::RELU, 0.1));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::RELU, 0.1));

    //new_net.set_seed("teller");
    //new_net.set_seed("I said");
    new_net.compile();

    new_net.fit(&inputs, &outputs, 100);

    println!("1 and 0: {:?}", new_net.predict(vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(vec![0.0,0.0])[0]);
}
