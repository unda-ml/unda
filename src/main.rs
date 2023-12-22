use std::error::Error;

use triton_grow::network::{network::Network, activations::Activations, modes::Mode, layer::layers::LayerTypes, input::Input};
use triton_grow::helper::data_vis;

fn main() -> Result<(), Box<dyn Error>> {
    let inputs: Vec<Vec<f32>> = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0], vec![1.0,1.0]];
    let outputs: Vec<Vec<f32>> = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new(16);

    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(4, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::TANH, 0.01));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.01));

    new_net.compile();

    new_net.fit(inputs, outputs, 100);

    //let mut new_net = Network::load("best_network.json");
    
    println!("1 and 0: {:?}", new_net.predict(vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(vec![0.0,0.0])[0]);
    new_net.save("best_network.json");
    new_net.plot_layer_loss("loss_graph_per_layer.png")?;
    new_net.plot_loss_history("loss_history.png")?;
    Ok(())
}
