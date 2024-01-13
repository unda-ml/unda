use rand_distr::Distribution;
use triton_grow::{network::{network::Network, activations::Activations, layer::{layers::LayerTypes, conv::Convolutional, distributions::Distributions}, input::*, matrix::Matrix, matrix3d::Matrix3D}, helper::{mnist::MnistEntry, categorical::to_categorical}};

fn main() {
    //Convolutional Example
    //
    let mut inputs: Vec<&dyn Input> = vec![];
    let outputs: Vec<Vec<f32>>;

    let inputs_undyn: Vec<Matrix>;
    let outputs_uncat: Vec<usize>;

    println!("Generating MNIST....");
    (inputs_undyn, outputs_uncat) = MnistEntry::generate_mnist();
    println!("Done Generating MNIST");

    outputs = to_categorical(outputs_uncat);
    for i in 0..inputs_undyn.len(){
        inputs.push(&inputs_undyn[i]);
    }

    let mut network = Network::new(128);

    network.add_layer(LayerTypes::DENSE(784, Activations::RELU, 0.1));
    network.add_layer(LayerTypes::DENSE(64, Activations::RELU, 0.1));
    network.add_layer(LayerTypes::DENSE(32, Activations::RELU, 0.1));
    network.add_layer(LayerTypes::DENSE(10, Activations::RELU, 0.1));

    network.compile();

    network.fit(&inputs, &outputs, 1);

    println!("{:?}", network.predict(inputs[0]));
    
    //Dense Example
    //
    /*let mut inputs: Vec<&dyn Input> = vec![];
    let input_1 = vec![1.0,1.0];
    let input_2 = vec![vec![0.0], vec![1.0]];
    let input_3 = Matrix::from(vec![vec![1.0],vec![0.0]]);
    let input_4 = Matrix3D::from(vec![vec![vec![0.0,0.0]]]);
    inputs.push(&input_1);
    inputs.push(&input_2);
    inputs.push(&input_3);
    inputs.push(&input_4);
    
    let outputs: Vec<Vec<f32>> = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new(4);


    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.1));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.1));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.1));

    //new_net.set_seed("teller");
    //new_net.set_seed("I said");
    new_net.compile();

    new_net.fit(&inputs, &outputs, 100);

    println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(&vec![0.0,0.0])[0]);*/
}
