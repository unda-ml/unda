use triton_grow::{network::{network::Network, activations::Activations, layer::layers::LayerTypes, input::*, matrix::Matrix, matrix3d::Matrix3D}, helper::{categorical::to_categorical, mnist::MnistEntry}};

#[tokio::main]
async fn main() {
    //Convolutional Example
    /*
    let mut inputs: Vec<&dyn Input> = vec![];
    let outputs: Vec<Vec<f32>>;
    let mut true_outputs: Vec<Vec<f32>> = vec![];

    let inputs_undyn: Vec<Matrix>;
    let outputs_uncat: Vec<usize>;

    println!("Generating MNIST....");
    (inputs_undyn, outputs_uncat) = MnistEntry::generate_mnist();
    println!("Done Generating MNIST");

    outputs = to_categorical(outputs_uncat);
    for i in 0..500{
        inputs.push(&inputs_undyn[i]);
        true_outputs.push(outputs[i].clone());
    }

    let mut network = Network::new(10);

    network.add_layer(LayerTypes::DENSE(784, Activations::RELU, 0.1));
    network.add_layer(LayerTypes::DENSE(64, Activations::RELU, 0.1));
    network.add_layer(LayerTypes::DENSE(32, Activations::RELU, 0.1));
    network.add_layer(LayerTypes::DENSE(10, Activations::SOFTMAX, 0.1));

    

    network.compile();

    network.fit_minibatch(&inputs, &true_outputs, 1000).await;
    for i in 0..10{
        println!("predicted: {:?} \n\n actual: {:?}\n", network.predict(inputs[i]), true_outputs[i]);
    }
    network.save("mnist_weight_check.json");*/
    //Dense Example
    let mut inputs: Vec<&dyn Input> = vec![];
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

    new_net.add_layer(LayerTypes::DENSE(2, Activations::RELU, 0.0001));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::RELU, 0.0001));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.0001));

    new_net.compile();

    new_net.fit_minibatch(&inputs, &outputs, 500).await;
    println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(&vec![0.0,0.0])[0]);
}
