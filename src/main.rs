use unda::{core::{data::{input::Input, matrix::Matrix, matrix3d::Matrix3D}, network::Network, layer::{layers::{LayerTypes, InputTypes}, methods::{activations::Activations, errors::ErrorTypes}}}, util::{mnist::MnistEntry, categorical::to_categorical}};


#[tokio::main]
async fn main() {
    //Convolutional Example
    let mut inputs: Vec<&dyn Input> = vec![];
    let outputs: Vec<Vec<f32>>;
    let mut true_outputs: Vec<Vec<f32>> = vec![];

    let inputs_undyn: Vec<Matrix>;
    let outputs_uncat: Vec<usize>;

    println!("Generating MNIST....");
    (inputs_undyn, outputs_uncat) = MnistEntry::generate_mnist();
    println!("Done Generating MNIST");

    outputs = to_categorical(outputs_uncat);
    for i in 0..600{
        inputs.push(&inputs_undyn[i]);
        true_outputs.push(outputs[i].clone());
    }
    loop{
        let mut network = Network::new(128);

        network.set_input(InputTypes::DENSE(784));
        network.add_layer(LayerTypes::DENSE(64, Activations::RELU, 0.001));
        network.add_layer(LayerTypes::DENSE(32, Activations::RELU, 0.001));
        network.add_layer(LayerTypes::DENSE(10, Activations::SOFTMAX, 0.001));



        network.compile();

        network.fit(&inputs, &true_outputs, 5, ErrorTypes::CategoricalCrossEntropy);
        for i in 0..5{
            println!("predicted: {:?} \n\n actual: {:?}\n", network.predict(inputs[i]), true_outputs[i]);
        }
    }
    //Dense Example
    /*
    let mut inputs: Vec<&dyn Input> = vec![];
    let input_1 = vec![1.0,1.0];
    let input_2 = vec![vec![0.0], vec![1.0]];
    let input_3 = Matrix::from(vec![vec![1.0],vec![0.0]]);
    let input_4 = Matrix3D::from(vec![vec![vec![0.0,0.0]]]);
    inputs.push(&input_1);
    inputs.push(&input_2);
    inputs.push(&input_3);
    inputs.push(&input_4);
    
    let outputs: Vec<Vec<f32>> = vec![vec![0.0, 1.0],vec![1.0, 0.0],vec![1.0, 0.0], vec![0.0, 1.0]];

    loop{

        let mut new_net = Network::new(4);
        new_net.set_log(false);

        new_net.set_input(InputTypes::DENSE(2));
        new_net.add_layer(LayerTypes::DENSE(10, Activations::RELU, 0.001));
        new_net.add_layer(LayerTypes::DENSE(2, Activations::SOFTMAX, 0.001));

        new_net.compile();

        new_net.fit(&inputs, &outputs, 2, ErrorTypes::CategoricalCrossEntropy);

        println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0]));
        println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0]));
        println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0]));
        println!("0 and 0: {:?}\n", new_net.predict(&vec![0.0,0.0]));

        new_net.save("test.json");


        //println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0])[0]);
        //println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0])[0]);
        //println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0])[0]);
        //println!("0 and 0: {:?}\n", new_net.predict(&vec![0.0,0.0])[0]);
    }*/
}
