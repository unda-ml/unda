use unda::{core::{data::{input::Input, matrix::Matrix, matrix3d::Matrix3D}, network::Network, layer::{layers::{LayerTypes, InputTypes}, methods::{activations::Activations, errors::ErrorTypes}}}, util::{mnist::MnistEntry, categorical::to_categorical}};

fn main() {
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

    loop{

        let mut new_net = Network::new(4);
        new_net.set_log(false);

        new_net.set_input(InputTypes::DENSE(2));
        new_net.add_layer(LayerTypes::DENSE(10, Activations::RELU, 0.001));
        new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));

        new_net.compile();

        new_net.fit(&inputs, &outputs, 2, ErrorTypes::CategoricalCrossEntropy);

        println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0]));
        println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0]));
        println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0]));
        println!("0 and 0: {:?}\n", new_net.predict(&vec![0.0,0.0]));
    }
}
