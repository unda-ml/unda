use unda::{core::{data::{input::Input, matrix::Matrix}, network::Sequential, layer::{layers::{LayerTypes, InputTypes}, methods::{activations::Activations, errors::ErrorTypes}}}, util::{mnist::MnistEntry, categorical::to_categorical}};

fn main() {
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
    let mut network = Sequential::new(128);

    network.set_input(InputTypes::DENSE(784));
    network.add_layer(LayerTypes::DENSE(256, Activations::RELU, 0.001));
    network.add_layer(LayerTypes::DENSE(64, Activations::RELU, 0.001));
    network.add_layer(LayerTypes::DENSE(10, Activations::SOFTMAX, 0.001));

    //network.set_log(false);

    network.compile();

    network.fit(&inputs, &true_outputs, 1, ErrorTypes::CategoricalCrossEntropy);
    for i in 0..10{
        println!("predicted: {:?} \n\nactual: {:?}\n\n\n", network.predict(inputs[i]), true_outputs[i]);
    }
}
