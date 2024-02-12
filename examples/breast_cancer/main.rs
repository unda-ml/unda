use std::error::Error;
use unda::core::{data::input::Input, network::Network, layer::{layers::{LayerTypes, InputTypes}, methods::{activations::Activations, errors::ErrorTypes}}};


use rand::seq::SliceRandom;

mod med_model;
use med_model::MedModel;


fn main() -> Result<(), Box<dyn Error>>{
    let mut models: Vec<(MedModel, f32)> = MedModel::get_from_path("examples/breast_cancer/data/data.csv")?;
    let mut rng = rand::thread_rng();
    models.shuffle(&mut rng);

    let mut inputs: Vec<&dyn Input> = vec![];
    let mut outputs: Vec<Vec<f32>> = vec![];
    models.iter_mut().for_each(|model| {
        outputs.push(vec![model.1]);
        inputs.push(&model.0);
    });

    let mut network = Network::new(128);
    network.set_input(InputTypes::DENSE(inputs[0].to_param().len()));
    network.add_layer(LayerTypes::DENSE(16, Activations::RELU, 0.001));
    network.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));

    network.compile();

    network.fit(&inputs[0..inputs.len()-100].to_vec(), &outputs[0..outputs.len()-100].to_vec(), 1, ErrorTypes::CategoricalCrossEntropy);

    for i in inputs.len()-10..inputs.len(){
        println!("Model output: {:?}\nActual: {:?}", network.predict(&inputs[i].to_param()), outputs[i]);
    }

    Ok(())
}
