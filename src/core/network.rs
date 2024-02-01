use super::layer::layers::{Layer, LayerTypes};
use super::layer::methods::pair::GradientPair;
use super::data::matrix::Matrix;
use super::data::input::Input;
use super::serialize::ser_layer::SerializedLayer;

use rand::{RngCore, Rng, thread_rng};
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use serde::{Serialize, Deserialize};

use futures::stream::{StreamExt, FuturesUnordered};

use serde_json::{to_string, from_str};
use std::{io, fs};
use std::{
    fs::File,
    io::{Read,Write},
};

#[derive(Serialize, Deserialize)]
pub struct Network {
    batch_size: usize,
    pub layer_sizes: Vec<usize>,
    pub loss: f32,
    loss_train: Vec<f32>,
    pub layers: Vec<Box<dyn Layer>>,
    uncompiled_layers: Vec<LayerTypes>,
    seed: Option<String>,
    #[serde(skip)]
    #[serde(default = "Network::thread_rng")]
    rng: Box<dyn RngCore>
}

const ITERATIONS_PER_EPOCH: usize = 1000;

impl Network{
    fn thread_rng() -> Box<dyn RngCore> {
        Box::new(thread_rng())
    }
    ///Creates a new neural network that is completely empty
    ///
    ///Example:
    ///```
    ///use triton_grow::core::network::Network;
    ///let mut new_net = Network::new(10);
    ///```
    pub fn new(batch_size: usize) -> Network{
        Network{
            batch_size,
            layer_sizes: vec![],
            loss: 1.0,
            layers: vec![],
            uncompiled_layers: vec![],
            loss_train: vec![],
            seed: None,
            rng: Box::new(thread_rng())
        }
    } 
    pub fn get_layer_loss(&self) -> Vec<(f32, f32)> {
        let mut res: Vec<(f32, f32)> = vec![];
        for i in 0..self.layers.len() - 1{
            res.push(((i) as f32, self.layers[i].get_loss()));
        }
        res
    }
    pub fn get_loss_history(&self) -> Vec<f32> {
        self.loss_train.clone()
    }
    ///Adds a new Layer to the queue of a neural network
    ///
    ///# Arguments
    ///* `layer` - An enum depicting the options available from the Layers that exist(Dense,
    ///Convolutional, etc)
    ///
    ///# Example
    ///
    ///```
    ///use triton_grow::core::network::Network;
    ///use triton_grow::core::layer::layers::LayerTypes;
    ///use triton_grow::core::layer::methods::activations::Activations;
    ///
    ///let mut new_net = Network::new(2);
    ///new_net.add_layer(LayerTypes::DENSE(4, Activations::SIGMOID, 0.01));
    ///```
    ///Adds a new Dense layer of 4 nodes with the sigmoid activation and a learning rate of 0.01
    pub fn add_layer(&mut self, layer: LayerTypes){
        self.layer_sizes.push(layer.get_size());
        self.uncompiled_layers.push(layer);
    }
    ///Compiles a network by constructing each of its layers accordingly
    ///Must be done after all layers are added as the sizes of layer rows depends on the columns of
    ///the next layer
    pub fn compile(&mut self){
        for i in 0..self.uncompiled_layers.len() - 1 {
            let layer = self.uncompiled_layers[i].to_layer(self.layer_sizes[i+1], &mut self.rng);
            self.layers.push(layer);
        }
        
        //let final_layer = self.uncompiled_layers[self.uncompiled_layers.len()-1].to_layer(self.layer_sizes[self.layer_sizes.len()-1], &mut self.rng);
        //self.layers.push(final_layer);
    }
    pub fn predict(&mut self, input: &dyn Input) -> Vec<f32>{
        let in_box: Box<dyn Input> = input.to_box();
        self.feed_forward(&in_box)
    }
    pub fn set_seed(&mut self, seed: &str){
        self.seed = Some(String::from(seed));
        self.rng = self.get_rng();
    }

    async fn get_minibatch_gradient(&self, minibatch: &Vec<(Box<dyn Input>, Vec<f32>)>) -> (Vec<Box<dyn Input>>, Vec<Box<dyn Input>>) {
        let len = minibatch.len();
        let gradients = futures::stream::iter(minibatch)
            .map(|input_output| self.feed_forward_async(&input_output.0, &input_output.1))
            .buffer_unordered(len)
            .map(|data_output| self.back_propegate_async(data_output.0, data_output.1))
            .buffer_unordered(len)
            .collect::<FuturesUnordered<_>>()
            .await;

        let (mut bias_gradients,mut weight_gradients) = (vec![], vec![]);

        gradients.iter().for_each(|pair| {
            let mut gradient_bias = vec![];
            let mut gradient_weight = vec![];
            pair.iter().for_each(|GradientPair(bias, weight)| {
                gradient_bias.push(bias);
                gradient_weight.push(weight);
            });
            bias_gradients.push(gradient_bias);
            weight_gradients.push(gradient_weight);
        });
       
        let mut avg_weights_gradient:Vec<Box<dyn Input>> = vec![];
        let mut avg_bias_gradient:Vec<Box<dyn Input>> = vec![];

        for layer_gradient in 0..weight_gradients[0].len() {
            avg_bias_gradient.push(self.layers[layer_gradient].avg_gradient(bias_gradients.iter().map(|grad| grad[layer_gradient]).collect::<Vec<_>>()));
            avg_weights_gradient.push(self.layers[layer_gradient].avg_gradient(weight_gradients.iter().map(|grad| grad[layer_gradient]).collect::<Vec<_>>()));
        }
        //println!("{:?}", avg_bias_gradient.iter().map(|x| x.to_param()).collect::<Vec<_>>());
        //println!("{:?}", avg_weights_gradient.iter().map(|x| x.to_param()).collect::<Vec<_>>());
        (avg_bias_gradient, avg_weights_gradient)
    }

    ///Travels through a neural network's abstracted Layers and returns the resultant vector at the
    ///end
    ///
    ///# Arguments
    ///* `input_obj` - Any structure that implements the Input trait to act as an input to the data
    ///# Returns
    ///A vector at the end of the feed forward
    ///
    ///# Examples
    ///
    ///```
    ///use triton_grow::core::network::Network;
    ///use triton_grow::core::layer::layers::LayerTypes;
    ///use triton_grow::core::layer::methods::activations::Activations;

    ///let mut new_net = Network::new(4);
    ///new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    ///new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.01));
    ///new_net.add_layer(LayerTypes::DENSE(4, Activations::SIGMOID, 0.01));
    ///new_net.add_layer(LayerTypes::DENSE(2, Activations::TANH, 0.01));
    ///new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.01));
    ///
    ///new_net.compile();
    ///
    ///let res = new_net.predict(&vec![1.0, 0.54]);
    ///```
    fn feed_forward(&mut self, input_obj: &Box<dyn Input>) -> Vec<f32> {
        if input_obj.to_param().shape() != self.layers[0].shape(){
            panic!("Input shape does not match input layer shape \nInput: {:?}\nInput Layer:{:?}", input_obj.shape(), self.layers[0].shape());
        }
        
        let mut data_at: Box<dyn Input> = Box::new(input_obj.to_param());
        for i in 0..self.layers.len(){
            data_at = self.layers[i].forward(&data_at);
            self.layers[i].set_data(&data_at);
        }
        data_at.to_param().to_owned()
    }
    async fn feed_forward_async(&self, input_obj: &Box<dyn Input>, output: &Vec<f32>) -> (Vec<Box<dyn Input>>, Vec<f32>) {
        if input_obj.to_param().shape() != self.layers[0].shape(){
            panic!("Input shape does not match input layer shape \nInput: {:?}\nInput Layer:{:?}", input_obj.shape(), self.layers[0].shape());
        }
        
        let mut data_at: Box<dyn Input> = Box::new(input_obj.to_param());
        let mut res: Vec<Box<dyn Input>> = vec![data_at.to_box()];
        for i in 0..self.layers.len(){
            data_at = self.layers[i].forward(&data_at);
            res.push(data_at.to_box())
        }
        //Expected output is passed through asynchronously as well so we don't need to worry about
        //needing to get the response in order
        (res, output.clone())
    }
    ///Travels backwards through a neural network and updates weights and biases accordingly
    ///
    ///The backward behavior is different depending on the layer type, and therefore the weight and
    ///bias updating is different as well
    ///
    ///When constructing a neural network, be cautious that your layers behave well with each other
    fn back_propegate(&mut self, outputs: &Vec<f32>, target_obj: &Box<dyn Input>) {
        let mut parsed = Matrix::from(outputs.to_param_2d());
        
        if let None = self.layers[self.layers.len()-1].get_activation() {
            panic!("Output layer is not a dense layer");
        }
        
        let mut gradients: Box<dyn Input>;
        let mut errors: Box<dyn Input> = Box::new((parsed - &Matrix::from(target_obj.to_param_2d())).transpose());

        for i in (0..self.layers.len() - 1).rev() {
            gradients = self.layers[i + 1].update_gradient();
            let data_box: Box<dyn Input> = self.layers[i].get_data();
            errors = self.layers[i+1].backward(gradients, errors, data_box);
        }
    }
    async fn back_propegate_async(&self, data: Vec<Box<dyn Input>>, output: Vec<f32>) -> Vec<GradientPair> {
        let mut res = vec![];
        let parsed = Matrix::from(data[data.len()-1].to_param_2d());
        
        if let None = self.layers[self.layers.len()-1].get_activation() {
            panic!("Output layer is not a dense layer");
        }
        
        let mut errors: Box<dyn Input> = Box::new(Matrix::from(output.to_param_2d()).transpose() - &parsed);

        //println!("{}", data.len());

        for i in (0..self.layers.len()).rev() {
            res.push(self.layers[i].get_gradients(&data[i + 1], &data[i], &errors));
            errors = self.layers[i].update_errors(errors);
        }
        res.reverse();
        res
    }

    ///Trains a neural network by iteratively feeding forward a series of inputs and then doing
    ///back propegation based on the outputs supplied
    ///
    ///# Arguments
    ///* `train_in` - A vector of objects that implement the Input trait, used as the training
    ///input
    ///* `train_out` - A vector of objects that implement the Input trait, used as the results
    ///compared to what is actually derived during back propegation
    ///* `epochs` - How many epochs you want your model training for
    ///
    pub fn fit(&mut self, train_in: &Vec<&dyn Input>, train_out: &Vec<Vec<f32>>, epochs: usize) {
        self.loss_train = vec![];

        let mut loss: f32;
        let num_batches = train_in.len() / self.batch_size;

        let mut iterations_per_epoch: usize = 40;

        if train_in.len() < ITERATIONS_PER_EPOCH {
            let iteration_scale_factor = ITERATIONS_PER_EPOCH / train_in.len();
            iterations_per_epoch = (iteration_scale_factor as f32 * 25.0).ceil() as usize;
        }
        println!("{}", iterations_per_epoch);
        let iterations_divided_even = iterations_per_epoch / 40;

        for epoch in 0..epochs {
            let _ = io::stdout().flush();
            print!("Epoch {}: [", epoch+1);
            loss = 0.0;
            for iteration in 0..iterations_per_epoch {
                if iteration % iterations_divided_even == 0 {
                    let _ = io::stdout().flush();
                    print!("=");
                }
                for batch_index in 0..num_batches {
                    let start = batch_index * self.batch_size;
                    let end = start + self.batch_size;
                    let end = end.min(train_in.len()); // Ensure 'end' doesn't go out of bounds

                    let mut batch_loss: f32 = 0.0;
                    for input_index in start..end {
                        let mut loss_on_input: f32 = 0.0;
                        let input: Box<dyn Input> = train_in[input_index].to_box();
                        let output: Box<dyn Input> = Box::new(train_out[input_index].clone());
                        let outputs = self.feed_forward(&input);
                        self.back_propegate(&outputs, &output);

                        for i in 0..outputs.len() {
                            loss_on_input += (outputs[i] - train_out[input_index].to_param()[i]).powi(2);
                        }
                        batch_loss += loss_on_input / outputs.len() as f32;
                    }
                    loss += batch_loss / self.batch_size as f32;
                }
            }
            self.loss_train.push(loss / (iterations_per_epoch * num_batches) as f32);
            println!("] Loss: {}", self.loss_train[self.loss_train.len()-1]);
        }

        self.loss = self.loss_train[self.loss_train.len() - 1];
        println!("Trained to a loss of {:.2}%", self.loss * 100.0);
        for i in 0..self.layers.len() - 1 {
            println!("Error on layer {}: +/- {:.2}", i + 1, self.layers[i].get_loss());
        }
    }

    fn update_gradients(&mut self, gradient_pairs: &(Vec<Box<dyn Input>>, Vec<Box<dyn Input>>)) {//, noise: &f32) {
        if gradient_pairs.0.len() != self.layers.len() {
            panic!("Gradients length not equal to number of layers:
                   \nGradients: {}\nLayers: {}", 
                   gradient_pairs.0.len(),
                   self.layers.len());
        }
        for i in 0..self.layers.len() {
            self.layers[i].update_gradients((&gradient_pairs.0[i], &gradient_pairs.1[i]), None);//Some(-1.0..1.0));//, noise);
        }
    }
    pub async fn fit_minibatch(&mut self, train_in: &Vec<&dyn Input>, train_out: &Vec<Vec<f32>>, epochs: usize) {
        let _ = io::stdout().flush();
        print!("[");
        for _ in 1..=epochs {
            let minibatches: Vec<Vec<(Box<dyn Input>, Vec<f32>)>> = 
                self.generate_minibatches(train_in.clone(), train_out.clone());
            let len = minibatches.len();
            let _ = io::stdout().flush();
            print!("#");
            let all_gradients = futures::stream::iter(&minibatches)
                .map(|batch| self.get_minibatch_gradient(batch))
                .buffer_unordered(len)
                .collect::<FuturesUnordered<_>>();
            let res = all_gradients.await;
            for gradient_pair in res.iter() {
                self.update_gradients(&gradient_pair);
            }
        }
        println!("]");
    }

   
    fn generate_minibatches(&self,mut inputs: Vec<&dyn Input>,mut outputs: Vec<Vec<f32>>) -> Vec<Vec<(Box<dyn Input>, Vec<f32>)>> {
        let mut res = vec![];
        let mut rng = self.get_rng();

        let mut minibatch: Vec<(Box<dyn Input>, Vec<f32>)>;

        let mut iterations: usize;
        while inputs.len() > 0 {
            minibatch = vec![];
            iterations = inputs.len().min(self.batch_size);
            for _ in 0..iterations {
                let location = rng.gen_range(0..inputs.len());
                minibatch.push((inputs[location].to_box(), outputs[location].clone()));
                inputs.remove(location);
                outputs.remove(location);
            }
            res.push(minibatch);
        }
        res
    }

    fn get_rng(&self) -> Box<dyn RngCore> {
        match &self.seed {
            Some(seed_rng) => Box::new(Seeder::from(seed_rng).make_rng::<Pcg64>()),
            None => Box::new(rand::thread_rng())
        }
    }

    pub fn save(&self, path: &str) {
        let mut file = File::create(path).expect("Unable to hit save file :(");
        let file_ser = to_string(self).expect("Unable to serialize network :(((");
        file.write_all(file_ser.to_string().as_bytes()).expect("Write failed :(");
    }
    pub fn save_cbor(&self, path: &str) {
        let res_file = File::create(path).expect("Unable to save file");
        serde_cbor::to_writer(res_file, self).expect("Unable to write or compile cbor");
    }
    pub fn load(path: &str) -> Network{
        let mut buffer = String::new();
        let mut file = File::open(path).expect("Unable to read file :(");

        file.read_to_string(&mut buffer).expect("Unable to read file but even sadder :(");

        let mut net: Network = from_str(&buffer).expect("Json was not formatted well >:(");
        net.rng = net.get_rng();
        net
    }
    pub fn load_cbor(path: &str) -> Result<Network, serde_cbor::Error> {
        let file = File::open(path).expect("error loading file");
        let mut network: Network = serde_cbor::from_reader(file)?;
        network.rng = network.get_rng();
        Ok(network)
    }
    pub fn to_vec(&self) -> Result<Vec<u8>, serde_cbor::Error> {
        serde_cbor::to_vec(self)
    }
    pub fn from_vec(data: Vec<u8>) -> Result<Network, serde_cbor::Error> {
        serde_cbor::from_slice(&data[..])
    }

    pub fn serialize_triton_fmt(&self, path: &str) {
        let mut str_fmt: Vec<String> = vec![];
        for i in 0..self.layers.len() {
            let layer_serialized: SerializedLayer = SerializedLayer::new(&self.layers[i], &self.uncompiled_layers[i]);
            str_fmt.push(layer_serialized.to_string());
        }
        fs::write(path, str_fmt.join("#")).expect("Error writing to file");
    }
    pub fn deserialize_triton_fmt_string(format_string: String) -> Network {
        let mut net: Network = Network::new(0);
        let parse_triton = format_string.split("#");
        for layer in parse_triton {
            let new_layer: Box<dyn Layer> = SerializedLayer::from_string(layer.to_string()).from();
            net.layers.push(new_layer);
        }

        net
    }
}
