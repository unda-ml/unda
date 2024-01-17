use super::layer::layers::{Layer, LayerTypes};
use super::matrix::Matrix;
use super::input::Input;
use serde::{Serialize, Deserialize};

use futures::{stream::{StreamExt}};

use serde_json::{to_string, from_str};
use std::io;
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
    seed: Option<String>
}

const ITERATIONS_PER_EPOCH: usize = 1000;

impl Network{
    ///Creates a new neural network that is completely empty
    ///
    ///Example:
    ///```
    ///let mut new_net = Network::new();
    ///```
    pub fn new(batch_size: usize) -> Network{
        Network{
            batch_size,
            layer_sizes: vec![],
            loss: 1.0,
            layers: vec![],
            uncompiled_layers: vec![],
            loss_train: vec![],
            seed: None
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
    ///let mut new_net = Network::new();
    ///new_new.add_layer(LayerTypes::Dense(4, Activations::SIGMOID, 0.01));
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
        let input_size = self.uncompiled_layers[0].get_size();
        for i in 0..self.uncompiled_layers.len() - 1 {
            let layer = self.uncompiled_layers[i].to_layer(self.layer_sizes[i+1], &self.seed, input_size);
            self.layers.push(layer);
        }
        //println!("{:?}", self.layer_sizes);

    }
    pub fn predict(&mut self, input: &dyn Input) -> Vec<f32>{
        let in_box: Box<dyn Input> = input.to_box();
        self.feed_forward(&in_box)
    }
    pub fn set_seed(&mut self, seed: &str){
        self.seed = Some(String::from(seed));
    }

    async fn get_minibatch_gradients(&self, minibatch: Vec<(&Box<dyn Input>, Vec<f32>)>) -> Vec<Vec<Box<dyn Input>>> {
        let _len = minibatch.len();
        let _minibatch_futures = futures::stream::iter(minibatch)
            .map(|input| self.feed_forward_async(input.0, input.1)); //gives us an iterator of all data of every input (Vec<Box<dyn Input>)

        vec![]
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
    ///let new_net = Network::New();
    ///new_new.add_layer(LayerTypes::Dense(2, Activations::SIGMOID, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(3, Activations::SIGMOID, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(4, Activations::SIGMOID, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(2, Activations::TANH, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(1, Activations::SIGMOID, 0.01));
    ///
    ///new_net.compile()
    ///
    ///let res = new_net.feed_forward(vec![1.0, 0.54]);
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
    async fn feed_forward_async(&self, input_obj: &Box<dyn Input>, output: Vec<f32>) -> (Vec<Box<dyn Input>>, Vec<f32>) {
        let mut res: Vec<Box<dyn Input>> = vec![];
        if input_obj.to_param().shape() != self.layers[0].shape(){
            panic!("Input shape does not match input layer shape \nInput: {:?}\nInput Layer:{:?}", input_obj.shape(), self.layers[0].shape());
        }
        
        let mut data_at: Box<dyn Input> = Box::new(input_obj.to_param());
        for i in 0..self.layers.len(){
            data_at = self.layers[i].forward(&data_at);
            res.push(data_at.to_box())
        }
        //Expected output is passed through asynchronously as well so we don't need to worry about
        //needing to get the response in order
        (res, output)
    }
    ///Travels backwards through a neural network and updates weights and biases accordingly
    ///
    ///The backward behavior is different depending on the layer type, and therefore the weight and
    ///bias updating is different as well
    ///
    ///When constructing a neural network, be cautious that your layers behave well with each other
    fn back_propegate(&mut self, outputs: Vec<f32>, target_obj: &Box<dyn Input>) {
        let parsed = Matrix::from(outputs.to_param_2d());
        
        if let None = self.layers[self.layers.len()-1].get_activation() {
            panic!("Output layer is not a dense layer");
        }
        
        let mut gradients: Box<dyn Input>;
        let mut errors: Box<dyn Input> = Box::new((Matrix::from(target_obj.to_param_2d()) - &parsed).transpose());

        for i in (0..self.layers.len() - 1).rev() {
            gradients = self.layers[i + 1].update_gradient();
            let data_box: Box<dyn Input> = self.layers[i].get_data();
            errors = self.layers[i+1].backward(gradients, errors, data_box);
        }
    }
    async fn back_propegate_async(&self, outputs: Vec<f32>, target_obj: &Box<dyn Input>) -> Vec<Box<dyn Input>> {
        let mut res = vec![];
        let parsed = Matrix::from(outputs.to_param_2d());
        
        if let None = self.layers[self.layers.len()-1].get_activation() {
            panic!("Output layer is not a dense layer");
        }
        
        let mut gradients: Box<dyn Input>;
        let _errors: Box<dyn Input> = Box::new((Matrix::from(target_obj.to_param_2d()) - &parsed).transpose());

        for i in (0..self.layers.len() - 1).rev() {
            gradients = self.layers[i + 1].update_gradient();
            res.push(gradients);
        }
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
            io::stdout().flush();
            print!("Epoch {}: [", epoch+1);
            loss = 0.0;
            for iteration in 0..iterations_per_epoch {
                if iteration % iterations_divided_even == 0 {
                    io::stdout().flush();
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
                        self.back_propegate(outputs.clone(), &output);

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

    pub fn save(&self, path: &str) {
        let mut file = File::create(path).expect("Unable to hit save file :(");
        let file_ser = to_string(self).expect("Unable to serialize network :(((");
        file.write_all(file_ser.to_string().as_bytes()).expect("Write failed :(");
    }
    pub fn load(path: &str) -> Network{
        let mut buffer = String::new();
        let mut file = File::open(path).expect("Unable to read file :(");

        file.read_to_string(&mut buffer).expect("Unable to read file but even sadder :(");

        let net: Network = from_str(&buffer).expect("Json was not formatted well >:(");
        net
    }

    pub fn fit_to_loss(&mut self, train_in: Vec<&dyn Input>, train_out: Vec<Vec<f32>>, desired_loss: f32, steps_per: usize) -> usize{
        let mut epochs_total = 0;
        while self.loss > desired_loss{
            self.fit(&train_in, &train_out, steps_per);
            epochs_total += steps_per;
            
        }
        println!("Trained to a loss of {:.2}%", self.loss * 100.0);
        for i in 0..self.layers.len()-1{
            println!("Error on layer {}: +/- {:.2}", i+1, self.layers[i].get_loss());
        }
        epochs_total
    }
}
