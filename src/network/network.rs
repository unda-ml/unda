use super::layer::layers::{Layer, LayerTypes};
use super::{matrix::Matrix, modes::Mode};
use super::input::Input;
use serde::{Serialize, Deserialize};

use serde_json::{to_string, from_str};
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
}

const ITERATIONS_PER_EPOCH: usize = 10000;

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
            loss_train: vec![]
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
        for i in 0..self.uncompiled_layers.len() - 1 {
            let layer = self.uncompiled_layers[i].to_layer(self.layer_sizes[i+1]);
            self.layers.push(layer);
        }
        //println!("{:?}", self.layer_sizes);

    }
    pub fn predict(&mut self, input: Vec<f32>) -> Vec<f32>{
        let in_box: Box<dyn Input> = Box::new(input);
        self.feed_forward(&in_box)
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

        if input_obj.shape().0 != self.layers[0].shape().1{
            panic!("Input shape does not match input layer shape \nInput: {:?}\nInput Layer:{:?}", input_obj.shape(), self.layers[0].shape());
        }

        let mut data_at: Box<dyn Input> = Box::new(input_obj.to_param());
        for i in 0..self.layers.len(){
            data_at = self.layers[i].forward(&data_at);
        }
        data_at.to_param().to_owned()
    }
    ///Travels backwards through a neural network and updates weights and biases accordingly
    ///
    ///The backward behavior is different depending on the layer type, and therefore the weight and
    ///bias updating is different as well
    ///
    ///When constructing a neural network, be cautious that your layers behave well with each other
    fn back_propegate(&mut self, outputs: Vec<f32>, target_obj: &Box<dyn Input>) {
        let targets = target_obj.to_param();
        if targets.len() != self.layer_sizes[self.layer_sizes.len()-1]{
            panic!("Output size does not match network output size");
        }
        let mut parsed = Matrix::from(outputs.to_param_2d()).transpose();
        
        let mut errors = Matrix::from(targets.to_param_2d()) - &parsed; 
        
        if let None = self.layers[self.layers.len()-1].get_activation() {
            panic!("Output layer is not a dense layer");
        }

        let mut gradients = parsed.map(self.layers[self.layers.len()-1].get_activation().unwrap().get_function().derivative);
        let target_matrix = Matrix::from(vec![targets.clone()]);
        let mut new_weights: Matrix;
        let mut new_bias: Matrix;
        for i in (0..self.layers.len() - 1).rev() {
            let layers_prev = self.layers[i+1].get_weights();
            let bias_prev = self.layers[i+1].get_bias();
            (new_bias, new_weights, gradients, errors) = self.layers[i].backward(&target_matrix, &gradients, &errors, &layers_prev, &bias_prev);
            self.layers[i+1].set_weights(new_weights);
            self.layers[i+1].set_bias(new_bias);
        }
    }
    fn implicit_back_propegation(&mut self, target_matrix: Matrix, mut gradients: Matrix, mut errors: Matrix) -> (Matrix, Matrix, Matrix, Matrix){
        let mut new_weights: Matrix = Matrix::new_random(0, 0);
        let mut new_bias: Matrix = Matrix::new_random(0, 0);
        for i in (0..self.layers.len() - 1).rev() {
            let layers_prev = self.layers[i+1].get_weights();
            let bias_prev = self.layers[i+1].get_bias();
            (new_bias, new_weights, gradients, errors) = self.layers[i].backward(&target_matrix, &gradients, &errors, &layers_prev, &bias_prev);
            self.layers[i+1].set_weights(new_weights.clone());
            self.layers[i+1].set_bias(new_bias.clone());
        }

        (new_bias, new_weights, gradients, errors)
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
    pub fn fit(&mut self, train_in: Vec<Vec<f32>>, train_out: Vec<Vec<f32>>, epochs: usize){
        let mut input_batch: Box<dyn Input>;
        let mut output_batch: Box<dyn Input>;

        self.loss_train = vec![];
        let mut loss: f32;
        for _ in 0..epochs {
            loss = 0.0;
            for i in 0..(train_in.len() / self.batch_size){

                input_batch = self.get_batch(&train_in, i);
                let inputs = input_batch.to_param_2d();
                output_batch = self.get_batch(&train_out, i);
                let outputs = output_batch.to_param_2d();

                for _ in 0..ITERATIONS_PER_EPOCH{
                    for input in 0..inputs.len(){
                        let mut loss_on_input: f32 = 0.0;
                        let inp: Box<dyn Input> = Box::new(inputs[input].clone());
                        let out: Box<dyn Input> = Box::new(outputs[input].clone());
                        let outputs = self.feed_forward(&inp);
                        self.back_propegate(outputs.clone(), &out);
                        for i in 0..outputs.len(){
                            loss_on_input += (outputs[i] - out.to_param()[i]).powi(2);
                        }
                        loss += loss_on_input / outputs.len() as f32;
                    }
                }
            }
            self.loss_train.push(loss / (ITERATIONS_PER_EPOCH * train_out.len()) as f32);
        }
        self.loss = self.loss_train[self.loss_train.len()-1];
        println!("Trained to a loss of {:.2}%", self.loss * 100.0);
        for i in 0..self.layers.len()-1{
            println!("Error on layer {}: +/- {:.2}", i+1, self.layers[i].get_loss());
        }
    }

    fn get_batch(&self, inputs: &Vec<Vec<f32>>, idx: usize) -> Box<dyn Input> {
        let mut res: Vec<Vec<f32>> = vec![];
        let get_range = (idx * self.batch_size)..(idx * self.batch_size + self.batch_size);
        for i in get_range{
            if i < inputs.len(){
                res.push(inputs[i].clone());
            } else{
                break;
            }
        }
        //println!("{:?}", res);
        Box::new(res)
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
}

#[typetag::serde]
impl Layer for Network{

    fn forward(&mut self,inputs: &Box<dyn Input>) -> Box<dyn Input> {
        Box::new(self.feed_forward(inputs))
    }

    fn backward(&mut self,inputs: &Matrix,gradients: &Matrix,errors: &Matrix,layer_prev: &Matrix,layer_prev_bias: &Matrix) -> (Matrix,Matrix,Matrix,Matrix) {
        self.implicit_back_propegation(inputs.clone(), gradients.clone(), errors.clone())
    }

    fn shape(&self) -> (usize,usize,usize) {
        self.layers[0].shape()
    }
    fn set_weights(&mut self,new_weights:Matrix) {
        self.layers[0].set_weights(new_weights)
    }
    fn set_bias(&mut self,new_bias:Matrix) {
        self.layers[0].set_bias(new_bias)
    }
    fn get_cols(&self) -> usize {
        self.layers[self.layers.len() - 1].get_cols()
    }
    fn get_rows(&self) -> usize {
        self.layers[self.layers.len() - 1].get_rows()
    }
    fn get_bias(&self) -> Matrix {
        self.layers[0].get_bias()
    }
    fn get_loss(&self) -> f32 {
        self.loss
    }
    fn get_weights(&self) -> Matrix {
        self.layers[0].get_weights()
    }
    fn get_activation(&self) -> Option<super::activations::Activations> {
        self.layers[0].get_activation()
    }
}
