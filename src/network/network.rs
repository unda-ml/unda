use rand::Rng;

use super::{matrix::Matrix, activations::Activations, modes::Mode};
use serde::{Serialize, Deserialize};
use serde_json::{to_string, from_str};
use std::{
    fs::File,
    io::{Read,Write},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<usize>,
    pub loss: f32,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f32,
    activation: Activations
}

impl<'a> Network{
    ///Creates a new neural network instance filled with random neuron weights and biases
    ///
    ///layers: a vector with the size of each column you desire
    ///EX: vec![1,10,5,3];
    ///activation: the activation function you wish to use
    ///learning_rate: the steps taken during back propegation to get closer to the desired values
    ///
    ///Example:
    ///```
    ///let mut new_net = Network::new(vec![2,4,8,14,1], Activation::SIGMOID, 0.1);
    ///```
    ///Creates a new neural net with 2 input parameters, 3 hidden layers with sizes 4, 8 and 14,
    ///and 1 output
    pub fn new(layers: Vec<usize>, activation: Activations, learning_rate: f32) -> Network{
        let mut net = Network{
            layers: layers,
            loss: 1.0,
            weights: vec![],
            biases: vec![],
            data: vec![],
            learning_rate: learning_rate,
            activation
        };

        for i in 0..net.layers.len() - 1 {
            net.weights.push(Matrix::new_random(net.layers[i+1],net.layers[i]));
            net.biases.push(Matrix::new_random(net.layers[i+1], 1));
        }
        net
    }
    pub fn insert_at(&mut self, newlayer_pos: usize, newlayer_len: usize){
        if !(newlayer_pos > 0 && newlayer_pos < self.layers.len()){
            panic!("Attempting to add a new layer in an invalid slot [out of bounds, input layer, output layer]");
        }
        self.layers.insert(newlayer_pos,newlayer_len);
        

        self.weights.insert(newlayer_pos ,Matrix::new_random(self.layers[newlayer_pos+1], self.layers[newlayer_pos]));
        self.biases.insert(newlayer_pos, Matrix::new_random(self.layers[newlayer_pos+1], 1));

    }
    ///Creates a neural network from one network, but splices in a new column for dynamic growth
    ///At column {newlayer_pos}, insert a new layer. The layer initially at that spot is randomized
    ///for its matrices and spliced in two, acting as a wrapper around the new layer
    ///
    ///This method allows for most weights to remain constant, with only the new layer and the
    ///wrapper layer having new weights.
    ///
    ///Example:
    ///```
    ///let mut new_net = Network::new(vec![2,4,8,14,1], activation::SIGMOID, 0.1);
    ///let mut newer_net = Network::from(&new_net, 2, 10);
    ///
    ///assert_eq!(newer_net.layers, vec![2,4,8,10,8,14,1]);
    ///```
    pub fn from(network_from: Network, newlayer_pos: usize, newlayer_len: usize) -> Network{
        if !(newlayer_pos > 0 && newlayer_pos < network_from.layers.len()){
            panic!("Attempting to add a new layer in an invalid slot [out of bounds, input layer, output layer]");
        }
        let mut new_net = Network { 
            layers: vec![],
            weights: vec![],
            biases: vec![],
            data: vec![],
            loss: network_from.loss,
            learning_rate: network_from.learning_rate,
            activation: network_from.activation.clone()
        };
        for i in 0..network_from.layers.len(){
            new_net.layers.push(network_from.layers[i]);
            if i == newlayer_pos{
                new_net.layers.push(newlayer_len);

                new_net.layers.push(network_from.layers[i]);
            }
        }
        
        for i in 0..network_from.layers.len() - 1 {

            if i == newlayer_pos {
                new_net.weights.push(Matrix::new_random(newlayer_len, new_net.layers[i]));
                new_net.biases.push(Matrix::new_random(newlayer_len, 1));

                new_net.weights.push(Matrix::new_random(network_from.layers[i], newlayer_len));
                new_net.biases.push(Matrix::new_random(network_from.layers[i], 1));


                new_net.weights.push(Matrix::new_random(network_from.layers[i+1], network_from.layers[i]));
                new_net.biases.push(Matrix::new_random(network_from.layers[i+1], 1));

            }else{

                new_net.weights.push(network_from.weights[i].clone());
                new_net.biases.push(network_from.biases[i].clone());
            }

        }
        new_net
    }
    pub fn feed_to_point(&mut self, inputs: &Vec<f32>, layer_to: usize) -> Vec<f32>{
         if inputs.len() != self.layers[0]{
             panic!("Invalid numer of inputs");
         }
         if layer_to >= self.layers.len(){
             panic!("To destination is larger than network size");
         }
         let mut current = Matrix::from(vec![inputs.clone()]).transpose();
         self.data = vec![current.clone()];

         for i in 0..layer_to{
             current = (&(&self.weights[i]
                        * &current)
                        + &self.biases[i])
                        .map(self.activation.get_function().function);
             self.data.push(current.clone());
         }
         current.transpose().data[0].to_owned()
    }
    pub fn point_to_feed(&mut self, inputs_at_point: &Vec<f32>, layer_at: usize) -> Vec<f32> {
        if inputs_at_point.len() != self.layers[layer_at]{
            panic!("Invalid number of inputs for layer");
        }
        let mut current = Matrix::from(vec![inputs_at_point.clone()]).transpose();
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len()-1{
            current = (&(&self.weights[i]
                        * &current)
                        + &self.biases[i])
                        .map(self.activation.get_function().function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs");
        }
         
        let mut current = Matrix::from(vec![inputs.clone()]).transpose();
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len()-1{
            current = (&(&self.weights[i]
                 * &current)
                 + &self.biases[i]) 
                .map(self.activation.get_function().function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn back_propegate(&mut self, outputs: Vec<f32>, targets: Vec<f32>, mode: &Mode, get_loss: bool) -> Vec<f32>{
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();

        let mut layer_loss: Vec<f32> = vec![]; 
        if get_loss{
            layer_loss = match mode{
                Mode::Min => vec![f32::MAX; self.data.len()],
                Mode::Max => vec![f32::MIN; self.data.len()],
                Mode::Avg => vec![0.0; self.data.len()],
            };
        }
        let mut errors = &Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.get_function().derivative);
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = &self.weights[i] + &(&gradients * (&self.data[i].transpose()));

            self.biases[i] = &self.biases[i] + &gradients;
            errors = &self.weights[i].transpose() * (&errors);

            gradients = self.data[i].map(self.activation.get_function().derivative);
            if get_loss {
                let loses = &errors.transpose().data[0];
                layer_loss[i] = match mode{
                    Mode::Min => loses.iter().fold(f32::INFINITY, |prev, &post| prev.min(post)),
                    Mode::Max => loses.iter().fold(f32::MIN, |prev, &post| prev.max(post)),
                    Mode::Avg => loses.iter().sum::<f32>() / loses.len() as f32
                };
            }
        }
        layer_loss
    }
    ///Does back propegation at every layer and caches the net loss found at each layer
    pub fn get_layer_loss(&mut self, outputs: Vec<f32>, targets: Vec<f32>, mode: &Mode) -> Vec<f32> {
        let mut response: Vec<f32> = vec![0.0; self.layers.len()-2];
        
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = &Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.get_function().derivative);

        let mut weights: Vec<Matrix> = self.weights.clone();
        let mut biases: Vec<Matrix> = self.biases.clone();
        let mut data: Vec<Matrix> = self.data.clone();

        for i in (1..self.layers.len()-1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);

            errors = &weights[i].transpose() * (&errors);
            weights[i] = &weights[i] + &(&gradients * (&self.data[i].transpose()));
            biases[i] = &biases[i] + &gradients;
            gradients = data[i].map(self.activation.get_function().derivative);

            let loses = &errors.transpose().data[0];
            let val = match mode{
                Mode::Min => loses.iter().fold(f32::INFINITY, |prev, &post| prev.min(post)),
                Mode::Max => loses.iter().fold(f32::MIN, |prev, &post| prev.max(post)),
                Mode::Avg => loses.iter().sum::<f32>() / loses.len() as f32
            };

            response[i-1] = val;
        }

        response
    }
    ///Performs back propegation only on a specified layer and the layers around it. This is
    ///especially important for 'catching up' when a new layer is dynamically added.
    pub fn back_propegate_one_layer(&mut self, outputs: Vec<f32>, targets: Vec<f32>, column_target: usize){
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        if column_target <= 1 && self.layers.len()-2 >= column_target{
            panic!("Target column out of bounds or illegal column choice (input row or output row)");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = &Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.get_function().derivative);
        for i in (0..self.layers.len()-1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);

            errors = &self.weights[i].transpose() * (&errors);
            if i == column_target || i == column_target-1 || i == column_target+1 {
                self.weights[i] = &self.weights[i] + &(&gradients * (&self.data[i].transpose()));

                self.biases[i] = &self.biases[i] + &gradients;
            }
            if i == column_target -1 {
                return;
            }
            gradients = self.data[i].map(self.activation.get_function().derivative);
        }
    }
    pub fn back_propegate_one_layer_removal(&mut self, outputs: Vec<f32>, targets: Vec<f32>, column_target: usize){
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        if column_target < 1 && self.layers.len() >= column_target{
            panic!("Target column out of bounds or illegal column choice (input row or output row)");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = &Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.get_function().derivative);
        for i in (0..self.layers.len()-1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);

            errors = &self.weights[i].transpose() * (&errors);
            if i == column_target {
                self.weights[i] = &self.weights[i] + &(&gradients * (&self.data[i].transpose()));

                self.biases[i] = &self.biases[i] + &gradients;
                return; 
            }
            gradients = self.data[i].map(self.activation.get_function().derivative);
        }
    }
    pub fn train(&mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, epochs: usize, mode: &Mode) -> (f32, Vec<f32>) {
        let mut accuracy: f32 = match mode {
            Mode::Min => f32::MAX,
            Mode::Max => f32::MIN,
            Mode::Avg => 0.0
        };
        let mut layer_loss: Vec<f32> = match mode{
            Mode::Min => vec![f32::MAX; self.layers.len()],
            Mode::Max => vec![f32::MIN; self.layers.len()],
            Mode::Avg => vec![0.0; self.layers.len()]
        };
        let mut layer_loss_avg: Vec<f32> = match mode{
            Mode::Min => vec![f32::MAX; self.layers.len()],
            Mode::Max => vec![f32::MIN; self.layers.len()],
            Mode::Avg => vec![0.0; self.layers.len()]
        };

        for i in 1..=epochs{
            let mut inner_accuracy: Vec<f32> = vec![];
            inputs.iter().enumerate().for_each(|(j, _)| {
                let outputs = self.feed_forward(&inputs[j]);
                let mut running_num:f32 = 0.0;
                if i == epochs{
                    for i in 0..outputs.len() {
                        let accuracy_at_node:f32 = (targets[j][i] - outputs[i]).abs();//(1.0 + (&targets[place][i] - resp[i])).abs() / (1.0 + targets[place][i]).abs(); 

                        inner_accuracy.push(accuracy_at_node);
                        running_num = match mode{
                            Mode::Avg => {
                                if i == 0 {
                                    accuracy_at_node
                                }else{
                                    ((running_num * (i as f32)) + accuracy_at_node) / i as f32
                                }

                            },
                            Mode::Min => running_num,
                            Mode::Max => running_num
                        };
                    }
                    accuracy = match mode {
                        Mode::Avg => {
                            if j == 0 {
                                running_num
                            }else{
                                ((accuracy * (j as f32)) + running_num) / j as f32
                            }
                        },
                        Mode::Min => f32::min(accuracy, running_num),
                        Mode::Max => f32::max(accuracy, running_num)
                    };
                    
                }
                if i == epochs{
                    layer_loss = self.back_propegate(outputs, targets[j].clone(), mode, true);
                    for layer in 1..layer_loss.len()-1{
                        layer_loss_avg[layer] = match mode{
                            Mode::Avg => {
                                if j == 0 {
                                    layer_loss[layer]
                                }else{
                                    ((layer_loss_avg[layer] * (j as f32)) + layer_loss[layer]) / j as f32 + 1.0
                                }
                            }
                            Mode::Min => f32::min(layer_loss_avg[layer], layer_loss[layer]),
                            Mode::Max => f32::max(layer_loss_avg[layer], layer_loss[layer])
                        };
                    }
                } else{
                    self.back_propegate(outputs, targets[j].clone(), mode, false);
                }

            });
        }
        /*let val:f32 = match mode {
            Mode::Min => accuracies.iter().fold(f32::MAX, |prev, &post| prev.min(post)),
            Mode::Max => accuracies.iter().fold(f32::MIN, |prev, &post| prev.max(post)),
            Mode::Avg => accuracies.iter().sum::<f32>() / accuracies.len() as f32
        };*/
        (accuracy, layer_loss_avg)
    }
    pub fn train_one_layer_removal(&mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, epochs: usize, layer: usize) {
        for i in 1..=epochs{
            for j in 0..inputs.len(){
                let outputs = self.feed_forward(&inputs[j]);
                self.back_propegate_one_layer_removal(outputs, targets[j].clone(), layer);
            }
        }
    }
    pub fn train_one_layer(&mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, epochs: usize, layer: usize) {
        for i in 1..=epochs{
            for j in 0..inputs.len(){
                let outputs = self.feed_forward(&inputs[j]);
                self.back_propegate_one_layer(outputs, targets[j].clone(), layer);
            }
        }
    }
    pub fn get_loss(&mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, mode: &Mode) -> (Vec<f32>, f32){
        let mut accuracies: Vec<f32> = vec![];
        inputs.iter().enumerate().for_each(|(place, input)| {
            let mut inner_accuracy: Vec<f32> = vec![];
            let resp: Vec<f32> = self.feed_forward(input);
            for i in 0..resp.len() {
                let accuracy_at_node:f32 = (&targets[place][i] - resp[i]).abs();//(1.0 + (&targets[place][i] - resp[i])).abs() / (1.0 + targets[place][i]).abs(); 
                inner_accuracy.push(accuracy_at_node);
            }
            accuracies.push(inner_accuracy.iter().sum::<f32>() / inner_accuracy.len() as f32);
        });
        let val = match mode{
            Mode::Min => accuracies.iter().fold(f32::MAX, |prev, &post| prev.min(post)),
            Mode::Max => accuracies.iter().fold(f32::MIN, |prev, &post| prev.max(post)),
            Mode::Avg => accuracies.iter().sum::<f32>() / accuracies.len() as f32
        };
        (accuracies, val)
    }
    ///Takes a current model and repeatedly trains and mutates upon a dataset until a certain loss
    ///threshold has been met. Layers will be added and removed very conservatively as to prevent
    ///overfitting
    ///
    ///# Arguments
    ///
    ///* `inputs` - A vector of all inputs
    ///* `outputs` - A vector of the matching outputs
    ///* `desired_loss` - The loss desired for the model to reach
    ///* `steps_per` - How many epochs the model trains for between evaluating loss
    ///* `accuracy mode` - Whether you want the Avg, Min, or Max error derived during analysis to
    ///be used as loss model
    ///* `loss_threshold` - The offness of loss between current and needed loss to actually develop
    ///a new layer
    ///* `kill_thresh` - The threshold between the current and previous loss needed to determine
    ///that the last layer was blighted and needs to be removed
    ///* `min` - The minimum amount of neurons added to a layer
    ///* `max` - The maximum amount of neurons added to a layer
    ///# Example
    ///
    ///```
    ///fn main() {
    ///    let inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    ///    let outputs: Vec<Vec<f32>> = vec![vec![0.0],vec![1.0],vec![1.0],vec![0.0]];
    ///    let mut new_net: Network = Network::new(vec![2,3,1], Activations::SIGMOID, 0.1);
    ///    let mut new_net: Network = Network::load("/root/source/rust/triton/save/net.json");
    ///    new_net = new_net.train_to_loss(inputs, outputs, 0.00005, 50000, Mode::Avg, 0.1, 0.0001, 3, 10);
    ///    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0])[0]);
    ///    println!("0 and 1: {:?}", new_net.feed_forward(&vec![0.0,1.0])[0]);
    ///    println!("1 and 1: {:?}", new_net.feed_forward(&vec![1.0,1.0])[0]);
    ///    println!("0 and 0: {:?}", new_net.feed_forward(&vec![0.0,0.0])[0]);
    ///    println!("New network made: {:?}", new_net.layers);
    ///}
    ///```
    pub fn train_to_loss(mut self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, desired_loss: f32, steps_per: usize, accuracy_mode: Mode, loss_threshold: f32, kill_thresh: f32, min: usize, max: usize) -> Network{
        let mut rng = rand::thread_rng();
        let mut loss: f32 = 1.0;
        let mut loss_cache: f32 = f32::MAX;
        let mut most_recent_pos: usize = 0;
        let mut layer_loss: Vec<f32>;
        let mut total_steps_taken: usize = 0;
        while loss > desired_loss {

            //Train model for [steps_per] steps, then analyze accuracy
            (loss, layer_loss) = self.train(&inputs, &targets, steps_per, &accuracy_mode);
            total_steps_taken += steps_per;
            println!("{}", loss);
            //let new_accuracy = self.get_loss(inputs.clone(), targets.clone(), &accuracy_mode);

            /*let mut layer_loss: Vec<Vec<f32>> = vec![];
            for i in 0..inputs.len(){
                let resp = self.feed_forward(&inputs[i]);
                let targ = targets[i].clone();
                let input_accuracy = self.get_layer_loss(resp, targ, &accuracy_mode);
                layer_loss.push(input_accuracy);
            }*/
            //let max_std_dev = std_dev_per_layer.iter().enumerate().fold(f64::MIN, |prev, (_, &post)| prev.max(post));
            if (loss > loss_cache || loss_cache - loss <= kill_thresh) && most_recent_pos != 0 && self.layers.len() > 5 {
                println!("Removing layer at {}", most_recent_pos);
                let layer_len = self.layers[most_recent_pos];
                
                self.data.remove(most_recent_pos);
                self.data.remove(most_recent_pos);
                self.data.remove(most_recent_pos);

                self.layers.remove(most_recent_pos);
                self.layers.remove(most_recent_pos);
                self.layers.remove(most_recent_pos);

                self.weights.remove(most_recent_pos);
                self.weights.remove(most_recent_pos);
                self.weights.remove(most_recent_pos);

                self.biases.remove(most_recent_pos);
                self.biases.remove(most_recent_pos);
                self.biases.remove(most_recent_pos);

                self.insert_at(most_recent_pos, layer_len);

                self.train_one_layer_removal(&inputs, &targets, total_steps_taken, most_recent_pos);
                most_recent_pos = 0;
            }

            if loss > desired_loss && (loss - loss_cache).abs() >= loss_threshold {
                let max_loss = layer_loss.iter()
                    .enumerate()
                    .fold((0, f32::MIN), |(max_index, max_val), (i, &post)| {
                        if post > max_val {
                            (i, post)
                        } else {
                            (max_index, max_val)
                        }
                    });
                if max_loss.1 > loss_threshold {
                    //Mutate self
                    let pos = max_loss.0;
                    most_recent_pos = pos;
                    println!("Add a new layer at index {}", pos);
                    let mut new_net = Network::from(self.clone(), pos, rng.gen_range(min..=max));
                    new_net.train_one_layer(&inputs, &targets,total_steps_taken, pos+1);
                    self = new_net;
                }
            }
            //accuracy_cache.push(new_accuracy.1);
            loss_cache = loss;
        }
        println!("Done in {} epochs", total_steps_taken);
        self.loss = loss_cache;
        self
    }
    
    //Save and Load functions
    pub fn save(&self, path: &'a str) {
        let mut file = File::create(path).expect("Unable to hit save file :(");
        let file_ser = to_string(self).expect("Unable to serialize network :(((");
        file.write_all(file_ser.to_string().as_bytes()).expect("Write failed :(");
    }
    pub fn load(path: &'a str) -> Network{
        let mut buffer = String::new();
        let mut file = File::open(path).expect("Unable to read file :(");

        file.read_to_string(&mut buffer).expect("Unable to read file but even sadder :(");

        let net: Network = from_str(&buffer).expect("Json was not formatted well >:(");
        net
    }
}
