use super::{matrix::Matrix, activations::Activation, modes::Mode};

#[derive(Clone)]
pub struct Network<'a> {
    pub layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>
}

impl Network<'_>{
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
    pub fn new<'a>(layers: Vec<usize>, activation: Activation<'a>, learning_rate: f64) -> Network{
        let mut net = Network{
            layers: layers,
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
    ///Creates a neural network from one network, but splices in a new column for dynamic growth
    ///
    ///Example:
    ///```
    ///let mut new_net = Network::new(vec![2,4,8,14,1], activation::SIGMOID, 0.1);
    ///let mut newer_net = Network::from(&new_net, 2, 10);
    ///
    ///assert_eq!(newer_net.layers, vec![2,4,10,8,14,1]);
    ///```
    pub fn from<'a>(network_from: &'a Network, newlayer_pos: usize, newlayer_len: usize) -> Network<'a>{
        if !(newlayer_pos > 0 && newlayer_pos < network_from.layers.len()){
            panic!("Attempting to add a new layer in an invalid slot [out of bounds, input layer, output layer]");
        }
        let mut new_net = Network { 
            layers: vec![],
            weights: vec![],
            biases: vec![],
            data: vec![],
            learning_rate: network_from.learning_rate,
            activation: network_from.activation.clone()
        };

        for i in 0..network_from.layers.len(){
            if i == newlayer_pos{
                new_net.layers.push(newlayer_len);
            }
            new_net.layers.push(network_from.layers[i]);
        }
        
        for i in 0..new_net.layers.len() - 1 {
            new_net.weights.push(Matrix::new_random(new_net.layers[i+1],new_net.layers[i]));
        }

        new_net
    }
    pub fn feed_to_point(&mut self, inputs: &Vec<f64>, layer_to: usize) -> Vec<f64>{
         if inputs.len() != self.layers[0]{
             panic!("Invalid numer of inputs");
         }
         if layer_to >= self.layers.len(){
             panic!("To destination is larger than network size");
         }
         let mut current = Matrix::from(vec![inputs.clone()]).transpose();
         self.data = vec![current.clone()];

         for i in 0..layer_to{
             current = ((self.weights[i].clone()
                        * &current)
                        + &self.biases[i])
                        .map(self.activation.function);
             self.data.push(current.clone());
         }
         current.transpose().data[0].to_owned()
    }
    pub fn point_to_feed(&mut self, inputs_at_point: &Vec<f64>, layer_at: usize) -> Vec<f64> {
        if inputs_at_point.len() != self.layers[layer_at]{
            panic!("Invalid number of inputs for layer");
        }
        let mut current = Matrix::from(vec![inputs_at_point.clone()]).transpose();
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len()-1{
            current = ((self.weights[i].clone()
                        * &current)
                        + &self.biases[i])
                        .map(self.activation.function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn feed_forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs");
        }
         
        let mut current = Matrix::from(vec![inputs.clone()]).transpose();
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len()-1{
            current = ((self.weights[i].clone()
                 * &current)
                 + &self.biases[i]) 
                .map(self.activation.function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn back_propegate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> Option<Vec<f64>> {
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();
        //println!("{} {}",parsed.rows, parsed.columns);
        
        let mut errors = Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.derivative);
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i].clone() + &(gradients.clone() * (&self.data[i].transpose()));

            self.biases[i] = self.biases[i].clone() + &gradients;
            errors = self.weights[i].transpose() * (&errors);

            gradients = self.data[i].map(self.activation.derivative);
        }
        None
    }
    pub fn back_propegate_one_layer(&mut self, outputs: Vec<f64>, targets: Vec<f64>, column_target: usize){
        //TODO: Allow for back propegation to only happen to the newly added column. We can achieve
        //this by:
        //1. Move backwards in network until we reach the desired columns
        //2. Adjust weights and biases for this layer only
        //3. Break
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        if !(column_target <= 1 && self.layers.len()-2 <= column_target){
            panic!("Target column out of bounds or illegal column choice (input row or output row)");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.derivative);
        for i in (0..self.layers.len()-1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);

            errors = self.weights[i].transpose() * (&errors);
            if i == column_target {
                self.weights[i] = self.weights[i].clone() + &(gradients.clone() * (&self.data[i].transpose()));

                self.biases[i] = self.biases[i].clone() + &gradients;
 
            }
            gradients = self.data[i].map(self.activation.derivative);
        }
    }
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for i in 1..=epochs{
            if epochs < 1000 || i % (epochs/1000) == 0 {
                //println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len(){
                let outputs = self.feed_forward(&inputs[j]);
                self.back_propegate(outputs, targets[j].clone());
            }
        }
    }
    pub fn get_loss(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, mode: &Mode) -> f64{
        let mut accuracies: Vec<f64> = vec![];
        inputs.iter().enumerate().for_each(|(place, input)| {
            let mut inner_accuracy: Vec<f64> = vec![];
            let resp: Vec<f64> = self.feed_forward(input);
            for i in 0..resp.len() {
                let accuracy_at_node:f64 = (&targets[place][i] - resp[i]).abs();//(1.0 + (&targets[place][i] - resp[i])).abs() / (1.0 + targets[place][i]).abs(); 
                inner_accuracy.push(accuracy_at_node);
            }
            accuracies.push(inner_accuracy.iter().sum::<f64>() / inner_accuracy.len() as f64);
        });
        return match mode{
            Mode::Min => accuracies.iter().fold(f64::INFINITY, |prev, &post| prev.min(post)),
            Mode::Max => accuracies.iter().fold(f64::INFINITY, |prev, &post| prev.max(post)),
            Mode::Avg => accuracies.iter().sum::<f64>() / accuracies.len() as f64
        };
    }
    pub fn add_row(&mut self, pos: usize){
        
    }
    pub fn train_to_loss(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, desired_loss: f64, steps_per: usize, accuracy_mode: Mode){
        let mut accuracy_cache: Vec<f64> = vec![1.0];
        let mut total_steps_taken: usize = 0;
        while accuracy_cache[accuracy_cache.len()-1] > desired_loss {
            //Train model for [steps_per] steps, then analyze accuracy
            self.train(inputs.clone(), targets.clone(), steps_per);
            total_steps_taken += steps_per;
            let new_accuracy = self.get_loss(inputs.clone(), targets.clone(), &accuracy_mode);
            /*if new_accuracy < desired_accuracy {
                //Mutate self
            }*/
            accuracy_cache.push(new_accuracy);
        }
        println!("Done in {} epochs", total_steps_taken);
    }
}
