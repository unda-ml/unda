 <img align="left" src="./triton-logo.svg" width="80px" height="80px" alt="triton mascot icon">

# triton

### A self sustaining growing neural net that can repair itself until reaching a desired accuracy

[![crates.io](https://img.shields.io/crates/v/triton_grow.svg)](https://crates.io/crates/triton_grow)
[![Documentation](https://docs.rs/triton_grow/badge.svg)](https://docs.rs/triton_grow)

Triton aims to bring the future of deep learning to the world of rust. With dynamic input traits, concurrent minibatch processing, and full Dense network support(with convolutions soon to come), Triton is quickly emerging and making neural network development easy and robust.

Using the built in **Input** trait, practically any data type can be mapped to an input for a neural network without the need for cutting corners, and the inner trait for layers allows for a plug and play style to neural network development. Currently, Triton has full support for Dense layers, Adam Optimization for Backprop, Activation functions (Sigmoid, TanH, ReLU and LeakyReLU), and even loss analysis per model and per layer. 

Gradient descent currently can happen both syncronously as stochastic gradient descent or asynchronously through minibatch gradient descent. 

One feature in development is that of self growing systems, allowing the neural network to analyze the loss of every layer and algorithmically deduce where the best place to splice in a new layer of a certain length would be. This feature was finalized in an earlier version of Triton, but is currently unavailable with the new rewrite currently taking place. Self growing neural networks is the main goal of the Triton crate, and is currently one of the highest priorities in development.

Currently, the other features in development for Triton are as follows: Convolutional layers(Forward is finished, working on backprop now) and self growth systems. The future of Triton is unknown, but the goal would be to implement more layer types, with Recurrent layers likely being next and GAN support being a pipe dream for far into the future.

## Installation

Use the package manager [cargo](https://crates.io/) to add [triton](https://crates.io/crates/triton_grow) to your rust project.

```bash
cargo add triton_grow
```

or add the dependency directly in your **cargo.toml** file

```toml
[dependencies]
triton_grow = "{version}"
```
## Usage

### Dense Network
```rust
use triton_grow::network::{network::Network, activations::Activations, layer::layers::LayerTypes, input::Input};

fn main() {
    let inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0], vec![1.0,1.0]];
    let outputs = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new(4);

    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.1));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.1));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.1));

    new_net.compile();

    new_net.fit(&inputs, &outputs, 40);

    //let mut new_net = Network::load("best_network.json");
    println!("1 and 0: {:?}", new_net.predict(vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(vec![0.0,0.0])[0]);

    new_net.save("best_network.json");
}
```

### Data Visualization

Using the triton_grow::helper::data_vis extension, you can use the plotters library to visualize aspects of your neural network!

Currently the following visualizations exist:

- Loss history
- Error per layer

```rust
use std::error::Error;

use triton_grow::network::{network::Network, activations::Activations, layer::layers::LayerTypes, input::Input};
use triton_grow::helper::data_vis;

fn main() -> Result<(), Box<dyn Error>> {
    let inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0], vec![1.0,1.0]];
    let outputs = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new(4);

    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.1));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.1));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.1));

    new_net.compile();

    new_net.fit(&inputs, &outputs, 40);

    new_net.plot_loss_history("loss_history.png")?;
    new_net.plot_layer_loss("layer_loss.png")?;
    Ok(())
}
```

## TODO

Currently, triton is in a very beta stage, the following features are still in development:

[Growth Goals]
 - [ ]  Mutating a neural network
    - [ ]  Adding a new layer with ```n``` neurons into any point of an existent network
    - [ ]  Removing a layer from an existent network **!!IN PROGRESS!!**
- [ ]  Back propegation only affecting a single column (allows for a newly added layer to 'catch up')
- [X]  *Analysis* mode during back propegation allowing for all individual errors to be recorded
- [ ]  Updated training function
    - [ ]  Input desired success rate
    - [ ]  Dynamic error analysis to allow for choosing if the network should grow or shrink
    - [ ]  Acceptable threshold of +/- in the errors to allow for a less punishing learning process especially when a new neuron layer has been added
- [X]  Model serialization (serde)
- [ ] Accelerated matrix multiplication (Rayon or Cuda, or BOTH)

[Neural Network Goals]
- [X] Create abstract representation for layers (Layer trait)
    - [X] Dense
    - [ ] Convolutional
    - [ ] Recurrent
- [X] Allow for different activation functions and learning rates on each layer
- [X] Adam Optimization in backprop
- [X] Helper Function for parsing CSV data
- [X] Helper Function for generating the MNIST dataset
- [X] Helper Functions for generating and deriving categorical data

##### If open source development is your thing, we at Triton would love additional work on anything that can be implemented, please contact **eversonb@msoe.edu** if you'd like to help out!

# License
Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
