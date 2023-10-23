# triton 🦎

### A self sustaining growing neural net that can repair itself until reaching a desired accuracy


## Installation

Use the package manager [cargo](https://crates.io/) to add triton to your rust project.

```bash
cargo add triton_grow
```

or add the dependency directly in your **cargo.toml** file

```toml
[dependencies]
triton_grow = "{version}"
```

## Usage

Triton acts as a typical neural network implementation, but allows for a more dynamic way of solving problems you may not know how to solve. Acting as a 'brute force' approach to the world of deep learning, after ```n``` epochs in the training process triton will evaluate the specific error of each neuron and column, deciding whether to add a neuron to a column, add a new column entirely, remove a neuron or remove a column. 

Triton will train and grow a desirable neural network until a specific accuracy is matched, returning the finished model

```rust
use triton_grow::network::{network::Network, activations};

async fn main() -> Result<(),Error>{
    let new_net = Network::new(vec![1,2,3,2], activations::SIGMOID, 0.1);

    let newer_net = Network::from(&new_net, 2, 4);
}
```

## TODO

Currently, triton is in a very beta stage, the following features are still in development:

 - [ ]  Mutating a neural network (1/4)
    - [X]  Adding a new layer with ```n``` neurons into any point of an existent network
    - [ ]  Removing a layer from an existent network
    - [ ]  Adding a single neuron to a layer
    - [ ]  Removing a single neuron from a layer
- [X]  Back propegation only affecting a single column (allows for a newly added layer to 'catch up')
- [X]  *Analysis* mode during back propegation allowing for all individual errors to be recorded
- [ ]  Updated training function
    - [X]  Input desired success rate
    - [X]  Dynamic error analysis to allow for choosing if the network should grow or shrink
    - [X]  Acceptable threshold of +/- in the errors to allow for a less punishing learning process especially when a new neuron layer has been added
- [ ]  Model serialization (serde)

## License

[MIT](https://choosealicense.com/licenses/mit/)
