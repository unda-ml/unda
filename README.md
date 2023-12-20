# triton 🦎

### A self sustaining growing neural net that can repair itself until reaching a desired accuracy


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

Triton acts as a typical neural network implementation, but allows for a more dynamic way of solving problems you may not know how to solve. Acting as a 'brute force' approach to the world of deep learning, after ```n``` epochs in the training process triton will evaluate the specific error of each neuron and column, deciding whether to add a neuron to a column, add a new column entirely, remove a neuron or remove a column. 

Triton will train and grow a desirable neural network until a specific accuracy is matched, returning the finished model

```rust
use triton_grow::network::{network::Network, activations::Activations, modes::Mode, layer::layers::LayerTypes};

fn main() {
    let inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0], vec![1.0,1.0]];
    let outputs: Vec<Vec<f32>> = vec![vec![0.0],vec![1.0],vec![1.0], vec![0.0]];

    let mut new_net = Network::new();

    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(2, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.01));
    new_net.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.01));

    new_net.compile();

    new_net.fit(inputs, outputs, 100);

    //let mut new_net = Network::load("best_network.json");
    
    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.feed_forward(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.feed_forward(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.feed_forward(&vec![0.0,0.0])[0]);
    new_net.save("best_network.json");
}
```
## Proven Results

Upon testing Triton's self growth method against a traditional preconfigured network model. Three neural networks were all tasked with learning a simple **XOR predictor** with the following inputs and outputs:

### Inputs
```
[ 1.0 , 0.0 ]
[ 0.0 , 1.0 ]
[ 0.0 , 0.0 ]
[ 1.0 , 1.0 ]
```

### Outputs
```
[ 1.0 ]
[ 1.0 ]
[ 0.0 ]
[ 0.0 ]
```

### Testing

| Model Name    | Layers {input -[hidden] - output} | Epochs Needed to Get 0.001 Avg Loss |
| ------------- | ------------- | ------------- |
| Minimum  | 2 - { *3* } - 1  |  7,880,000 |
| Well Fit  | 2 - { *3 - 4 - 3* } - 1 | 2,790,000  |
| Triton  | 2 - { *self growing* } - 1 | 100,000  |

Triton was 98.09% more efficient than the minimum fit model, and 94.62% more than even the well fit model.


## TODO

Currently, triton is in a very beta stage, the following features are still in development:

 - [X]  Mutating a neural network (2/2)
    - [X]  Adding a new layer with ```n``` neurons into any point of an existent network
    - [X]  Removing a layer from an existent network **!!IN PROGRESS!!**
- [X]  Back propegation only affecting a single column (allows for a newly added layer to 'catch up')
- [X]  *Analysis* mode during back propegation allowing for all individual errors to be recorded
- [X]  Updated training function
    - [X]  Input desired success rate
    - [X]  Dynamic error analysis to allow for choosing if the network should grow or shrink
    - [X]  Acceptable threshold of +/- in the errors to allow for a less punishing learning process especially when a new neuron layer has been added
- [X]  Model serialization (serde)
- [ ] Accelerated matrix multiplication (Rayon or Cuda, or BOTH)

## License

[MIT](https://choosealicense.com/licenses/mit/)
