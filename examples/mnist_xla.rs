use unda::core::graph::*;
use xla::{ElementType::*, PjRtLoadedExecutable};
use std::io;
use std::fs::File;
use std::os::unix::fs::*;
use byteorder::{LittleEndian, ReadBytesExt};

// ABSTRACT API REQUIREMENT 1: Automatic Layer Construction
// We should have functions like this which, for a given layer type,
// automatically resolve shapes and dtypes and construct nodes for
// the parameters and outputs of a layer. In the final version,
// a function like this should also take an "initialization" parameter
// and run random initialization for the weights and bias.
fn dense(
    model: &mut Context,
    input_node: NodeIdentifier,
    out_size: u32,
    name: &str,
) -> Result<(NodeIdentifier, (NodeIdentifier, NodeIdentifier))> {
    let shape = model.nodes[input_node].shape;
    let last_dim = shape.sizes[shape.ndims() - 1];
    let dtype = model.nodes[input_node].dtype;

    let weights_shape = Shape::from([out_size, last_dim]);
    let mut weights_name = name.to_owned();
    weights_name.push_str("_weights");
    let weights = model.parameter(weights_name, weights_shape, dtype)?;

    let mut bias_shape = Shape::new();
    bias_shape.sizes.push(out_size);
    for i in 0..(shape.ndims() - 1) {
        bias_shape.sizes.push(1u32);
    }
    let mut bias_name = name.to_owned();
    bias_name.push_str("_bias");
    let bias = model.parameter(bias_name, bias_shape, dtype)?;

    let matmul_node = model.matmul(weights, input_node)?;
    let dense_node = model.add(matmul_node, bias)?;

    Ok((dense_node, (weights, bias)))
}

fn build_model_and_optimizer(client: &xla::PjRtClient) -> Result<PjRtLoadedExecutable> {
    let mut model = Context::new();

    // ABSTRACT API REQUIREMENT 2: Dynamic Batching
    // In this example, the batch size is hardcoded to 100.
    // This is fine because MNIST has exactly 60K training
    // and 10K testing examples. It should not be generally
    // assumed that the batch size divides the dataset size.
    // Abstract model objects must be optimized for a specific
    // batch size but be willing to take any. One simple way to
    // achieve this would be simply having constant batch size
    // but masking (via multiplication with a binary vector)
    // the loss on "empty" examples.
    let image_input = model.parameter("image_input", [100, 28 * 28], U8)?;
    let image_fp = model.type_cast(image_input, F32);
    // MNIST bytes range from 0 to 255, neural network only wants to see 0 to 1
    let scale = model.scalar(1f32 / 255f32, F32)?;
    let image_rescaled = model.div(image_fp, scale)?;

    let sparse_labels = model.parameter("sparse_labels", [100], S64)?;
    let one_hot_labels = model.one_hot(sparse_labels, 10, F32)?;

    let (d1, (w1, b1)) = dense(&mut model, image_rescaled, 784, "layer1")?;
    let d1_activation = model.relu(d1)?;
    let (d2, (w2, b2)) = dense(&mut model, d1_activation, 256, "layer2")?;
    let d2_activation = model.relu(d2)?;
    let (d3, (w3, b3)) = dense(&mut model, d2_activation, 64, "layer3")?;
    let d3_activation = model.relu(d3)?;
    let (logits, (w_out, b_out)) = dense(&mut model, d3_activation, 10, "out_layer")?;
    let probabilities = model.softmax(logits)?;
    let loss = model.mean_cross_entropy(probabilities, one_hot_labels)?;
    let accuracy = model.accuracy(probabilities, sparse_labels)?;

    // ABSTRACT API REQUIREMENT 3: Separate forward/backward pass
    // In this construction, the context contains both the forward
    // prediction computations and the backward update computations.
    // There should be a method for extracting ONLY the forward pass,
    // as during inference we do not want to perform the backward computations.
    // Part of this issue should be the implementation of optional
    // gradient clipping on the backward pass.
    let (w1_grad, b1_grad) = (model.diff(loss, w1)?, model.diff(loss, b1)?);
    let (w2_grad, b2_grad) = (model.diff(loss, w2)?, model.diff(loss, b2)?);
    let (w3_grad, b3_grad) = (model.diff(loss, w3)?, model.diff(loss, b3)?);
    let (w_out_grad, b_out_grad) = (model.diff(loss, w_out)?, model.diff(loss, b_out)?);

    // ABSTRACT API REQUIREMENT 4: Separate model/optimizer step
    // In general, models and optimizers are thought of as separate
    // objects, so should be separate in principle. Additionally,
    // with large-scale models we want to be able to compute the
    // gradients and then IN PARALLEL 1) compute parameter updates
    // and 2) bus the next model input to the device.
    // This will require binding XLA operations Send, Recv,
    // and potentially OptimizationBarrier.
    // Part of this issue should be the implementation of conventional
    // optimizers (SGD, RMSProp, Adam), and learning rate schedules
    // (ExponentialDecay, ReduceLROnPlateau, CosineAnnealing)
    let learning_rate = model.parameter("learning_rate", [], F32)?;
    // simple SGD updates
    let (w1_update, b1_update) = (
        model.mul(learning_rate, w1_grad)?,
        model.mul(learning_rate, b1_grad)?,
    );
    let (w2_update, b2_update) = (
        model.mul(learning_rate, w2_grad)?,
        model.mul(learning_rate, b2_grad)?,
    );
    let (w3_update, b3_update) = (
        model.mul(learning_rate, w3_grad)?,
        model.mul(learning_rate, b3_grad)?,
    );
    let (w_out_update, b_out_update) = (
        model.mul(learning_rate, w_out_grad)?,
        model.mul(learning_rate, b_out_grad)?,
    );
    // apply updates
    let (w1_new, b1_new ) = (
        model.sub(w1, w1_update)?,
        model.sub(b1, b1_update)?,
    );
    let (w2_new, b2_new ) = (
        model.sub(w2, w2_update)?,
        model.sub(b2, b2_update)?,
    );
    let (w3_new, b3_new ) = (
        model.sub(w3, w3_update)?,
        model.sub(b3, b3_update)?,
    );
    let (w_out_new, b_out_new ) = (
        model.sub(w_out, w_out_update)?,
        model.sub(b_out, b_out_update)?,
    );

    model.compile("train_step",
    [loss, accuracy, w1_new, b1_new, w2_new, b2_new, w3_new, b3_new, w_out_new, b_out_new],
    client)
}

// ABSTRACT API REQUIREMENT 5: Data prefetching
// Data input to the training loop should be in the form of an iterator.
// These iterators could be finite or infinite.
// In the finite case, we should support random shuffling.
// In either case, batches of data should be pre-fetched in parallel
// (and potentially preprocessed by the CPU) as the training loop is executing.
fn load_mnist_batch(images: File, labels: File, batch_idx: u64) -> io::Result<(xla::Literal, xla::Literal)> {
    let mut image_bytes = [0u8; 100*28*28];
    images.read_exact_at(&mut image_bytes, 100*28*28*batch_idx)?;
    let mut images_xla = xla::Literal::vec1(&image_bytes);
    images_xla = match images_xla.reshape(&[100, 28*28]) {
        Ok(x) => x,
        Err(_) => panic!("Failed to reshape MNIST image batch!")
    };

    let mut label_bytes = [0u8; 100*4];
    labels.read_exact_at(&mut label_bytes, 100*4*batch_idx)?;
    let labels_u32 = label_bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    let labels_xla = xla::Literal::vec1(labels_u32.as_slice());

    Ok((images_xla, labels_xla))
}

fn main() {
    println!("Not yet implemented!");
}
