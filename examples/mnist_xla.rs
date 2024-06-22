use std::fs::File;
use std::io;
use std::os::unix::fs::*;
use std::time::Instant;
use unda::graph::*;
use xla::{ElementType::*, PjRtClient, PjRtLoadedExecutable};

const USE_CPU: bool = false;
const MEM_FRAC: f64 = 0.9;
const MNIST_DIRECTORY: &str = "/home/ekadile/mnist";
const EPOCHS: usize = 100;
const INIT_LEARNING_RATE: f32 = 1e-3;
const LEARNING_RATE_DECAY: f32 = 0.95;
const MIN_LEARNING_RATE: f32 = 1e-5;

// ABSTRACT API REQUIREMENT 1: Automatic Layer Construction
// We should have functions like this which, for a given layer type,
// automatically resolve shapes and dtypes and construct nodes for
// the parameters and outputs of a layer. In the final version,
// a function like this should also take an "initialization" parameter
// and run random initialization for the weights and bias using XLA's
// random number generation functions.
fn dense(
    model: &mut Context,
    input_node: NodeIdentifier,
    out_size: u32,
    name: &str,
) -> Result<(NodeIdentifier, (NodeIdentifier, NodeIdentifier))> {
    let shape = model.nodes[input_node].shape.clone();
    let last_dim = shape.sizes[shape.ndims() - 1];
    let dtype = model.nodes[input_node].dtype;

    let weights_shape = Shape::from([last_dim, out_size]);
    let mut weights_name = name.to_owned();
    weights_name.push_str("_weights");
    let weights = model.parameter(weights_name, weights_shape, dtype)?;

    let mut bias_shape = Shape::new();
    for _ in 0..(shape.ndims() - 1) {
        bias_shape.sizes.push(1u32);
    }
    bias_shape.sizes.push(out_size);
    let mut bias_name = name.to_owned();
    bias_name.push_str("_bias");
    let bias = model.parameter(bias_name, bias_shape, dtype)?;

    let matmul_node = model.matmul(input_node, weights)?;
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
    let image_rescaled = model.mul(image_fp, scale)?;

    let sparse_labels = model.parameter("sparse_labels", [100], U8)?;
    let one_hot_labels = model.one_hot(sparse_labels, 10, F32)?;

    //let (d1, (w1, b1)) = dense(&mut model, image_rescaled, 2000, "layer1")?;
    //let d1_activation = model.leaky_relu(d1, 0.03)?;
    //let (d2, (w2, b2)) = dense(&mut model, d1_activation, 256, "layer2")?;
    //let d2_activation = model.leaky_relu(d2, 0.03)?;
    //let (d3, (w3, b3)) = dense(&mut model, d2_activation, 64, "layer3")?;
    //let d3_activation = model.leaky_relu(d3, 0.03)?;
    let (logits, (w_out, b_out)) = dense(&mut model, image_rescaled, 10, "out_layer")?;
    let probabilities = model.softmax(logits)?;
    let loss = model.mean_cross_entropy(probabilities, one_hot_labels)?;
    //println!("{}", model.to_string(loss));
    let accuracy = model.accuracy(probabilities, sparse_labels)?;

    // ABSTRACT API REQUIREMENT 3: Separate forward/backward pass
    // In this construction, the context contains both the forward
    // prediction computations and the backward update computations.
    // There should be a method for extracting ONLY the forward pass,
    // as during inference we do not want to perform the backward computations.
    // Part of this issue should be the implementation of optional
    // gradient clipping on the backward pass.
    //let (w1_grad, b1_grad) = (model.diff(loss, w1)?, model.diff(loss, b1)?);
    //let (w2_grad, b2_grad) = (model.diff(loss, w2)?, model.diff(loss, b2)?);
    //let (w3_grad, b3_grad) = (model.diff(loss, w3)?, model.diff(loss, b3)?);
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
    // optimizers (SGD, RMSProp, Adam), batch normalization (because this layer
    // has its own type of optimizer), and basic learning rate schedules
    // (ExponentialDecay, ReduceLROnPlateau, CosineAnnealing)
    let learning_rate = model.parameter("learning_rate", [], F32)?;
    // simple SGD updates
    //let (w1_update, b1_update) = (
    //    model.mul(learning_rate, w1_grad)?,
    //    model.mul(learning_rate, b1_grad)?,
    //);
    //let (w2_update, b2_update) = (
    //    model.mul(learning_rate, w2_grad)?,
    //    model.mul(learning_rate, b2_grad)?,
    //);
    //let (w3_update, b3_update) = (
    //    model.mul(learning_rate, w3_grad)?,
    //    model.mul(learning_rate, b3_grad)?,
    //);
    let (w_out_update, b_out_update) = (
        model.mul(learning_rate, w_out_grad)?,
        model.mul(learning_rate, b_out_grad)?,
    );
    // apply updates
    //let (w1_new, b1_new) = (model.sub(w1, w1_update)?, model.sub(b1, b1_update)?);
    //let (w2_new, b2_new) = (model.sub(w2, w2_update)?, model.sub(b2, b2_update)?);
    //let (w3_new, b3_new) = (model.sub(w3, w3_update)?, model.sub(b3, b3_update)?);
    let (w_out_new, b_out_new) = (
        model.sub(w_out, w_out_update)?,
        model.sub(b_out, b_out_update)?,
    );

    model.compile(
        "train_step",
        [
            loss, accuracy,
            //w1_new, b1_new,
            //w2_new, b2_new,
            //w3_new, b3_new,
            w_out_new, b_out_new,
        ],
        client,
    )
}

// This relates directly to ABSTRACT API REQUIREMENT 1
fn init_param(shape: Shape) -> xla::Literal {
    let size = shape.size();

    // deterministic version of xavier initialization
    // nobody uses this, but it doesn't matter because MNIST is simple
    let amplitude = 1f32 / ((shape.sizes[0] + shape.sizes[1]) as f32).sqrt();
    let mut vec = Vec::new();
    for i in 0..size {
        if i % 2 == 0 {
            vec.push(amplitude);
        } else {
            vec.push(-amplitude);
        };
    }
    let vec1 = xla::Literal::vec1(vec.as_slice());

    let xla_shape = shape.sizes.iter().map(|d| *d as i64).collect::<Vec<i64>>();
    match vec1.reshape(xla_shape.as_slice()) {
        Ok(x) => x,
        _ => panic!("Failed to reshape initial paramter value!"),
    }
}

// ABSTRACT API REQUIREMENT 5: Parameter structure abstraction
// This relates closely with ABSTRACT API REQUIREMENT 1
// This example works because I know the exact order in which parameters
// are declared in the model context. This becomes insanely hard
// to keep track of as the architecture grows, and the user shouldn't
// have to worry about it.
// I think the simplest way to achieve this (which is akin to how JAX does it)
// would be to have `Model` objects which are called with two Into<Vec<Literal>>
// structures, one for inputs and one for parameters.
fn init_params() -> (
    //xla::Literal,
    //xla::Literal,
    //xla::Literal,
    //xla::Literal,
    //xla::Literal,
    //xla::Literal,
    xla::Literal,
    xla::Literal,
) {
    (
        //init_param(Shape::from([28 * 28, 2000])),
        //init_param(Shape::from([1, 2000])),
        //init_param(Shape::from([784, 256])),
        //init_param(Shape::from([1, 256])),
        //init_param(Shape::from([256, 64])),
        //init_param(Shape::from([1, 64])),
        init_param(Shape::from([784, 10])),
        init_param(Shape::from([1, 10])),
    )
}

// ABSTRACT API REQUIREMENT 6: Data prefetching
// Data input to the training loop should be in the form of an iterator.
// These iterators could be finite or infinite.
// In the finite case, we should support random shuffling.
// In either case, batches of data should be pre-fetched and queued in parallel
// (and potentially preprocessed by the CPU) as the training loop is executing.
fn load_mnist_batch(
    images: &File,
    labels: &File,
    batch_idx: u64,
) -> io::Result<(xla::Literal, xla::Literal)> {
    let mut image_bytes = [0u8; 100 * 28 * 28];
    images.read_exact_at(&mut image_bytes, 8 + 100 * 28 * 28 * batch_idx)?;
    let mut images_xla = xla::Literal::vec1(&image_bytes);
    images_xla = match images_xla.reshape(&[100, 28 * 28]) {
        Ok(x) => x,
        Err(_) => panic!("Failed to reshape MNIST image batch!"),
    };

    let mut label_bytes = [0u8; 100];
    labels.read_exact_at(&mut label_bytes, 8 + 100 * batch_idx)?;
    let labels_xla = xla::Literal::vec1(&label_bytes);

    Ok((images_xla, labels_xla))
}

fn main() {
    let client = if USE_CPU {
        PjRtClient::cpu().expect("Failed to construct CPU client")
    } else {
        PjRtClient::gpu(MEM_FRAC, false).expect("Failed to construct GPU client")
    };

    let mut train_img_path = MNIST_DIRECTORY.to_owned();
    train_img_path.push_str("/train-images-idx3-ubyte");
    let train_images = File::open(train_img_path).expect("Failed to open training image file");

    let mut train_lbl_path = MNIST_DIRECTORY.to_owned();
    train_lbl_path.push_str("/train-labels-idx1-ubyte");
    let train_labels = File::open(train_lbl_path).expect("Failed to open training label file");

    let mut test_img_path = MNIST_DIRECTORY.to_owned();
    test_img_path.push_str("/t10k-images-idx3-ubyte");
    let test_images = File::open(test_img_path).expect("Failed to open training image file");

    let mut test_lbl_path = MNIST_DIRECTORY.to_owned();
    test_lbl_path.push_str("/t10k-labels-idx1-ubyte");
    let test_labels = File::open(test_lbl_path).expect("Failed to open training label file");

    println!("Building model and optimizer . . .");
    let now = Instant::now();
    let executable =
        build_model_and_optimizer(&client).expect("Failed to build model and optimizer");
    println!("Finished build in {:.2?}", now.elapsed());

    let (mut w_out, mut b_out) = init_params();

    println!("Training model . . .");
    let now = Instant::now();
    for epoch in 0..EPOCHS {
        let mut train_accuracy = 0f32;
        let mut train_loss = 0f32;

        for batch_idx in 0..600 {
            let (train_imgs, train_lbls) =
                load_mnist_batch(&train_images, &train_labels, batch_idx)
                    .expect("Failed to load MNIST batch");

            let lr =
                xla::Literal::scalar(MIN_LEARNING_RATE.max(INIT_LEARNING_RATE * (LEARNING_RATE_DECAY.powf(epoch as f32))));

            // This is where ABSTRACT API REQUIREMENT 5 becomes pertinent
            // The user should not have to explicitly reference a dozen parameters like this
            let xla_buffer = executable
                .execute(&[
                    &train_imgs,
                    &train_lbls,
                    //&w1,
                    //&b1,
                    //&w2,
                    //&b2,
                    //&w3,
                    //&b3,
                    &w_out,
                    &b_out,
                    &lr,
                ])
                .expect("Failed to run PjRt executable");

            // This is where ABSTRACT API REQUIREMENT 4 becomes pertinent
            // The user should not have to move all this junk to the host just to get accuracy and loss
            let xla_literal = xla_buffer[0][0]
                .to_literal_sync()
                .expect("Failed to copy buffer to host");
            let mut untupled_literals = xla_literal
                .to_tuple()
                .expect("Failed to untuple XLA literals");

            let loss = untupled_literals[0]
                .to_vec::<f32>()
                .expect("Failed vector conversion of loss")[0];
            train_loss += loss;
            let accuracy = untupled_literals[1]
                .to_vec::<f32>()
                .expect("Failed vector conversion of accuracy")[0];
            train_accuracy += accuracy;

            // This is really very silly. Because model/optimizer are not separate
            // we move the weights to the CPU just to move them back
            b_out = untupled_literals.pop().unwrap();
            w_out = untupled_literals.pop().unwrap();
            //b3 = untupled_literals.pop().unwrap();
            //w3 = untupled_literals.pop().unwrap();
            //b2 = untupled_literals.pop().unwrap();
            //w2 = untupled_literals.pop().unwrap();
            //b1 = untupled_literals.pop().unwrap();
            //w1 = untupled_literals.pop().unwrap();
        }
        println!(
            "Epoch {}: Training loss = {}; Training accuracy = {}",
            epoch,
            train_loss / 600f32,
            train_accuracy / 600f32
        );
    }
    println!("Finished training in {:.2?}", now.elapsed());

    // ABSTRACT API REQUIREMENT 7: Serialization
    // The model is not worth very much if it disappears after our training loop.
    // My main suggestion is to serialize the compute graph using XLA HLO
    // and serialize the paramters using the npz format.

    let mut test_accuracy = 0f32;
    let mut test_loss = 0f32;

    for batch_idx in 0..100 {
        let (test_imgs, test_lbls) = load_mnist_batch(&test_images, &test_labels, batch_idx)
            .expect("Failed to load MNIST batch");

        // GOOFY!!
        // Another consequence of ABSTRACT API REQUIREMENT 4 Not being implemented
        // To prevent the model from training on the testing data, I have to
        // set the learning rate to zero
        let lr = xla::Literal::scalar(0f32);

        let xla_buffer = executable
            .execute(&[
                &test_imgs, &test_lbls, &w_out, &b_out, &lr,
            ])
            .expect("Failed to run PjRt executable");

        // This is where ABSTRACT API REQUIREMENT 4 becomes pertinent
        // The user should not have to move all this junk to the host just to get accuracy and loss
        let xla_literal = xla_buffer[0][0]
            .to_literal_sync()
            .expect("Failed to copy buffer to host");
        let untupled_literals = xla_literal
            .to_tuple()
            .expect("Failed to untuple XLA literals");

        let loss = untupled_literals[0]
            .to_vec::<f32>()
            .expect("Failed vector conversion of loss")[0];
        test_loss += loss;
        let accuracy = untupled_literals[1]
            .to_vec::<f32>()
            .expect("Failed vector conversion of accuracy")[0];
        test_accuracy += accuracy;
    }
    println!(
        "Testing loss = {}; Testing accuracy = {}",
        test_loss / 100f32,
        test_accuracy / 100f32
    );
}
