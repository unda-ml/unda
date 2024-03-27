# Contributing to unda

Thanks for taking the time to contribute! :tada::+1:

As a singular college student working on this crate for fun to better understand rust and machine learning it really does mean a lot :)

## Development

### Running code locally

To begin, please follow the same instructions listed in the README to ensure XLA is properly installed and sourced on your machine.

1) Identify the [latest compatible versions of CUDA and cuDNN](https://www.tensorflow.org/install/source#gpu). Adapt [these instructions](https://medium.com/@gokul.a.krishnan/how-to-install-cuda-cudnn-and-tensorflow-on-ubuntu-22-04-2023-20fdfdb96907) to install the two version of CUDA and cuDNN together.

2) Install `clang` and `libclang1`.

3) Download and extract [xla_extension](https://github.com/elixir-nx/xla/releases/tag/v0.6.0).

4) Make sure `LD_LIBRARY_PATH` includes `/path/to/xla_extension/lib`, and make sure the relevant CUDA paths are also visible to the system.

Assuming you downloaded unda to `/home/user/unda` and you want to run
the crate locally, it's relatively self explanatory as you would run any rust project:
```bash
cd /home/user/unda
cargo build --release
./target/release/unda
```
or run any of the example programs with
```bash
cd /home/user/unda
cargo run --release --example {example_name}
```
Once changes are made, please make a new branch and submit a PR :)

### Running tests

Nothing special, just `cargo test`.
