use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Missing manifest");
    let xla_extension_dir = PathBuf::from(manifest_dir).join("xla_extension");

    if !xla_extension_dir.exists() {
        panic!("The xla_extension library was not found in the expected path: {}", xla_extension_dir.display());
    }
    println!("cargo:rustc-env=XLA_EXTENSION_DIR={}", xla_extension_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}
