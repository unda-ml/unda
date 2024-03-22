use std::{fs::File, io::{Read, Write, stdout}};


use crate::core::data::{matrix::Matrix, input::Input};

fn read_u32<T: Read>(reader: &mut T) -> u32 {
    let mut buffer = [0; 4];
    reader.read_exact(&mut buffer).expect("Failed to read u32");
    u32::from_be_bytes(buffer)
}

pub struct MnistEntry {
    pub data: Matrix,
    pub label: usize
}

impl MnistEntry {
    pub fn new(data: Vec<f32>, label: usize) -> MnistEntry {
        MnistEntry { data: Matrix::from_sized(data, 28, 28), label }
    }
    pub fn generate_mnist() -> (Vec<Matrix>, Vec<usize>) {
        let mut matrices = vec![];
        let mut labels = vec![];
        
        MnistEntry::load_mnist().into_iter().for_each(|entry| {
            matrices.push(entry.data);
            labels.push(entry.label);
        });
        (matrices, labels)
    }
    fn load_mnist() -> Vec<MnistEntry> {
        let train_image_path = "src/util/mnist/train-images.idx3-ubyte";
        let train_label_path = "src/util/mnist/train-labels.idx1-ubyte";

        let mut res = vec![];
        
        let labels = MnistEntry::read_label_file(train_label_path);
        let images = MnistEntry::read_image_file(train_image_path);

        for i in 0..labels.len(){
            res.push(MnistEntry::new(images[i].clone(), labels[i]));
            //res[i].draw();
            //println!("{}", labels[i]);
        }

        res
    }
    fn read_label_file(file_path: &str) -> Vec<usize> {
        let mut file = File::open(file_path).expect("Failed to open label file");
        let _magic_number = read_u32(&mut file);
        let num_items = read_u32(&mut file);

        let mut labels = Vec::with_capacity(num_items as usize);
        for _ in 0..num_items {
            let mut label_buffer = [0; 1];
            file.read_exact(&mut label_buffer).expect("Failed to read label");
            labels.push(label_buffer[0] as usize);
        }

        labels
    }

    fn read_image_file(file_path: &str) -> Vec<Vec<f32>> {
        let mut file = File::open(file_path).expect("Failed to open image file");
        let _magic_number = read_u32(&mut file);
        let num_items = read_u32(&mut file);
        let num_rows = read_u32(&mut file);
        let num_cols = read_u32(&mut file);

        let count_div = 1000;

        let _ = stdout().flush(); print!("[");

        let mut images = Vec::with_capacity(num_items as usize);
        for i in 0..num_items {
            if i % count_div == 0{
                let _ = stdout().flush(); print!("#");
            }
            let mut image = Vec::with_capacity((num_rows * num_cols) as usize);
            for _ in 0..(num_rows * num_cols) {
                let mut pixel_buffer = [0; 1];
                file.read_exact(&mut pixel_buffer).expect("Failed to read pixel value");
                image.push(pixel_buffer[0] as f32 / 255.0); 
            }
            images.push(image);
        }
        println!("]");

        images
    }
    pub fn draw(&self) {
        for row in 0..28 {
            for col in 0..28 {
                let pixel_value = self.data.data[row][col];
                let symbol = if pixel_value > 0.3 { "#" } else { " " };
                print!("{}", symbol);
            }
            println!();
        }
    }
}

impl Input for MnistEntry {
    fn shape(&self) -> (usize, usize, usize) {
        (self.data.rows, self.data.columns, 1)
    }
    fn to_param(&self) -> Vec<f32> {
        self.data.to_param()
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        self.data.to_param_2d()
    }
    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        self.data.to_param_3d()
    }
    fn to_box(&self) -> Box<dyn Input> {
        self.data.to_box()
    }
}
