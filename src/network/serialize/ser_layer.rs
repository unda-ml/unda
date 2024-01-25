use crate::network::{layer::{layers::{Layer, LayerTypes}, dense::Dense}, input::Input};

pub struct SerializedLayer {
    pub name: char,
    pub rows: usize,
    pub cols: usize,
    pub weights: String,
    pub bias: String
}

impl SerializedLayer {
    pub fn new(layer: &Box<dyn Layer>, layer_type: &LayerTypes) -> Self {
        let rows = layer.get_weights().to_param_2d().len();
        let cols = layer.get_weights().to_param_2d()[0].len();
        let weights: String = SerializedLayer::flatten_string(&layer.get_weights().to_param_2d());
        let bias = SerializedLayer::flatten_string(&layer.get_biases().to_param_2d());

        Self { name: 'D', rows, cols, weights, bias }
    }
    pub fn from(&self) -> Box<dyn Layer> {
        match self.name {
            'D' => {
                let weights_f32: Vec<f32> = self.weights.split(" ").into_iter().map(|val| val.parse().unwrap()).collect();
                let bias_f32: Vec<f32> = self.bias.split(" ").into_iter().map(|val| val.parse().unwrap()).collect();
                let dense_layer: Dense = Dense::new_ser(self.rows, self.cols, weights_f32, bias_f32);
                return Box::new(dense_layer)
            },
            _ => panic!("Not a supported type"),
        };
    }
    fn flatten_string(data: &Vec<Vec<f32>>) -> String {
        data.to_param()
            .into_iter()
            .map(|d| d.to_string() + " ")
            .collect::<String>().trim_end().to_string()
    }
    pub fn to_string(&self) -> String {
        format!("{}|{}|{}|{}|{}", self.name, self.rows, self.cols, self.weights, self.bias)
    }
    pub fn from_string(data: String) -> Self {
        let mut parse_res = data.split("|");
        let name: char = parse_res.next().unwrap().chars().next().unwrap();
        let rows: usize = str::parse(parse_res.next().unwrap()).unwrap();
        let cols: usize = str::parse(parse_res.next().unwrap()).unwrap();
        let weights = parse_res.next().unwrap().to_string();
        let bias = parse_res.next().unwrap().to_string();

        Self { name, rows, cols, weights, bias }
    }
}
