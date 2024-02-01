use crate::core::data::input::Input;

///Custom return type for generating a gradient in the (Bias, Weight) format
pub struct GradientPair(pub Box<dyn Input>,pub Box<dyn Input>);
