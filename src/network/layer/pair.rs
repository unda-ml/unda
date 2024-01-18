use crate::network::input::Input;

pub struct GradientPair(pub Box<dyn Input>,pub Box<dyn Input>);
