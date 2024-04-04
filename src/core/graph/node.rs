

use super::*;
use rand_distr::num_traits::Zero;
use slotmap::new_key_type;
use xla::Literal;
use std::{fmt::{Display, Formatter, Result}, hash::Hash};

use half::bf16;
use half::f16;

/// A node in the compute graph
#[derive(Clone, Debug)]
pub struct Node {
    /// helps identify where in the user's source code this node originated
    // TODO: gate this so its not present at all in release builds
    pub(crate) callsite: Callsite,
    /// shape of the output of this node
    pub shape: Shape,
    /// the operation this node performs
    pub(crate) operation: Operation,
    //// output type of the operation
    pub dtype: xla::ElementType,
}


new_key_type! {
    pub struct NodeIdentifier;
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.operation == other.operation
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.operation.hash(state);
    }
}


impl Node {
    /// Identifies constant operation node for easier
    /// constant folding in context.rs
    pub(crate) fn is_const(&self) -> Option<Literal> {
        match &self.operation {
            Operation::Constant(a) => Some(a.value.clone()),
            _ => None,
        }
    }
    pub(crate) fn is_one(&self) -> super::Result<bool> {
        //TODO! Convert type to primative type so we can collect the values
        match &self.operation {
            Operation::Constant(a) => {
                match a.value.element_type()? {
                    xla::ElementType::Pred => {
                      let data_ref = a.value.to_vec::<u8>()?;
                        for i in data_ref.iter() {
                            if i != &1u8 {
                                return Ok(false);
                            }
                        }

                    },
                    xla::ElementType::F16 => {
                        let data_ref = a.value.to_vec::<f16>()?;
                        for i in data_ref.iter() {
                            if *i != f16::ONE {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::F32 => {
                        let data_ref = a.value.to_vec::<f32>()?;
                        for i in data_ref.iter() {
                            if i != &1f32 {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::F64 => {
                        let data_ref = a.value.to_vec::<f64>()?;
                        for i in data_ref.iter() {
                            if i != &1f64 {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::U8 => {
                        let data_ref = a.value.to_vec::<u8>()?;
                        for i in data_ref.iter() {
                            if i != &1u8 {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::U16 => {
                        let data_ref = a.value.to_vec::<u16>()?;
                        for i in data_ref.iter() {
                            if i != &1u16 {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::U32 => {
                        let data_ref = a.value.to_vec::<u32>()?;
                        for i in data_ref.iter() {
                            if i != &1u32 {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::U64 => {
                        let data_ref = a.value.to_vec::<u64>()?;
                        for i in data_ref.iter() {
                            if i != &1u64 {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::S8 => {
                        let data_ref = a.value.to_vec::<i8>()?;
                        for i in data_ref.iter() {
                            if i != &1i8 {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::S16 => {
                        let data_ref = a.value.to_vec::<i16>()?;
                        for i in data_ref.iter() {
                            if i != &1i16 {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::S32 => {
                        let data_ref = a.value.to_vec::<i32>()?;
                        for i in data_ref.iter() {
                            if i != &1i32 {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::S64 => {
                        let data_ref = a.value.to_vec::<i64>()?;
                        for i in data_ref.iter() {
                            if i != &1i64 {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::C64 => {
                        //TODO
                        return Ok(false);
                    },
                    xla::ElementType::C128 => {
                        //TODO
                        return Ok(false);
                    },
                    xla::ElementType::Bf16 => {
                        let data_ref: Vec<bf16> = a.value.to_vec()?;
                        for i in data_ref.iter() {
                            if i != &bf16::ONE {
                                return Ok(false);
                            }
                        }
                    }
                }

                Ok(true)
            },
            _ => Ok(false),
        }
    }
    pub(crate) fn is_zero(&self) -> super::Result<bool> {
        //TODO! Convert type to primative type so we can collect the values
        match &self.operation {
            Operation::Constant(a) => {
                match a.value.element_type()? {
                    xla::ElementType::Pred => {
                      let data_ref = a.value.to_vec::<u8>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }

                    },
                    xla::ElementType::F16 => {
                        let data_ref = a.value.to_vec::<f16>()?;
                        for i in data_ref.iter() {
                            if *i != f16::ZERO {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::F32 => {
                        let data_ref = a.value.to_vec::<f32>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::F64 => {
                        let data_ref = a.value.to_vec::<f64>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::U8 => {
                        let data_ref = a.value.to_vec::<u8>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::U16 => {
                        let data_ref = a.value.to_vec::<u16>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::U32 => {
                        let data_ref = a.value.to_vec::<u32>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::U64 => {
                        let data_ref = a.value.to_vec::<u64>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::S8 => {
                        let data_ref = a.value.to_vec::<i8>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },

                    xla::ElementType::S16 => {
                        let data_ref = a.value.to_vec::<i16>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::S32 => {
                        let data_ref = a.value.to_vec::<i32>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::S64 => {
                        let data_ref = a.value.to_vec::<i64>()?;
                        for i in data_ref.iter() {
                            if !i.is_zero() {
                                return Ok(false);
                            }
                        }
                    },
                    xla::ElementType::C64 => {
                        //TODO
                        return Ok(false);
                    },
                    xla::ElementType::C128 => {
                        //TODO
                        return Ok(false);
                    },
                    xla::ElementType::Bf16 => {
                        let data_ref: Vec<bf16> = a.value.to_vec()?;
                        for i in data_ref.iter() {
                            if i != &bf16::ZERO || i == &bf16::NEG_ZERO {
                                return Ok(false);
                            }
                        }
                    }
                }

                Ok(true)
            },
            _ => Ok(false),
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{} {}", self.operation, self.callsite)
    }
}


