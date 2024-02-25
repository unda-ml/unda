use super::*;
use rand_distr::num_traits::Zero;
use slotmap::new_key_type;
use std::{fmt::{Display, Formatter, Result}, error::Error};

/// A node in the compute graph
pub struct Node {
    /// helps identify where in the user's source code this node originated
    // TODO: gate this so its not present at all in release builds
    pub(crate) callsite: Callsite,
    /// shape of the output of this node
    pub(crate) shape: Shape,
    /// the operation this node performs
    pub(crate) operation: Operation,
    //// output type of the operation
    pub(crate) dtype: xla::ElementType,
}

new_key_type! {
    pub struct NodeIdentifier;
}


impl Node {
    /// Identifies constant operation node for easier
    /// constant folding in context.rs
    pub(crate) fn is_const(&self) -> bool {
        return match self.operation {
            Operation::Constant(_) => true,
            _ => false,
        };
    }
    pub(crate) fn is_zero(&self) -> super::Result<bool> {
        //TODO! Convert type to primative type so we can collect the values
        return match &self.operation {
            Operation::Constant(a) => {
                match self.dtype {
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
                    _ => { return Ok(false); }
                }

                Ok(true)
            },
            _ => Ok(false),
        };
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{} {}", self.operation, self.callsite)
    }
}


