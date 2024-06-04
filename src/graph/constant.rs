use super::*;
use std::{path::Path};
use xla::FromRawBytes;

#[derive(Debug, Clone)]
pub struct ConstantBinding {
    /// unstructured float data. only makes sense combined with Node::dimension
    pub(crate) value: xla::Literal,
}

impl std::fmt::Display for ConstantBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.value.element_count() == 0 {
            write!(f, "0")?;
            return Ok(());
        }
        match self.value.ty() {
            xla::Result::Err(e) => write!(f, "XLA error: {}", e),
            xla::Result::Ok(t) => match t {
                xla::ElementType::F32 => match self.value.get_first_element::<f32>() {
                    Ok(x) => write!(f, "{}", x),
                    Err(_) => write!(f, "Unknown error getting first element of literal."),
                },
                xla::ElementType::F64 => match self.value.get_first_element::<f64>() {
                    Ok(x) => write!(f, "{}", x),
                    Err(_) => write!(f, "Unknown error getting first element of literal."),
                },
                xla::ElementType::S32 => match self.value.get_first_element::<i32>() {
                    Ok(x) => write!(f, "{}", x),
                    Err(_) => write!(f, "Unknown error getting first element of literal."),
                },
                xla::ElementType::S64 => match self.value.get_first_element::<i64>() {
                    Ok(x) => write!(f, "{}", x),
                    Err(_) => write!(f, "Unknown error getting first element of literal."),
                },
                unsupported => write!(f, "{:?} type not yet supported for Display", unsupported),
            },
        }?;
        //write!(f, "{}", self.value.get_first_element())?;
        // TODO: proper matrix printing?
        if self.value.element_count() > 1 {
            write!(f, "..")?;
        }
        Ok(())
    }
}

impl Context {
    pub fn literal_const(&mut self, value: xla::Literal) -> Result<NodeIdentifier> {
        let dtype = value.element_type()?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::new(),
            operation: Operation::Constant(ConstantBinding { value }),
            dtype: dtype,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn scalar<T: xla::ArrayElement + xla::NativeType>(
        &mut self,
        value: T,
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let value = xla::Literal::scalar(value).convert(dtype.primitive_type())?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::new(),
            operation: Operation::Constant(ConstantBinding { value }),
            dtype,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn vector<T: xla::ArrayElement + xla::NativeType, const N: usize>(
        &mut self,
        values: [T; N],
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let value = xla::Literal::vec1(&values).convert(dtype.primitive_type())?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: [N as u32].into(),
            operation: Operation::Constant(ConstantBinding { value }),
            dtype: T::TY,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn matrix<T: xla::ArrayElement + xla::NativeType, const N: usize, const M: usize>(
        &mut self,
        values: [[T; M]; N],
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let vec = values
            .into_iter()
            .flat_map(|f| f.into_iter())
            .collect::<Vec<T>>();
        let slice = vec.as_slice();
        let value = xla::Literal::vec1(slice).convert(dtype.primitive_type())?;
        let reshaped = value.reshape(&[N as i64, M as i64])?;
        let casted = reshaped.convert(dtype.primitive_type())?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: [N as u32, M as u32].into(),
            operation: Operation::Constant(ConstantBinding { value: casted }),
            dtype: T::TY,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn tensor_4d<T: xla::ArrayElement + xla::NativeType, const N: usize, const M: usize, const K: usize>(
        &mut self,
        values: [[[T; M]; N]; K],
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let vec = values
            .into_iter()
            .flat_map(|f| f.into_iter()
                      .flat_map(|k| k.into_iter()))
            .collect::<Vec<T>>();
        let slice = vec.as_slice();
        let value = xla::Literal::vec1(slice).convert(dtype.primitive_type())?;
        let reshaped = value.reshape(&[N as i64, M as i64, K as i64])?;
        let casted = reshaped.convert(dtype.primitive_type())?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: [N as u32, M as u32, K as u32].into(),
            operation: Operation::Constant(ConstantBinding { value: casted }),
            dtype: T::TY,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }


    pub fn zeroes<S: Into<Shape>>(&mut self, shape: S, dtype: xla::ElementType) -> Result<NodeIdentifier> {
        let shape = shape.into();
        let vec = (0..shape.size())
            .map(|_i| 0)
            .collect::<Vec<i64>>();
        let slice = vec.as_slice();
        let value = xla::Literal::vec1(slice).convert(dtype.primitive_type())?;
        let i64_vec = shape
            .sizes
            .iter()
            .map(|d| *d as i64)
            .collect::<Vec<i64>>();
        let i64_slice = i64_vec.as_slice();
        let reshaped = value.reshape(i64_slice)?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape,
            operation: Operation::Constant(ConstantBinding { value: reshaped }),
            dtype,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn const_from_npy<T: AsRef<Path>>(&mut self, path: T) -> Result<NodeIdentifier> {
        let l = xla::Literal::read_npy(path, &())?;
        let s = l.shape()?;
        let t = l.ty()?;
        let s = Shape::from_xla_shape(s)?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::Constant(ConstantBinding { value: l }),
            dtype: t,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn reshape_const<S: Into<Shape>>(
        &mut self,
        const_id: NodeIdentifier,
        new_shape: S,
    ) -> Result<NodeIdentifier> {
        let value = match &self.nodes[const_id].operation {
            Operation::Constant(b) => &b.value,
            _ => return Err(ContextError::NonConstantReshape(callsite!(1))),
        };
        let new_shape = new_shape.into();
        let i64_vec = new_shape
            .sizes
            .iter()
            .map(|d| *d as i64)
            .collect::<Vec<i64>>();
        let i64_slice = i64_vec.as_slice();
        let new_value = value.reshape(i64_slice)?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: new_shape,
            operation: Operation::Constant(ConstantBinding { value: new_value }),
            dtype: self.nodes[const_id].dtype,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }

    pub fn typecast_const(
        &mut self,
        const_id: NodeIdentifier,
        new_type: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let value = match &self.nodes[const_id].operation {
            Operation::Constant(b) => &b.value,
            _ => return Err(ContextError::NonConstantTypecast(callsite!(1))),
        };
        let new_value = value.convert(new_type.primitive_type())?;
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[const_id].shape.clone(),
            operation: Operation::Constant(ConstantBinding { value: new_value }),
            dtype: new_type,
        });
        self.constants.push(node_id);
        Ok(node_id)
    }
}
