use super::*;

impl Context {
    // TODO: use trait aliases for `Into<NodeIdentifier> + Copy`
    // when they get stablized: https://github.com/rust-lang/rust/issues/41517
    pub fn add<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::Add(a, b),
                        dtype: node_a.dtype,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn mul<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::Mul(a, b),
                        dtype: node_a.dtype,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn eq<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::Equal(a, b),
                        dtype: xla::ElementType::Pred,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn lt<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::LessThan(a, b),
                        dtype: xla::ElementType::Pred,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn gt<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::GreaterThan(a, b),
                        dtype: xla::ElementType::Pred,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn le<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::LessThanEq(a, b),
                        dtype: xla::ElementType::Pred,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn ge<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.broadcast(&node_b.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: s,
                        operation: Operation::GreaterThanEq(a, b),
                        dtype: xla::ElementType::Pred,
                    };
                    Ok(self.nodes.insert(node))
                }
            }
        }
    }

    pub fn minimum<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let pred = self.lt(a, b)?;
        self.select(pred, a, b)
    }

    pub fn maximum<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let b = b.into();
        let pred = self.gt(a, b)?;
        self.select(pred, a, b)
    }

    pub fn relu<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let a_dtype = self.nodes[a].dtype;

        let const_zero = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::new(),
            operation: Operation::Constant(ConstantBinding {
                value: xla::Literal::scalar(0).convert(a_dtype.primitive_type())?,
            }),
            dtype: a_dtype,
        });

        self.maximum(const_zero, a)
    }

    pub fn type_cast<A: Into<NodeIdentifier> + Copy>(&mut self, a: A, dtype: xla::ElementType) -> NodeIdentifier {
        let a = a.into();
        let a_shape = self.nodes[a].shape.clone();
        self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: a_shape,
            operation: Operation::TypeCast(a, dtype),
            dtype: dtype
        })
    }
}
