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
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::Add(a, b),
                dtype: node_a.dtype,
            };
            // TODO: special case adding const zero
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
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
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::Mul(a, b),
                dtype: node_a.dtype,
            };
            // TODO: special case adding const zero
            // TODO: support broadcastable shapes
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
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
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::LessThan(a, b),
                dtype: xla::ElementType::Pred,
            };
            // TODO: special case adding const zero
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
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
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::GreaterThan(a, b),
                dtype: xla::ElementType::Pred,
            };
            // TODO: special case adding const zero
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
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
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::LessThanEq(a, b),
                dtype: xla::ElementType::Pred,
            };
            // TODO: special case adding const zero
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
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
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::GreaterThanEq(a, b),
                dtype: xla::ElementType::Pred,
            };
            // TODO: special case adding const zero
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
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
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            let pred_node_id = self.nodes.insert(Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::LessThanEq(a, b),
                dtype: xla::ElementType::Pred,
            });
            let node_a = &self.nodes[a];
            let node_b = &self.nodes[b];
            let node = Node {
                callsite: callsite!(1),
                /// TODO: this is not general enough
                shape: Shape::new(),
                operation: Operation::Select { pred: pred_node_id, on_true: a, on_false: b },
                dtype: node_a.dtype,
            };
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
            }
        }
    }

    pub fn maximum<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
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
            let pred_node_id = self.nodes.insert(Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::GreaterThanEq(a, b),
                dtype: xla::ElementType::Pred,
            });
            let node_a = &self.nodes[a];
            let node_b = &self.nodes[b];
            let node = Node {
                callsite: callsite!(1),
                /// TODO: this is not general enough
                shape: node_a.shape,
                operation: Operation::Select { pred: pred_node_id, on_true: a, on_false: b },
                dtype: node_a.dtype,
            };
            if !node_a.shape.broadcastable(&node_b.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
            }
        }
    }

    pub fn relu<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
    ) -> Result<NodeIdentifier> {
        let a = a.into();
        let a_dtype = self.nodes[a].dtype;

        let const_zero = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::new(),
            operation: Operation::Constant(ConstantBinding{ value: xla::Literal::scalar(0).convert(a_dtype.primitive_type())? }),
            dtype: a_dtype,
        });

        self.maximum(a, b)

    }
}
