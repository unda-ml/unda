use smallvec::SmallVec;

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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
                }
            }
        }
    }

    pub fn smallvec_add(&mut self, mut nodes: SmallVec<[NodeIdentifier; 2]>, default_dtype: xla::ElementType) -> Result<NodeIdentifier> {
        if nodes.len() == 1 {
            Ok(nodes[0])
        } else if nodes.len() > 1 {
            let node0 = nodes.pop().unwrap();
            let node1 = nodes.pop().unwrap();
            let mut add_node = self.add(node0, node1)?;
            for next_node in nodes.into_iter() {
                add_node = self.add(add_node, next_node)?;
            }
            Ok(add_node)
        } else {
            self.scalar(0, default_dtype)
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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
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
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
                    self.dependent_nodes.entry(b).or_insert(Vec::new()).push(node_id);
                    Ok(node_id)
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
        let const_zero = self.scalar(0, a_dtype)?;
        self.maximum(const_zero, a)
    }

    pub fn type_cast<A: Into<NodeIdentifier> + Copy>(&mut self, a: A, dtype: xla::ElementType) -> NodeIdentifier {
        let a = a.into();
        let a_shape = self.nodes[a].shape.clone();
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: a_shape,
            operation: Operation::TypeCast(a, dtype),
            dtype: dtype
        });
        self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
        node_id
    }

    /// TODO: Need shape-checking here
    pub fn slice_in_dim<A: Into<NodeIdentifier> + Copy>(&mut self, a: A, start: i64, stop: i64, stride: i64, dim: i64) -> Result<NodeIdentifier> {
        let a = a.into();
        let mut s = Shape::new();
        for d in (0..self.nodes[a].shape.ndims()).rev() {
            if d as i64 == dim {
                s.sizes.push(1)
            } else {
                s.sizes.push(self.nodes[a].shape.sizes[d])
            }
        }
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::SliceInDim { node: a, start: start, stop: stop, stride: stride, dim: dim },
            dtype: self.nodes[a].dtype
        });
        self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
        Ok(node_id)
    }

    pub fn zeros_like<A: Into<NodeIdentifier> + Copy>(&mut self, a: A) -> NodeIdentifier {
        let a = a.into();
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[a].shape.clone(),
            operation: Operation::ZerosLike(a),
            dtype: self.nodes[a].dtype
        });
        self.dependent_nodes.entry(a).or_insert(Vec::new()).push(node_id);
        node_id
    }
}
