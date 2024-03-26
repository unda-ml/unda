use smallvec::SmallVec;

use self::dtypes::*;

use super::*;

impl Context {
    // TODO: use trait aliases for `Into<NodeIdentifier> + Copy`
    // when they get stablized: https://github.com/rust-lang/rust/issues/41517
    pub fn add(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn sub(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                        operation: Operation::Sub(a, b),
                        dtype: node_a.dtype,
                    };
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn neg(&mut self, a: NodeIdentifier) -> NodeIdentifier {
        let node = Node {
            callsite: callsite!(1),
            shape: self.nodes[a].shape.clone(),
            operation: Operation::Neg(a),
            dtype: self.nodes[a].dtype,
        };
        let node_id = self.nodes.insert(node);
        self.dependent_nodes.entry(a).or_default().push(node_id);
        node_id
    }

    pub fn log(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let node = Node {
            callsite: callsite!(1),
            shape: self.nodes[a].shape.clone(),
            operation: Operation::Log(a),
            dtype: self.nodes[a].dtype,
        };
        let node_id = self.nodes.insert(node);
        self.dependent_nodes.entry(a).or_default().push(node_id);
        Ok(node_id)
    }

    pub fn exp(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let node = Node {
            callsite: callsite!(1),
            shape: self.nodes[a].shape.clone(),
            operation: Operation::Exp(a),
            dtype: self.nodes[a].dtype,
        };
        let node_id = self.nodes.insert(node);
        self.dependent_nodes.entry(a).or_default().push(node_id);
        Ok(node_id)
    }

    pub fn pow(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                        operation: Operation::Pow(a, b),
                        dtype: node_a.dtype,
                    };
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn smallvec_add(
        &mut self,
        mut nodes: SmallVec<[NodeIdentifier; 2]>,
        default_dtype: xla::ElementType,
        default_shape: Shape,
    ) -> Result<NodeIdentifier> {
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
            self.zeroes(default_shape, default_dtype)
        }
    }

    pub fn matmul(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        if node_a.dtype != node_b.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_a.dtype,
                node_b.dtype,
                callsite!(1),
            ))
        } else {
            match node_a.shape.matmul_shape(&node_b.shape.sizes) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_a.shape.clone(),
                    node_b.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => {
                    let node = Node {
                        callsite: callsite!(1),
                        shape: Shape::from(s.as_slice()),
                        operation: Operation::MatMul(a, b),
                        dtype: node_a.dtype,
                    };
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn mul(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn div(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                        operation: Operation::Div(a, b),
                        dtype: node_a.dtype,
                    };
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn neq(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                        operation: Operation::NotEqual(a, b),
                        dtype: xla::ElementType::Pred,
                    };
                    let node_id = self.nodes.insert(node);
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    self.dependent_nodes.entry(b).or_default().push(node_id);
                    Ok(node_id)
                }
            }
        }
    }

    pub fn eq(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn lt(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn gt(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn le(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn ge(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
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
                    self.dependent_nodes.entry(a).or_default().push(node_id);
                    if a != b {
                        self.dependent_nodes.entry(b).or_default().push(node_id);
                    }
                    Ok(node_id)
                }
            }
        }
    }

    pub fn minimum(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
        let pred = self.lt(a, b)?;
        self.select(pred, a, b)
    }

    pub fn maximum(&mut self, a: NodeIdentifier, b: NodeIdentifier) -> Result<NodeIdentifier> {
        let pred = self.gt(a, b)?;
        self.select(pred, a, b)
    }

    pub fn relu(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        let const_zero = self.scalar(0, a_dtype)?;
        self.maximum(const_zero, a)
    }

    pub fn leaky_relu(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        //TODO: force dtype to be floating point or else this just becomes normal relu
        let const_small = self.scalar(0.001, a_dtype)?;
        let small_x = self.mul(a, const_small)?;

        self.maximum(small_x, a)
    }

    pub fn sigmoid(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        let one = self.scalar(1, a_dtype)?;
        let neg_x = self.neg(a);
        let exp_x = self.exp(neg_x)?;

        let one_p_exp_x = self.add(one, exp_x)?;

        self.div(one, one_p_exp_x)
    }

    pub fn softmax(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let dtype = check_fp_type(self.nodes[a].dtype)?;

        let max = self.reduce_max(a, 0, true)?;
        let stop_grad = self.stop_gradient(max);
        let unnormalized = self.sub(a, stop_grad)?;
        let unnormalized_exp = self.exp(unnormalized)?;

        let sum = self.reduce_sum(unnormalized_exp, 0, true)?;
        let eps = self.scalar(1e-8, dtype)?;
        // prevent division by 0
        let sum = self.add(sum, eps)?;

        self.div(unnormalized_exp, sum)
    }

    pub fn tanh(&mut self, a: NodeIdentifier) -> Result<NodeIdentifier> {
        let a_dtype = self.nodes[a].dtype;
        let two = self.scalar(2, a_dtype)?;
        let one = self.scalar(1, a_dtype)?;

        let two_a = self.mul(two, a)?;
        let sigmoid_a_2 = self.sigmoid(two_a)?;

        let two_sigmoid = self.mul(two, sigmoid_a_2)?;
        self.sub(two_sigmoid, one)
    }

    pub fn type_cast(&mut self, a: NodeIdentifier, dtype: xla::ElementType) -> NodeIdentifier {
        let a_shape = self.nodes[a].shape.clone();
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: a_shape,
            operation: Operation::TypeCast(a, dtype),
            dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        node_id
    }

    pub fn reshape<S: Into<Shape>>(
        &mut self,
        a: NodeIdentifier,
        shape: S,
    ) -> Result<NodeIdentifier> {
        let shape = shape.into();
        let a_size = self.nodes[a].shape.size();
        if a_size != shape.size() {
            Err(ContextError::ShapeConversion(
                ShapeConversionError::MismatchedSizes(
                    self.nodes[a].shape.clone(),
                    shape,
                    callsite!(1),
                ),
            ))
        } else {
            let node_id = self.nodes.insert(Node {
                callsite: callsite!(1),
                shape,
                operation: Operation::Reshape(a),
                dtype: self.nodes[a].dtype,
            });
            self.dependent_nodes.entry(a).or_default().push(node_id);
            Ok(node_id)
        }
    }

    pub(crate) fn inv_perm(index_perm: &[i64]) -> Vec<i64> {
        let mut res = vec![0i64; index_perm.len()];

        for (idx, val) in index_perm.iter().enumerate() {
            res[*val as usize] = idx as i64;
        }

        res
    }

    pub fn transpose(&mut self, a: NodeIdentifier, index_perm: &[i64]) -> Result<NodeIdentifier> {
        if index_perm.len() != self.nodes[a].shape.ndims() {
            return Err(ContextError::TransposeLenError(
                self.nodes[a].shape.ndims(),
                index_perm.len(),
                callsite!(1),
            ));
        }
        let mut new_shape = Shape::new();
        for d in 0..index_perm.len() {
            new_shape
                .sizes
                .push(self.nodes[a].shape.sizes[index_perm[d] as usize]);
        }
        let index_perms_deref = index_perm.to_vec();
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: new_shape,
            operation: Operation::Transpose(a, index_perms_deref),
            dtype: self.nodes[a].dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        Ok(node_id)
    }

    /// TODO: Need shape-checking here
    pub fn slice_in_dim(
        &mut self,
        a: NodeIdentifier,
        start: i64,
        stop: i64,
        stride: i64,
        dim: i64,
    ) -> Result<NodeIdentifier> {
        let mut s = self.nodes[a].shape.clone();
        s.sizes[dim as usize] = ((start - stop) / stride) as u32;

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::SliceInDim {
                node: a,
                start,
                stop,
                stride,
                dim,
            },
            dtype: self.nodes[a].dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        Ok(node_id)
    }

    pub fn tile_in_dim(
        &mut self,
        a: NodeIdentifier,
        n_tiles: i64,
        dim: i64,
    ) -> Result<NodeIdentifier> {
        let mut s = self.nodes[a].shape.clone();
        let node = if s.sizes.is_empty() {
            self.reshape(a, [1])?
        } else {
            a
        };
        if s.sizes.is_empty() {
            s.sizes.push(n_tiles as u32);
        } else {
            s.sizes[dim as usize] *= n_tiles as u32;
        }

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::TileInDim { node, n_tiles, dim },
            dtype: self.nodes[a].dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        Ok(node_id)
    }

    // Utility function for tiling a small tensor to a larger shape
    // when it is known that the smaller tensor's shape broadcasts to the larger shape
    pub fn tile_to_shape(&mut self, a: NodeIdentifier, shape: Shape) -> Result<NodeIdentifier> {
        let node_a_shape = self.nodes[a].shape.clone();

        if node_a_shape == shape {
            return Ok(a);
        }

        match node_a_shape.broadcast(&shape) {
            None => Err(ContextError::IncompatibleOperandShapes(
                node_a_shape,
                shape.clone(),
                callsite!(1),
            )),
            Some(s) => {
                if s.size() > shape.size() {
                    return Err(ContextError::IncompatibleOperandShapes(
                        node_a_shape,
                        shape.clone(),
                        callsite!(1),
                    ));
                }
                if node_a_shape.sizes.is_empty() {
                    let mut tiled = a;
                    for d in (0..s.ndims()).rev() {
                        tiled = self.tile_in_dim(tiled, s.sizes[d] as i64, 0)?;
                    }
                    Ok(tiled)
                } else {
                    let mut tiled = a;
                    for d in 0..s.ndims() {
                        if node_a_shape.sizes[d] == 1 {
                            tiled = self.tile_in_dim(tiled, s.sizes[d] as i64, d as i64)?;
                        }
                    }
                    Ok(tiled)
                }
            }
        }
    }

    pub fn zeros_like(&mut self, a: NodeIdentifier) -> NodeIdentifier {
        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[a].shape.clone(),
            operation: Operation::ZerosLike(a),
            dtype: self.nodes[a].dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        node_id
    }

    fn maybe_keepdims(
        &mut self,
        a: NodeIdentifier,
        dim: i64,
        keepdims: bool,
    ) -> Result<NodeIdentifier> {
        if keepdims {
            let mut s_keepdim = self.nodes[a].shape.clone();
            s_keepdim.sizes.insert(dim as usize, 1u32);
            self.reshape(a, s_keepdim)
        } else {
            Ok(a)
        }
    }

    pub fn reduce_max(
        &mut self,
        a: NodeIdentifier,
        dim: i64,
        keepdims: bool,
    ) -> Result<NodeIdentifier> {
        let mut s = self.nodes[a].shape.clone();
        if s.sizes.is_empty() {
            return Ok(a);
        }
        s.sizes.remove(dim as usize);

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::ReduceMax { node: a, dim },
            dtype: self.nodes[a].dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        self.maybe_keepdims(node_id, dim, keepdims)
    }

    pub fn reduce_sum(
        &mut self,
        a: NodeIdentifier,
        dim: i64,
        keepdims: bool,
    ) -> Result<NodeIdentifier> {
        let mut s = self.nodes[a].shape.clone();
        if s.sizes.is_empty() {
            return Ok(a);
        }
        s.sizes.remove(dim as usize);

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::ReduceSum { node: a, dim },
            dtype: self.nodes[a].dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);
        self.maybe_keepdims(node_id, dim, keepdims)
    }

    pub fn reduce_mean(
        &mut self,
        a: NodeIdentifier,
        dim: i64,
        keepdims: bool,
    ) -> Result<NodeIdentifier> {
        let dtype = check_fp_type(self.nodes[a].dtype)?;

        let mut s = self.nodes[a].shape.clone();
        if s.sizes.is_empty() {
            return Ok(a);
        }
        s.sizes.remove(dim as usize);

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::ReduceMean { node: a, dim },
            dtype: dtype,
        });
        self.dependent_nodes.entry(a).or_default().push(node_id);

        self.maybe_keepdims(node_id, dim, keepdims)
    }

    pub fn reduce_argmax(
        &mut self,
        a: NodeIdentifier,
        dim: i64,
        keepdims: bool,
    ) -> Result<NodeIdentifier> {
        let mut s = self.nodes[a].shape.clone();
        s.sizes.remove(dim as usize);

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: s,
            operation: Operation::ReduceArgmax { node: a, dim: dim },
            dtype: xla::ElementType::S64,
        });
        self.dependent_nodes
            .entry(a)
            .or_insert(Vec::new())
            .push(node_id);
        self.maybe_keepdims(node_id, dim, keepdims)
    }

    // assumes dense_predictions is rank 2 with dimension 0 being batch and dimension 1 being predictions
    // assumes sparse_label_vector is rank 1 i64 of class labels
    pub fn accuracy(
        &mut self,
        dense_predictions: NodeIdentifier,
        sparse_label_vector: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let converted_labels = match check_int_type(self.nodes[sparse_label_vector].dtype) {
            Ok(_) => self.type_cast(sparse_label_vector, xla::ElementType::S64),
            Err(e) => return Err(e),
        };
        let sparse_predictions = self.reduce_argmax(dense_predictions, 1, false)?;
        let compare = self.eq(sparse_predictions, converted_labels)?;
        let converted = self.type_cast(compare, xla::ElementType::F32);
        self.reduce_mean(converted, 0, false)
    }

    pub fn one_hot(
        &mut self,
        sparse_label_vector: NodeIdentifier,
        n_classes: usize,
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        if self.nodes[sparse_label_vector].shape.ndims() != 1 {
            return Err(ContextError::RankError(
                1,
                self.nodes[sparse_label_vector].shape.ndims(),
                callsite!(1),
            ));
        }
        let label_len = self.nodes[sparse_label_vector].shape.sizes[0];

        let converted_labels = match check_int_type(self.nodes[sparse_label_vector].dtype) {
            Ok(_) => self.type_cast(sparse_label_vector, xla::ElementType::S64),
            _ => unreachable!(),
        };

        let node_id = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: Shape::from([label_len, n_classes as u32]),
            operation: Operation::OneHot(converted_labels),
            dtype: dtype,
        });
        self.dependent_nodes
            .entry(converted_labels)
            .or_insert(Vec::new())
            .push(node_id);
        Ok(node_id)
    }

    pub fn mean_cross_entropy(
        &mut self,
        prediction_probabilities: NodeIdentifier,
        one_hot_labels: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let dtype = check_real_type(self.nodes[prediction_probabilities].dtype)?;
        if dtype != self.nodes[one_hot_labels].dtype {
            return Err(ContextError::IncompatibleOperandTypes(
                dtype,
                self.nodes[one_hot_labels].dtype,
                callsite!(1),
            ));
        }

        let eps = self.scalar(1e-8, dtype)?;
        // prevent logarithm of zero
        let offset = self.add(prediction_probabilities, eps)?;
        let log = self.log(offset)?;
        let neglog = self.neg(log);
        let mul = self.mul(one_hot_labels, neglog)?;
        let sum = self.reduce_sum(mul, 1, false)?;
        self.reduce_mean(sum, 0, false)
    }
}
