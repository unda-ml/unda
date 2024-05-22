use smallvec::SmallVec;
use std::cmp::Ordering::{Less, Greater, Equal};

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

    pub fn rng_uniform(&mut self, min: NodeIdentifier, max: NodeIdentifier, shape: &[u32]) -> Result<NodeIdentifier> {
        if self.nodes[min].dtype != self.nodes[max].dtype {
            Err(ContextError::IncompatibleOperandTypes(
                self.nodes[min].dtype,
                self.nodes[max].dtype,
                callsite!(1),
            ))
        } else {
            let shape_node = Shape::from(shape);
            let node = Node {
                callsite: callsite!(1),
                shape: shape_node.clone(),
                operation: Operation::RngUniform(min, max, shape_node),
                dtype: self.nodes[min].dtype,
            };
            let node_id = self.nodes.insert(node);
            self.dependent_nodes.entry(min).or_default().push(node_id);
            self.dependent_nodes.entry(max).or_default().push(node_id);

            Ok(node_id)
        }
    }

    pub fn rng_normal(&mut self, mu: NodeIdentifier, sigma: NodeIdentifier, shape: &[u32]) -> Result<NodeIdentifier> {
        if self.nodes[mu].dtype != self.nodes[sigma].dtype {
            Err(ContextError::IncompatibleOperandTypes(
                self.nodes[mu].dtype,
                self.nodes[sigma].dtype,
                callsite!(1),
            ))
        } else {
            let shape_node = Shape::from(shape);
            let node = Node {
                callsite: callsite!(1),
                shape: shape_node.clone(),
                operation: Operation::RngNormal(mu, sigma, shape_node),
                dtype: self.nodes[mu].dtype,
            };
            let node_id = self.nodes.insert(node);
            self.dependent_nodes.entry(mu).or_default().push(node_id);
            self.dependent_nodes.entry(sigma).or_default().push(node_id);

            Ok(node_id)
        }
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
        match nodes.len().cmp(&1) {
            Less => self.zeroes(default_shape, default_dtype),
            Equal => Ok(nodes[0]),
            Greater => {
                let node0 = nodes.pop().unwrap();
                let node1 = nodes.pop().unwrap();
                let mut add_node = self.add(node0, node1)?;
                for next_node in nodes.into_iter() {
                    add_node = self.add(add_node, next_node)?;
                }
                Ok(add_node)
            }

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
        for idx in index_perm {
            new_shape
                .sizes
                .push(self.nodes[a].shape.sizes[*idx as usize]);
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
    // This utility is very handy for dealing with tiled constants in fold_consts
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
                    return Err(ContextError::ExpectedGreaterSize(
                        shape.clone(),
                        s.clone(),
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

    // Utility function for summing a large tensor to a smaller shape
    // when it is known that the smaller shape broadcasts to the larger tensor's shape
    // This utility is very handy for dealing with broadcasted operands in autodiff
    pub fn sum_to_shape(&mut self, a: NodeIdentifier, shape: Shape) -> Result<NodeIdentifier> {
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
                if shape.size() > s.size() {
                    return Err(ContextError::ExpectedGreaterSize(
                        s.clone(),
                        shape.clone(),
                        callsite!(1),
                    ));
                }
                if shape.sizes.is_empty() {
                    let mut summed = a;
                    for _d in (0..s.ndims()).rev() {
                        summed = self.reduce_sum(summed, 0, false)?;
                    }
                    Ok(summed)
                } else {
                    let mut summed = a;
                    for d in 0..s.ndims() {
                        if shape.sizes[d] == 1 {
                            summed = self.reduce_sum(summed, d as i64, true)?;
                        }
                    }
                    Ok(summed)
                }
            }
        }
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
            dtype,
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
            operation: Operation::ReduceArgmax { node: a, dim },
            dtype: xla::ElementType::S64,
        });
        self.dependent_nodes
            .entry(a)
            .or_default()
            .push(node_id);
        self.maybe_keepdims(node_id, dim, keepdims)
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
            dtype,
        });
        self.dependent_nodes
            .entry(converted_labels)
            .or_default()
            .push(node_id);
        Ok(node_id)
    }

}
