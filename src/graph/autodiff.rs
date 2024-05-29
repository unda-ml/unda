use smallvec::SmallVec;

use self::dtypes::check_fp_type;

use super::*;

impl Context {
    pub fn stop_gradient(&mut self, node: NodeIdentifier) -> NodeIdentifier {
        let new_node = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[node].shape.clone(),
            operation: Operation::StopGradient(node),
            dtype: self.nodes[node].dtype,
        });
        self.dependent_nodes.entry(node).or_default().push(new_node);
        new_node
    }

    /// TODO: Better names for these variables
    fn gradient_is_dependent(
        &mut self,
        dependent: NodeIdentifier,
        independent: NodeIdentifier,
    ) -> bool {
        if dependent == independent {
            true
        } else if self.dependent_nodes.contains_key(&independent) {
            let dep_nodes = self.dependent_nodes[&independent].clone();
            for node_id in dep_nodes {
                match self.nodes[independent].operation {
                    Operation::StopGradient(_) => continue,
                    _ => {
                        if self.gradient_is_dependent(dependent, node_id) {
                            return true;
                        }
                    }
                }
            }
            false
        } else {
            false
        }
    }

    pub fn diff(
        &mut self,
        output: NodeIdentifier,
        with_respect_to: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let wrt_shape = self.nodes[with_respect_to].shape.clone();
        let wrt_dtype = check_fp_type(self.nodes[with_respect_to].dtype)?;

        if output == with_respect_to {
            return self.scalar(1, wrt_dtype);
        }

        let mut dependent_pullbacks: SmallVec<[NodeIdentifier; 2]> = SmallVec::new();

        if self.dependent_nodes.contains_key(&with_respect_to) {
            let dependent_nodes = self.dependent_nodes[&with_respect_to].clone();

            for dependent_node in dependent_nodes {
                if self.gradient_is_dependent(output, dependent_node) {
                    //Again again, clone() here is not wonderful, there's gotta be a better way to
                    //store the i64 vec for Transpose
                    match self.nodes[dependent_node].operation.clone() {
                        Operation::Constant(_) 
                            | Operation::RngUniform(_, _, _) 
                            | Operation::RngNormal(_, _, _) => panic!("Constant found as dependent node!"),
                        Operation::Parameter(_) => panic!("Parameter found as dependent node!"),
                        Operation::StopGradient(_) => continue,

                        Operation::Equal(_, _)
                        | Operation::NotEqual(_, _)
                        | Operation::LessThan(_, _)
                        | Operation::LessThanEq(_, _)
                        | Operation::GreaterThan(_, _)
                        | Operation::GreaterThanEq(_, _) => {
                            return Err(ContextError::NonDifferentiableOpError(
                                self.nodes[dependent_node].callsite.clone(),
                            ));
                        }

                        Operation::OneHot(_)
                        | Operation::ReduceArgmax { node: _, dim: _ }
                        | Operation::TypeCast(_, _) => {
                            return Err(ContextError::NonDifferentiableOpError(
                                self.nodes[dependent_node].callsite.clone(),
                            ));
                        }

                        Operation::Reshape(node) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            let node_sh = self.nodes[node].shape.clone();
                            let pullback = self.reshape(next_pullback, node_sh)?;
                            dependent_pullbacks.push(pullback);
                        }

                        Operation::Transpose(a, p) => {
                            if a == with_respect_to {
                                let next_pullback = self.diff(output, dependent_node)?;
                                let inv_perm = Context::inv_perm(&p);

                                let pullback = self.transpose(next_pullback, &inv_perm)?;
                                dependent_pullbacks.push(pullback);
                            }
                        }

                        Operation::ZerosLike(_) => continue,

                        Operation::Add(a, b) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            if a == with_respect_to {
                                let next_pullback =
                                    self.sum_to_shape(next_pullback, self.nodes[a].shape.clone())?;
                                dependent_pullbacks.push(next_pullback);
                            }
                            if b == with_respect_to {
                                let next_pullback =
                                    self.sum_to_shape(next_pullback, self.nodes[b].shape.clone())?;
                                dependent_pullbacks.push(next_pullback);
                            }
                        }

                        Operation::Sub(a, b) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            if a == with_respect_to {
                                let next_pullback =
                                    self.sum_to_shape(next_pullback, self.nodes[a].shape.clone())?;
                                dependent_pullbacks.push(next_pullback);
                            }
                            if b == with_respect_to {
                                let next_pullback =
                                    self.sum_to_shape(next_pullback, self.nodes[b].shape.clone())?;
                                dependent_pullbacks.push(self.neg(next_pullback));
                            }
                        }

                        Operation::Mul(a, b) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            if a == b && a == with_respect_to {
                                let two = self.scalar(2, wrt_dtype)?;
                                let mul2 = self.mul(two, a)?;
                                let this_pullback = self.mul(mul2, next_pullback)?;
                                let this_pullback =
                                    self.sum_to_shape(this_pullback, self.nodes[a].shape.clone())?;
                                dependent_pullbacks.push(this_pullback);
                            } else if a == with_respect_to {
                                let this_pullback = self.mul(next_pullback, b)?;
                                let this_pullback =
                                    self.sum_to_shape(this_pullback, self.nodes[a].shape.clone())?;
                                dependent_pullbacks.push(this_pullback);
                            } else if b == with_respect_to {
                                let this_pullback = self.mul(next_pullback, a)?;
                                let this_pullback =
                                    self.sum_to_shape(this_pullback, self.nodes[b].shape.clone())?;
                                dependent_pullbacks.push(this_pullback);
                            }
                        }

                        Operation::MatMul(a, b) => {
                            let next_pullback = self.diff(output, dependent_node)?;

                            // TODO: Add case for a == b
                            if a == with_respect_to {
                                let mut transpose_dims =
                                    Vec::from_iter(0..(self.nodes[b].shape.ndims() as i64));
                                let ndims = transpose_dims.len();
                                transpose_dims.swap(ndims - 2, ndims - 1);

                                let transpose = self.transpose(b, &transpose_dims)?;
                                let this_pullback = self.matmul(next_pullback, transpose)?;
                                dependent_pullbacks.push(this_pullback);
                            } else if b == with_respect_to {
                                let mut transpose_dims =
                                    Vec::from_iter(0..(self.nodes[a].shape.ndims() as i64));
                                let ndims = transpose_dims.len();
                                transpose_dims.swap(ndims - 2, ndims - 1);

                                let transpose = self.transpose(a, &transpose_dims)?;
                                let this_pullback = self.matmul(transpose, next_pullback)?;
                                dependent_pullbacks.push(this_pullback);
                            }
                        }

                        Operation::Div(a, b) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            if a == with_respect_to {
                                let this_pullback = self.div(next_pullback, b)?;
                                let this_pullback =
                                    self.sum_to_shape(this_pullback, self.nodes[a].shape.clone())?;
                                dependent_pullbacks.push(this_pullback);
                            }
                            if b == with_respect_to {
                                let mul = self.mul(b, b)?;
                                let div = self.div(a, mul)?;
                                let neg = self.neg(div);
                                let this_pullback = self.mul(neg, next_pullback)?;
                                let this_pullback =
                                    self.sum_to_shape(this_pullback, self.nodes[b].shape.clone())?;
                                dependent_pullbacks.push(this_pullback);
                            }
                        }

                        // TODO: handle potentially broadcasted operands here
                        Operation::Pow(a, b) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            if a == with_respect_to {
                                let one = self.scalar(1, wrt_dtype)?;
                                let b_min_one = self.sub(b, one)?;

                                let new_pow = self.pow(a, b_min_one)?;
                                let power_rule = self.mul(b, new_pow)?;

                                let this_pullback = self.mul(power_rule, next_pullback)?;
                                dependent_pullbacks.push(this_pullback);
                            }
                            if b == with_respect_to {
                                let log_a = self.log(a)?;
                                let log_times_orig = self.mul(log_a, dependent_node)?;
                                let this_pullback = self.mul(log_times_orig, next_pullback)?;

                                dependent_pullbacks.push(this_pullback);
                            }
                        }

                        Operation::Log(a) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            let one = self.scalar(1, wrt_dtype)?;
                            let quotient = self.div(one, a)?;

                            let next_pullback = self.mul(quotient, next_pullback)?;
                            dependent_pullbacks.push(next_pullback);
                        }

                        Operation::Neg(_) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            dependent_pullbacks.push(self.neg(next_pullback));
                        }

                        Operation::Exp(_) => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            let this_pullback = self.mul(next_pullback, dependent_node)?;

                            dependent_pullbacks.push(this_pullback);
                        }

                        // TODO: Test this!
                        Operation::TileInDim { node, n_tiles, dim } => {
                            let next_pullback = self.diff(output, dependent_node)?;

                            let mut new_shape = self.nodes[node].shape.clone();
                            new_shape.sizes.insert(dim as usize, n_tiles as u32);

                            let reshaped_pullback = self.reshape(next_pullback, new_shape)?;
                            dependent_pullbacks.push(self.reduce_sum(
                                reshaped_pullback,
                                dim,
                                false,
                            )?);
                        }

                        Operation::SliceInDim {
                            node,
                            start: _,
                            stop: _,
                            stride: _,
                            dim: _,
                        } => {
                            if self.gradient_is_dependent(node, dependent_node) {
                                panic!(
                                "Gradient of SliceInDim requires XLA scatter op to be implemented."
                            );
                            } else {
                                continue;
                            }
                        }

                        Operation::ReduceMax { node, dim: _ } => {
                            if self.gradient_is_dependent(node, dependent_node) {
                                panic!(
                                "Gradient of ReduceMax requires XLA scatter op to be implemented."
                            );
                            } else {
                                continue;
                            }
                        }

                        Operation::ReduceSum { node, dim } => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            let n_tiles = self.nodes[node].shape.sizes[dim as usize] as i64;

                            let mut new_shape = self.nodes[next_pullback].shape.clone();
                            if new_shape.ndims() != self.nodes[node].shape.ndims() {
                                new_shape.sizes.insert(dim as usize, 1u32);
                            }
                            let reshaped_pullback =
                                self.reshape(next_pullback, new_shape.clone())?;
                            let tiled_pullback =
                                self.tile_in_dim(reshaped_pullback, n_tiles, dim)?;

                            dependent_pullbacks.push(tiled_pullback);
                        }

                        Operation::ReduceMean { node, dim } => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            let n_tiles = self.nodes[node].shape.sizes[dim as usize] as i64;

                            let mut new_shape = self.nodes[next_pullback].shape.clone();
                            if new_shape.ndims() != self.nodes[node].shape.ndims() {
                                new_shape.sizes.insert(dim as usize, 1u32);
                            }
                            let reshaped_pullback = self.reshape(next_pullback, new_shape)?;
                            let tiled_pullback =
                                self.tile_in_dim(reshaped_pullback, n_tiles, dim)?;

                            let scale =
                                self.scalar(1.0 / (n_tiles as f32), self.nodes[node].dtype)?;
                            let rescaled_pullback = self.mul(scale, tiled_pullback)?;
                            dependent_pullbacks.push(rescaled_pullback);
                        }

                        Operation::Select {
                            pred,
                            on_true,
                            on_false,
                        } => {
                            let next_pullback = self.diff(output, dependent_node)?;
                            let const_zero = self.scalar(0, wrt_dtype)?;
                            if on_true == with_respect_to {
                                let select = self.select(pred, next_pullback, const_zero)?;
                                dependent_pullbacks.push(select);
                            }
                            if on_false == with_respect_to {
                                let select = self.select(pred, const_zero, next_pullback)?;
                                dependent_pullbacks.push(select);
                            }
                        }
                    }
                }
            }
        }

        self.smallvec_add(dependent_pullbacks, wrt_dtype, wrt_shape)
    }
}
