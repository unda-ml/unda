use smallvec::SmallVec;

use super::*;

impl Context {
    pub fn stop_gradient(&mut self, node: NodeIdentifier) -> NodeIdentifier {
        let new_node = self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[node].shape.clone(),
            operation: Operation::StopGradient(node),
            dtype: self.nodes[node].dtype,
        });
        self.dependent_nodes.entry(node).or_insert(Vec::new()).push(new_node);
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
        node: NodeIdentifier,
        with_respect_to: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let wrt_dtype = self.nodes[with_respect_to].dtype;

        if ![
            xla::ElementType::F16,
            xla::ElementType::Bf16,
            xla::ElementType::F32,
            xla::ElementType::F64,
            xla::ElementType::C64,
            xla::ElementType::C128,
        ]
        .contains(&wrt_dtype)
        {
            return Err(ContextError::NonDifferentiableTypeError(
                self.nodes[with_respect_to].callsite.clone(),
            ));
        }
        if node == with_respect_to {
            return self.scalar(1, wrt_dtype);
        }

        let mut dependent_pullbacks: SmallVec<[NodeIdentifier; 2]> = SmallVec::new();

        if self.dependent_nodes.contains_key(&with_respect_to) {
            let dependent_nodes = self.dependent_nodes[&with_respect_to].clone();

            for dependent_node in dependent_nodes {
                match self.nodes[dependent_node].operation {
                    Operation::Constant(_) => panic!("Constant found as dependent node!"),
                    Operation::Parameter(_) => panic!("Parameter found as dependent node!"),
                    Operation::StopGradient(a) => continue,

                    Operation::Equal(a, b)
                    | Operation::NotEqual(a, b)
                    | Operation::LessThan(a, b)
                    | Operation::LessThanEq(a, b)
                    | Operation::GreaterThan(a, b)
                    | Operation::GreaterThanEq(a, b) => {
                        if self.gradient_is_dependent(node, dependent_node) {
                            return Err(ContextError::NonDifferentiableOpError(
                                self.nodes[dependent_node].callsite.clone(),
                            ));
                        } else {
                            continue;
                        }
                    }

                    Operation::OneHot(node) => {
                        if self.gradient_is_dependent(node, dependent_node) {
                            return Err(ContextError::NonDifferentiableOpError(
                                self.nodes[dependent_node].callsite.clone(),
                            ));
                        } else {
                            continue;
                        }
                    }

                    Operation::TypeCast(a, _) => {
                        if self.gradient_is_dependent(node, dependent_node) {
                            return Err(ContextError::NonDifferentiableOpError(
                                self.nodes[dependent_node].callsite.clone(),
                            ));
                        } else {
                            continue;
                        }
                    }

                    Operation::ZerosLike(_) => continue,

                    Operation::Add(a, b) => {
                        if a == with_respect_to {
                            dependent_pullbacks.push(self.diff(node, dependent_node)?);
                        }
                        if b == with_respect_to {
                            dependent_pullbacks.push(self.diff(node, dependent_node)?);
                        }
                    }

                    Operation::Sub(a, b) => {
                        if a == with_respect_to {
                            dependent_pullbacks.push(self.diff(node, dependent_node)?);
                        }
                        if b == with_respect_to {
                            let next_pullback = self.diff(node, dependent_node)?;
                            dependent_pullbacks.push(self.neg(next_pullback));
                        }
                    }

                    Operation::Mul(a, b) => {
                        let next_pullback = self.diff(node, dependent_node)?;
                        if a == with_respect_to {
                            let mul = self.mul(next_pullback, a)?;
                            dependent_pullbacks.push(mul);
                        }
                        if b == with_respect_to {
                            let mul = self.mul(a, next_pullback)?;
                            dependent_pullbacks.push(mul);
                        }
                    }

                    Operation::Neg(a) => {
                        let next_pullback = self.diff(node, dependent_node)?;
                        dependent_pullbacks.push(self.neg(next_pullback));
                    }

                    Operation::SliceInDim { node, start, stop, stride, dim } => {
                        if self.gradient_is_dependent(node, dependent_node) {
                            panic!("Gradient of SliceInDim requires XLA scatter op to be implemented.");
                        } else {
                            continue;
                        }
                    }

                    Operation::ReduceMax { node, dim, keepdims } => {
                        if self.gradient_is_dependent(node, dependent_node) {
                            panic!("Gradient of ReduceMax requires XLA scatter op to be implemented.");
                        } else {
                            continue;
                        }
                    }

                    Operation::Select { pred, on_true, on_false } => {
                        let next_pullback = self.diff(node, dependent_node)?;
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

        self.smallvec_add(dependent_pullbacks, wrt_dtype)
    }
}
