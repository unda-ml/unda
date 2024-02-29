use super::*;

impl Context {
    pub fn stop_gradient(&mut self, node: NodeIdentifier) -> NodeIdentifier {
        self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[node].shape.clone(),
            operation: Operation::StopGradient(node),
            dtype: self.nodes[node].dtype,
        })
    }

    pub fn diff(&mut self, node: NodeIdentifier, with_respect_to: Parameter) -> NodeIdentifier {
        self.nodes.insert(Node {
            callsite: callsite!(1),
            shape: self.nodes[node].shape.clone(),
            operation: Operation::Diff(node, with_respect_to),
            dtype: self.nodes[with_respect_to.node].dtype,
        })
    }

    /// expand all Diff nodes into analytic derivatives
    /// return true if there may be further changes needed
    pub(crate) fn autodiff<A: Into<NodeIdentifier> + Copy>(
        &mut self,
        input: A,
        modification_limit: usize,
    ) -> Result<bool> {
        if modification_limit == 0 {
            return Ok(true);
        }
        let input = input.into();
        let input_node = &self.nodes[input];

        // traverse nodes until we find a Diff node or a leaf
        match input_node.operation {
            // leaf nodes mean no further processing
            Operation::Constant(_) => Ok(false),
            Operation::Parameter(_) => Ok(false),
            Operation::StopGradient(_) => Ok(false),
            // operations mean we need to go deeper
            Operation::Add(a, b)
            | Operation::Mul(a, b)
            | Operation::Equal(a, b)
            | Operation::LessThan(a, b)
            | Operation::GreaterThan(a, b)
            | Operation::LessThanEq(a, b)
            | Operation::GreaterThanEq(a, b) => {
                let r = self.autodiff(a, modification_limit)?;
                self.autodiff(b, modification_limit - (r as usize))
                    .map(|v| v || r)
            }
            Operation::Select {
                pred: _,
                on_true,
                on_false,
            } => {
                let r = self.autodiff(on_true, modification_limit)?;
                self.autodiff(on_false, modification_limit - (r as usize))
                    .map(|v| v || r)
            }
            Operation::TypeCast(node, ty) => self.autodiff(node, modification_limit),
            Operation::SliceInDim { node, .. } => self.autodiff(node, modification_limit),
            Operation::ZerosLike(node) => self.autodiff(node, modification_limit),
            // finally a Diff node, lets distribute it
            Operation::Diff(outer, outer_param) => {
                let outer_node = &self.nodes[outer];
                let outer_dtype = outer_node.dtype.primitive_type();
                match outer_node.operation.clone() {
                    Operation::Constant(_) => {
                        // derivative of a constant with respect to anything is 0
                        self.nodes[input].operation = Operation::Constant(ConstantBinding {
                            value: xla::Literal::scalar(0).convert(outer_dtype)?,
                        });
                        self.nodes[input].shape = [1].into();
                        Ok(true)
                    }
                    Operation::Parameter(_) => {
                        // derivative of a parameter with respect to itself is one, and otherwise zero
                        self.nodes[input].operation = Operation::Constant(ConstantBinding {
                            value: xla::Literal::scalar(
                                (outer == outer_param.into()) as u32,
                            )
                            .convert(outer_dtype)?,
                        });
                        self.nodes[input].shape = [].into();
                        Ok(true)
                    }
                    Operation::StopGradient(_) => Ok(false),
                    Operation::Add(a, b) => {
                        // derivative of a sum is the sum of derivatives
                        // Diff (Sum a b) x = Sum (Diff a x) (Diff b x)
                        let diff_a_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[a].shape.clone(),
                            operation: Operation::Diff(a, outer_param),
                            dtype: self.nodes[a].dtype,
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Diff(b, outer_param),
                            dtype: self.nodes[b].dtype,
                        };
                        // propagate original Add callsite to the new Add node
                        self.nodes[input].callsite = outer_node.callsite.clone();
                        let diff_a = self.nodes.insert(diff_a_node);
                        let diff_b = self.nodes.insert(diff_b_node);
                        self.nodes[input].operation = Operation::Add(diff_a, diff_b);
                        // rerun autodiff on the node we replaced
                        self.autodiff(input, modification_limit - 1)
                    }
                    Operation::Mul(a, b) => {
                        // product rule
                        // Diff (Mul a b) x = Sum (Mul (Diff a x) b) (Mul a (Diff b x))
                        let diff_a_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[a].shape.clone(),
                            operation: Operation::Diff(a, outer_param),
                            dtype: self.nodes[a].dtype,
                        };
                        let diff_b_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Diff(b, outer_param),
                            dtype: self.nodes[b].dtype,
                        };
                        // propagate original Mul callsite to the new Add node
                        self.nodes[input].callsite = outer_node.callsite.clone();
                        let diff_a = self.nodes.insert(diff_a_node);
                        let diff_b = self.nodes.insert(diff_b_node);
                        let prod_a_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input].callsite.clone(),
                            shape: self.nodes[a].shape.clone(),
                            operation: Operation::Mul(diff_a, b),
                            dtype: self.nodes[a].dtype,
                        };
                        let prod_b_node = Node {
                            // propagate original Mul callsite to the new Mul node
                            callsite: self.nodes[input].callsite.clone(),
                            shape: self.nodes[b].shape.clone(),
                            operation: Operation::Mul(a, diff_b),
                            dtype: self.nodes[b].dtype,
                        };
                        let prod_a = self.nodes.insert(prod_a_node);
                        let prod_b = self.nodes.insert(prod_b_node);
                        self.nodes[input].operation = Operation::Add(prod_a, prod_b);
                        // rerun autodiff on the node we replaced
                        self.autodiff(input, modification_limit - 1)
                    }

                    Operation::Equal(_, _)
                    | Operation::LessThan(_, _)
                    | Operation::GreaterThan(_, _)
                    | Operation::LessThanEq(_, _)
                    | Operation::GreaterThanEq(_, _) => Err(ContextError::NonDifferentiableError(outer_node.callsite.clone())),
                    Operation::TypeCast(_, _) => Err(ContextError::NonDifferentiableError(outer_node.callsite.clone())),

                    Operation::Select {
                        pred,
                        on_true,
                        on_false,
                    } => {
                        // derivative of select is select of derivatives
                        let diff_true_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[on_true].shape.clone(),
                            operation: Operation::Diff(on_true, outer_param),
                            dtype: self.nodes[on_true].dtype,
                        };
                        let diff_false_node = Node {
                            // propagate original Diff callsite to the new Diff node
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[on_false].shape.clone(),
                            operation: Operation::Diff(on_false, outer_param),
                            dtype: self.nodes[on_false].dtype,
                        };
                        // propagate original Mul callsite to the new Add node
                        self.nodes[input].callsite = outer_node.callsite.clone();
                        let diff_true = self.nodes.insert(diff_true_node);
                        let diff_false = self.nodes.insert(diff_false_node);

                        self.nodes[input].operation = Operation::Select {
                            pred: pred,
                            on_true: diff_true,
                            on_false: diff_false,
                        };
                        // rerun autodiff on the node we replaced
                        self.autodiff(input, modification_limit - 1)
                    }

                    /*
                    Operation::SliceInDim { node, start, stop, stride, dim } => {
                        let diff_node = Node {
                            callsite: input_node.callsite.clone(),
                            shape: self.nodes[node].shape.clone(),
                            operation: Operation::Diff(node, outer_param),
                            dtype: self.nodes[node].dtype
                        };
                        self.nodes[input].callsite = outer_node.callsite.clone();
                        let diff_node = self.nodes.insert(diff_node);
                        let zero_node = self.nodes.insert(Node {
                            callsite: input_node.callsite.clone(),
                            shape: node.shape.clone(),
                            operation: Operation::ZerosLike(node)
                        });
                    }
                    */
                    Operation::SliceInDim { node, start, stop, stride, dim } => panic!("Differentiating SliceInDim not yet supported, xla-rs must implement scatter operation."),

                    Operation::ZerosLike(node) => {
                        self.nodes[input].operation = Operation::ZerosLike(node);
                        self.autodiff(input, modification_limit - 1)
                    }

                    Operation::Diff(inner, _) => {
                        // derivative of a derivative, apply the inner one first then try again on the outer.
                        let r = self.autodiff(inner, modification_limit)?;
                        self.autodiff(outer, modification_limit - (r as usize))
                            .map(|v| v || r)
                    }
                }
            }
        }
    }
}
