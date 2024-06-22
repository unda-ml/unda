use super::*;

impl Context {
    /// TODO: Typecheck pred
    pub fn select(
        &mut self,
        pred: NodeIdentifier,
        on_true: NodeIdentifier,
        on_false: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let pred = self.stop_gradient(pred);
        let node_pred = &self.nodes[pred];
        let node_true = &self.nodes[on_true];
        let node_false = &self.nodes[on_false];

        if node_true.dtype != node_false.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_true.dtype,
                node_false.dtype,
                callsite!(1),
            ))
        } else {
            match node_true.shape.broadcast(&node_false.shape) {
                None => Err(ContextError::IncompatibleOperandShapes(
                    node_true.shape.clone(),
                    node_false.shape.clone(),
                    callsite!(1),
                )),
                Some(s) => match s.broadcast(&node_pred.shape) {
                    None => Err(ContextError::IncompatibleOperandShapes(
                        s,
                        node_pred.shape.clone(),
                        callsite!(1),
                    )),
                    Some(sh) => {
                        let node = Node {
                            callsite: callsite!(1),
                            shape: sh,
                            operation: Operation::Select {
                                pred,
                                on_true,
                                on_false,
                            },
                            dtype: node_true.dtype,
                        };
                        let node_id = self.nodes.insert(node);
                        self.dependent_nodes
                            .entry(pred)
                            .or_default()
                            .push(node_id);
                        self.dependent_nodes
                            .entry(on_true)
                            .or_default()
                            .push(node_id);
                        if on_true != on_false {
                            self.dependent_nodes
                                .entry(on_false)
                                .or_default()
                                .push(node_id);
                        }
                        Ok(node_id)
                    }
                },
            }
        }
    }
}
