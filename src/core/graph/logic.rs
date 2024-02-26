use super::*;

impl Context {

    pub fn select<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy, C: Into<NodeIdentifier> + Copy>(
        &mut self,
        pred: A,
        on_true: B,
        on_false: C
    ) {
        let pred = pred.into();
        let on_true = on_true.into();
        let on_false = on_false.into();
        let node_pred = &self.nodes[pred];
        let node_true = &self.nodes[true];
        let node_false = &self.nodes[false];

        if node_true.dtype != node_false.dtype {
            Err(ContextError::IncompatibleOperandTypes(
                node_true.dtype,
                node_false.dtype,
                callsite!(1),
            ))
        } else {
            let node = Node {
                callsite: callsite!(1),
                // need to implement automatic shape broadcasting
                shape: Shape::new(),
                operation: Operation::Select{ pred: pred, on_true: on_true, on_false: on_false },
                dtype: node_true.dtype,
            };
            // TODO: special case adding const zero
            if !node_true.shape.broadcastable(&node_false.shape) {
                Err(ContextError::IncompatibleOperandShapes(
                    node_true.shape.clone(),
                    node_false.shape.clone(),
                    node.callsite,
                ))
            } else {
                Ok(self.nodes.insert(node))
            }
        }
    }
}