use super::*;

impl Context {
    // TODO: use trait aliases for `Into<NodeIdentifier> + Copy`
    // when they get stablized: https://github.com/rust-lang/rust/issues/41517
    pub fn add<A: Into<NodeIdentifier> + Copy, B: Into<NodeIdentifier> + Copy>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<NodeIdentifier> {
        let node_a = &self.nodes[a.into()];
        let node_b = &self.nodes[b.into()];

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
                operation: Operation::Add(a.into(), b.into()),
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
        let node_a = &self.nodes[a.into()];
        let node_b = &self.nodes[b.into()];

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
                operation: Operation::Mul(a.into(), b.into()),
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
}
