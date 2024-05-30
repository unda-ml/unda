use super::*;

impl Context {
    pub fn parameter<S: AsRef<str>, H: Into<Shape>>(
        &mut self,
        name: S,
        shape: H,
        dtype: xla::ElementType,
    ) -> Result<NodeIdentifier> {
        let name = name.as_ref().to_string();
        for node_id in self.parameters.iter() {
            let parameter = &self.nodes[*node_id];
            match &parameter.operation {
                Operation::Parameter(binding) => {
                    if *binding == name {
                        return Err(ContextError::DuplicateParameter(
                            name,
                            callsite!(1),
                            parameter.callsite.clone(),
                        ));
                    }
                }
                _ => unreachable!(),
            };
        }
        let param = self.nodes.insert(Node {
            callsite: callsite!(1),
            // It makes sense to pass the shape of the parameter to its constructor
            shape: shape.into(),
            operation: Operation::Parameter(name),
            dtype,
        });
        self.parameters.push(param);
        Ok(param)
    }
}
