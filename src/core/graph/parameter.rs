use super::*;

#[derive(Debug, Clone)]
pub struct ParameterBinding {
    // TODO: store something meaningful to XLA here
    pub(crate) name: String,
}

impl std::fmt::Display for ParameterBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Wraps a NodeIdentifier to have type-safe autodiff.
/// The constructor is private, so we know that the user
/// can only provide us one of these if we gave it to them.
/// This is why we need `Into<NodeIdentifier>` on all `Context` inputs.
#[derive(Debug, Clone, Copy)]
pub struct Parameter {
    pub(crate) node: NodeIdentifier,
}

impl From<Parameter> for NodeIdentifier {
    fn from(value: Parameter) -> Self {
        value.node
    }
}

impl Context {
    pub fn parameter<S: AsRef<str>, H: Into<Shape>>(
        &mut self,
        name: S,
        shape: H,
        dtype: xla::ElementType,
    ) -> Result<Parameter> {
        let name = name.as_ref().to_string();
        for node_id in self.parameters.iter() {
            let parameter = &self.nodes[*node_id];
            match &parameter.operation {
                Operation::Parameter(binding) => {
                    if binding.name == name {
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
        let param = Parameter {
            node: self.nodes.insert(Node {
                callsite: callsite!(1),
                // It makes sense to pass the shape of the parameter to its constructor
                shape: shape.into(),
                operation: Operation::Parameter(ParameterBinding { name }),
                dtype,
            }),
        };
        self.parameters.push(param.node);
        Ok(param)
    }
}
