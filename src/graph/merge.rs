use std::collections::HashMap;

use super::NodeIdentifier;
use super::Operation;
use super::Result;

use super::Context;


impl Context {
    pub fn merge_graphs(&mut self, other: &Context, other_outputs: &[NodeIdentifier], desired_remaps: &[NodeIdentifier]) -> Result<Vec<NodeIdentifier>> {

        let mut old_to_new: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        let mut addition_queue = other.inputs(other_outputs)?;

        macro_rules! generate_operation {
            ($OpType:ident, $($id:ident),+) => {
                Operation::$OpType($($id),+)
            }
        }

        macro_rules! generate_call {
            ($operation:ident, $($id:ident),+) => {
                self.$operation($(old_to_new[&$id]),+)
            };
        }

        while let Some(old_node) = addition_queue.pop() {
            //Add node to self's slotmap based on matching operation. Add old id and new id to
            //old_to_new map and if operation uses any NodeIdentifiers access those from old_to_new
            //before creating new node
            let new_id = match other.nodes[old_node].operation {
                generate_operation!(Add, a, b) => generate_call!(add, a, b),
                generate_operation!(Pow, a, b) => generate_call!(pow, a, b),
                generate_operation!(Sub, a, b) => generate_call!(sub, a, b),
                generate_operation!(Mul, a, b) => generate_call!(mul, a, b),
                generate_operation!(Div, a, b) => generate_call!(div, a, b),
                generate_operation!(Equal, a, b) => generate_call!(eq, a, b),
                generate_operation!(NotEqual, a, b) => generate_call!(neq, a, b),
                generate_operation!(LessThan, a, b) => generate_call!(lt, a, b),
                generate_operation!(GreaterThan, a, b) => generate_call!(gt, a, b),
                generate_operation!(GreaterThanEq, a, b) => generate_call!(ge, a, b),
                generate_operation!(LessThanEq, a, b) => generate_call!(le, a, b),
                generate_operation!(MatMul, a, b) => generate_call!(matmul, a, b),
                
                generate_operation!(Exp, a) => generate_call!(exp, a),
                generate_operation!(Log, a) => generate_call!(log, a),
                generate_operation!(StopGradient, a) => Ok(generate_call!(stop_gradient, a)),
                generate_operation!(Neg, a) => Ok(generate_call!(neg, a)),
                generate_operation!(ZerosLike, a) => Ok(generate_call!(zeros_like, a)),

                /* TODO
                 * TypeCast(ID, ELementType)
                 * ReShape(ID),
                 * Transpose(ID, Vec<I64>)
                 * Constants (gonna suck cause dims)
                 * Parameter(String)
                 * SliceInDim(ID, i64, i64, i64, i64)
                 * TileInDim(ID, i64, i64)
                 * ReduceMax{ID, i64}
                 * ReduceSum{ID, i64}
                 * ReduceMean{ID, i64}
                 * ReduceArgmax{ID, i64}
                 *
                 * OneHot(ID)
                 *
                 * RngUniform(ID, ID, Shape)
                 * RngNormal(ID, ID, Shape)
                 *
                 */
                Operation::Select { pred, on_true, on_false } => generate_call!(select, pred, on_true, on_false),
                _ => panic!()
            }?;

            if let Some(deps) = other.dependent_nodes.clone().get(&old_node) {
                for node in deps {
                    addition_queue.insert(0, *node);
                }
            }

            old_to_new.insert(old_node, new_id);
        }

        let mut new_remaps = vec![];

        for old in desired_remaps {
            new_remaps.push(old_to_new[old])
        }

        Ok(new_remaps)
    }

    pub fn find_and_replace_params(&mut self, param_reps: &[(&str, &[NodeIdentifier])]) -> Result<()> {
        for (param_name, rep_with) in param_reps {
            let params_with_name: Vec<NodeIdentifier> = self.nodes.clone().into_iter().filter(|(_, node)| {
                match node.operation.clone() {
                    Operation::Parameter(name) => name.contains(param_name),
                    _ => false
                }
            }).map(|(id, _)| id).collect();

            if params_with_name.len() != rep_with.len() {
                return Err(super::ContextError::IncorrectOutputSizeError(rep_with.len(), params_with_name.len()));
            }

            for i in 0..params_with_name.len() {
                self.replace_index(params_with_name[i], rep_with[i])?;
            }
        }

        Ok(())
    }

    fn inputs(&self, outputs: &[NodeIdentifier]) -> Result<Vec<NodeIdentifier>> {
        let mut inputs = vec![];
        let mut queue = outputs.to_vec();

        while let Some(current_node) = queue.pop() {
            todo!()
        }

        Ok(inputs)
    }
}
