pub struct DataConverter<T>{
    pub data_to_serialized: Vec<T>,
    pub serialized_to_data: Vec<Vec<f64>> 
}

impl<T: PartialEq> DataConverter<T>{
    pub fn get_serialized_val(&self, val: T) -> Option<&Vec<f64>>{

        if let Some(pointer) = self.data_to_serialized
            .iter()
            .enumerate()
            .filter(|(_, v)| *v == &val).collect::<Vec<(usize, &T)>>().get(0) {
                return self.serialized_to_data.get(pointer.0);
        }

        None
    }

    pub fn get_val_from_serialized(&self, val: Vec<f64>) -> Option<&T>{
        if let Some(index) = val
            .iter()
            .enumerate()
            .filter(|(_,ser)| *ser == &1.0).collect::<Vec<(usize,&f64)>>().get(0){
                return self.data_to_serialized.get(index.0);
            }

        None
    }
}

pub trait Serializer<A>{
    fn serialize_response(&self) -> DataConverter<A>;
}

impl<T: Clone> Serializer<T> for Vec<T>{
    fn serialize_response(&self) -> DataConverter<T> {
        let mut data_return: DataConverter<T> = DataConverter { 
            data_to_serialized: vec![],
            serialized_to_data: vec![]
        };

        for output in 0..self.len(){
            let mut vector_vis: Vec<f64> = vec![0.0; self.len()];
            vector_vis[output] = 1.0;
            data_return.serialized_to_data.push(vector_vis);
        }

        data_return.data_to_serialized = self.to_vec();
        data_return
    }
}
