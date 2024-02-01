use itertools::Itertools;

pub fn to_categorical(data: Vec<usize>) -> Vec<Vec<f32>> {
    let categorized_index = data.iter().unique().sorted().collect::<Vec<&usize>>();
    //println!("{:?}", categorized_index);
    
    let mut res = vec![];
        
    data.iter().for_each(|val| {
        let mut categorized_val = vec![0.0; categorized_index.len()];
        let index = categorized_index.iter().enumerate().find(|c| c.1 == &val).unwrap().0; 
        categorized_val[index] = 1.0;

        res.push(categorized_val);
        //println!("{:?}", res[res.len()-1]);
        //println!("{}", res.len());
    });
    res
}
