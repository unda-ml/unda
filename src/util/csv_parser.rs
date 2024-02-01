use std::error::Error;

pub trait CSVParse{
    fn parse_elem(&mut self, path: &str) -> Result<(), Box<dyn Error>>;
}

impl CSVParse for Vec<Vec<String>> {
    fn parse_elem(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let reader = csv::Reader::from_path(path)?;
        self.clear();
        for result in reader.into_records() {
            self.push(result?.deserialize::<Vec<String>>(None)?);
        }
        Ok(())
    }
}

impl CSVParse for Vec<Vec<f32>> {
    fn parse_elem(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.clear();
        let mut parse_str: Vec<Vec<String>> = vec![];
        parse_str.parse_elem(path)?;

        for row in parse_str.iter(){
            let mut row_f32: Vec<f32> = vec![];
            for col in row.iter(){
                if let Ok(num) = col.parse::<f32>(){
                    row_f32.push(num);
                }else{
                    row_f32.push(0.0);
                }
            }
            self.push(row_f32);
        }

        Ok(())
    }
}
