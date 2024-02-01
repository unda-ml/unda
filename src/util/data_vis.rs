use crate::core::network::Network;
use plotters::prelude::*;

impl Network {
    pub fn plot_layer_loss(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>{
        let data = self.get_layer_loss();
        let root = BitMapBackend::new(path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Layer Loss on Neural Network", ("Arial", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-0.1f32..data.len() as f32, -0.1f32..data.iter().max_by(|l, j| l.1.total_cmp(&j.1)).unwrap().1 + 1.0)?;

        chart.configure_mesh().draw()?;
        
        chart.configure_series_labels()
        .border_style(&BLACK)
        .label_font("Arial")
        .draw()?;

        chart.draw_series(LineSeries::new(data.clone(), &BLACK))?;
        chart.draw_series(PointSeries::of_element(data, 2, &BLACK, &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
            + Text::new(format!("({}, {:.2})", c.0 as i32, c.1), (10, 0), ("sans-serif", 10).into_font());
        }))?;

        root.present()?;

        Ok(())
    }
    pub fn plot_loss_history(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = self.get_loss_history();
        let root = BitMapBackend::new(path, (1280, 960)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Loss History on Neural Network", ("Arial", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-0.1f32..data.len() as f32, -0.1f32..data.iter().max_by(|l, j| l.total_cmp(&j)).unwrap() + 1.0)?;

        chart.configure_mesh().draw()?;
        
        chart.configure_series_labels()
        .border_style(&BLACK)
        .label_font("Arial")
        .draw()?;

        let mut data_ord: Vec<(f32, f32)> = vec![];

        for i in 0..data.len(){
            data_ord.push((i as f32, data[i]));
        }

        chart.draw_series(LineSeries::new(data_ord.clone(), &BLACK))?;
        chart.draw_series(PointSeries::of_element(data_ord, 2, &BLACK, &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
            + Text::new(format!("({}, {:.2})", c.0 as i32, c.1), (10, 0), ("sans-serif", 10).into_font());
        }))?;

        root.present()?;

        Ok(())
    }
}
