pub trait Tree<A, S> {
    fn flatten(self) -> (Vec<Box<A>>, S);
    fn unflatten(flat: &mut Vec<Box<A>>, structure: S) -> Self;
}

impl<A> Tree<A, ()> for Box<A> {
    fn flatten(self) -> (Vec<Box<A>>, ()) {
        (vec![self], ())
    }
    fn unflatten<'a>(flat: &mut Vec<Box<A>>, _: ()) -> Box<A> {
        match flat.pop() {
            None => panic!("Tried to unflatten empty vector!"),
            Some(b) => b
        }
    }
}

impl<A, S1, S2, T1: Tree<A, S1>, T2: Tree<A, S2>> Tree<A, ((usize, S1), (usize, S2))> for (Box<T1>, Box<T2>) {
    fn flatten(self) -> (Vec<Box<A>>, ((usize, S1), (usize, S2))) {
        let (mut v1, t1) = T1::flatten(*self.0);
        let (mut v2, t2) = T2::flatten(*self.1);
        let l1 = v1.len();
        let l2 = v2.len();
        v1.append(&mut v2);
        (v1, ((l1, t1), (l2, t2)))
    }
    fn unflatten(
        flat: &mut Vec<Box<A>>,
        structure: ((usize, S1), (usize, S2)),
    ) -> (Box<T1>, Box<T2>) {
        assert_eq!(flat.len(), structure.0.0 + structure.1.0, "Tried to unflatten vector into trees with incorrect sizes!");
        let mut v1 = flat.drain(0..structure.0.0).collect();
        let mut v2 = flat.drain(0..).collect();
        let t1 = T1::unflatten(&mut v1, structure.0 .1);
        let t2 = T2::unflatten(&mut v2, structure.1 .1);
        (Box::new(t1), Box::new(t2))
    }
}
