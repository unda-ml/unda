pub trait Tree<A, S> {
    fn flatten<'a, 'b: 'a>(&'a self) -> (&'b [&'a A], S);
    fn unflatten<'a, 'b: 'a>(flat: &'b [&'a A], structure: S) -> &'b Self;
}

impl<A> Tree<A, ()> for A {
    fn flatten<'a, 'b: 'a>(&'a self) -> (&'b [&'a A], ()) {
        (&[self], ())
    }
    fn unflatten<'a, 'b: 'a>(flat: &'b [&'a A], _: ()) -> &'b A {
        flat[0]
    }
}

impl<A, S1, S2, T1: Tree<A, S1>, T2: Tree<A, S2>> Tree<A, ((usize, S1), (usize, S2))> for (T1, T2) {
    fn flatten<'a, 'b: 'a>(&'a self) -> (&'b [&'a A], ((usize, S1), (usize, S2))) {
        let (mut v1, t1) = T1::flatten(&self.0);
        let (mut v2, t2) = T2::flatten(&self.1);
        let l1 = v1.len();
        let l2 = v2.len();
        let v = v1
            .into_iter()
            .chain(v2.into_iter())
            .map(|x| *x)
            .collect::<Vec<&A>>();
        (v, ((l1, t1), (l2, t2)))
    }
    fn unflatten<'a, 'b: 'a>(
        flat: &[&'a A],
        structure: ((usize, S1), (usize, S2)),
    ) -> &'b (T1, T2) {
        let (v1, v2) = flat.split_at(structure.0 .0);
        let t1 = T1::unflatten(v1, structure.0 .1);
        let t2 = T2::unflatten(v2, structure.1 .1);
        &(*t1, *t2)
    }
}
