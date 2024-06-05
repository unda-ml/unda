use crate::graph::Result;
/*
pub trait Tree<const N: usize, T> {
    fn flatten<'a>(&'a self) -> [&'a T; N];
    fn unflatten(flat: &[T; N]) -> Self;
}

impl<T> Tree<0, T> for () {
    fn flatten<'a>(&'a self) -> [&'a T; 0] {
        []
    }
    fn unflatten(_: &[T; 0]) -> Self {
        ()
    }
}

impl<T> Tree<2, T> for (T, T) {
    fn flatten<'a>(&'a self) -> [&'a T; 2] {
        [&self.0, &self.1]
    }
    #[inline(always)]
    fn unflatten(
        flat: &[T; 2],
    ) -> (T, T) {
        (flat[0], flat[1])
    }
}
*/
