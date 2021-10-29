pub mod linear_algebra;
pub mod misc;
pub mod random;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn it_works() {
        assert_eq!(4, 2 + 2);
    }

    
}
