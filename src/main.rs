#[path = "linear_algebra/vector.rs"]
mod vector;

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test() {
        assert_eq!(4, 2 + 2);
    }
}
