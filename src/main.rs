mod vector;
mod matrix;


fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn it_works() {
        assert_eq!(4, 2 + 2);
    }
}
