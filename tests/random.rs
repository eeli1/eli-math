#[cfg(test)]
mod tests {
    use eli_math::random::*;

    #[test]
    fn xorshift32() {
        let mut xorshift = Xorshift::new();
        assert_eq!(xorshift.xorshift32(), 2971524119);
    }

    #[test]
    fn xorshift64() {
        let mut xorshift = Xorshift::new();
        assert_eq!(xorshift.xorshift64(), 8748534153485358512);
    }

    #[test]
    fn xorshift128() {
        let mut xorshift = Xorshift::new();
        assert_eq!(xorshift.xorshift128(), 1254528582);
    }
}
