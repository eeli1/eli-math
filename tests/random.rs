#[cfg(test)]
mod tests {
    use eli_math::random::*;

    #[test]
    fn xorshift32() {
        let mut xorshift = Xorshift::new();
        assert_eq!(xorshift.xorshift32(), 2971524119);
        assert_eq!(xorshift.xorshift32(), 1501041240);
        assert_eq!(xorshift.xorshift32(), 1028966369);
    }

    #[test]
    fn xorshift64() {
        let mut xorshift = Xorshift::new();
        assert_eq!(xorshift.xorshift64(), 8748534153485358512);
        assert_eq!(xorshift.xorshift64(), 3040900993826735515);
        assert_eq!(xorshift.xorshift64(), 3453997556048239312);
    }

    #[test]
    fn xorshift128() {
        let mut xorshift = Xorshift::new();
        assert_eq!(xorshift.xorshift128(), 1138687896200805812714748853);
        assert_eq!(xorshift.xorshift128(), 11570983842918995070312666597);
        assert_eq!(xorshift.xorshift128(), 30411594405250797674514074901);
    }

    #[test]
    fn f32() {
        let mut rand = Random::new();
        assert_eq!(rand.f32(), 0.69186187);
        assert_eq!(rand.f32(), 0.3494884);
        assert_eq!(rand.f32(), 0.23957491);
        assert_eq!(rand.f32(), 0.06540034);
        assert_eq!(rand.f32(), 0.5443042);
        assert_eq!(rand.f32(), 0.013656098);
    }

    #[test]
    fn f64() {
        let mut rand = Random::new();
        assert_eq!(rand.f64(), 0.47425898676362294);
        assert_eq!(rand.f64(), 0.1648475731910138);
        assert_eq!(rand.f64(), 0.18724158270135619);
        assert_eq!(rand.f64(), 0.8907660227879807);
        assert_eq!(rand.f64(), 0.44477898328394805);
        assert_eq!(rand.f64(), 0.9650074960886351);
    }
}
