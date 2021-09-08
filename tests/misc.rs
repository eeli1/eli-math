#[cfg(test)]
mod tests {
    use math::misc::*;

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if !($x - $y < $d || $y - $x < $d) {
                panic!();
            }
        };
    }

    #[test]
    fn test_q_rsqrt() {
        assert_delta!(q_rsqrt(64.), 0.125, 0.001);
    }
}
