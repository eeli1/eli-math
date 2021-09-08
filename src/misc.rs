use std::mem;

/// implementation for the [`fast inverse square root`] 1/sqrt(x)
///
/// [`fast inverse square root`]:https://en.wikipedia.org/wiki/Fast_inverse_square_root
///
///
/// ## Example
///
/// ```rust
/// use math::misc::q_rsqrt;
/// macro_rules! assert_delta {
///   ($x:expr, $y:expr, $d:expr) => {
///     if !($x - $y < $d || $y - $x < $d) { panic!(); }
///   }
/// }
/// assert_delta!(q_rsqrt(64.), 0.125, 0.001);
/// ```
///
/// note this isn't that useful because you need a lot of calculation to feal the difference
pub fn q_rsqrt(number: f32) -> f32 {
    let mut i: i32;
    let x2: f32;
    let mut y: f32;
    const THREEHALVES: f32 = 1.5;

    x2 = number * 0.5;
    y = number;

    // Evil floating point bit level hacking
    i = unsafe { mem::transmute(y) };

    // What the fuck?
    i = 0x5f3759df - (i >> 1);
    y = unsafe { mem::transmute(i) };

    // 1st iteration
    y = y * (THREEHALVES - (x2 * y * y));

    // 2nd iteration, this can be removed
    // y = y * (THREEHALVES - (x2 * y * y));

    return y;
}
