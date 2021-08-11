#[derive(PartialEq, Clone, Copy, Debug)]
struct Xorshift32State {
    a: u32,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Xorshift64State {
    a: u64,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Xorshift128State {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Splitmix64State {
    s: u64,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct XorwowState {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    counter: u32,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Xorshift1024sState {
    array: [u64; 16],
    index: usize,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Xorshift128pState {
    a: u64,
    b: u64,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Xoshiro256ssState {
    s: [u64; 4],
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Xoshiro256pState {
    s: [u64; 4],
}

/// this stores all the seeds and stats for the random number generator with xor
pub struct Xorshift {
    xorshift32_state: Xorshift32State,
    xorshift64_state: Xorshift64State,
    xorshift128_state: Xorshift128State,
}

impl Xorshift {
    /// initialising seeds for the random number generator with seeds
    pub fn new() -> Self {
        Xorshift {
            xorshift32_state: Xorshift32State { a: 314159265 },
            xorshift64_state: Xorshift64State {
                a: 88172645463325252,
            },
            xorshift128_state: Xorshift128State {
                a: 123456789,
                b: 362436069,
                c: 521288629,
                d: 88675123,
            },
        }
    }

    /// generates a u32 random number using the Algorithm "xor" (from p. 4 of Marsaglia, "Xorshift RNGs")
    /// for more informaiton go to the [`wiki`]
    ///
    /// [`wiki`]: https://en.wikipedia.org/wiki/Xorshift
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::random::Xorshift;
    /// let mut xorshift = Xorshift::new();
    /// assert_eq!(xorshift.xorshift32(), 2971524119);
    /// ```
    pub fn xorshift32(&mut self) -> u32 {
        /*  */
        let mut state = self.xorshift32_state;
        let mut x = state.a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state.a = x;
        x
    }

    /// generates a u64 random number using the Algorithm "xor" (from p. 4 of Marsaglia, "Xorshift RNGs")
    /// for more informaiton go to the [`wiki`]
    ///
    /// [`wiki`]: https://en.wikipedia.org/wiki/Xorshift
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::random::Xorshift;
    /// let mut xorshift = Xorshift::new();
    /// assert_eq!(xorshift.xorshift64(), 8748534153485358512);
    /// ```
    pub fn xorshift64(&mut self) -> u64 {
        let mut state = self.xorshift64_state;

        let mut x = state.a;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.a = x;
        x
    }

    /// generates a u32 random number using the Algorithm "xor" (from p. 5 of Marsaglia, "Xorshift RNGs")
    /// for more informaiton go to the [`wiki`]
    ///
    /// [`wiki`]: https://en.wikipedia.org/wiki/Xorshift
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::random::Xorshift;
    /// let mut xorshift = Xorshift::new();
    /// assert_eq!(xorshift.xorshift128(), 1254528582);
    /// ```
    pub fn xorshift128(&mut self) -> u32 {
        let mut state = self.xorshift128_state;

        let mut t = state.d;

        let s = state.a;
        state.d = state.c;
        state.c = state.b;
        state.b = s;

        t ^= t << 11;
        t ^= t >> 8;
        state.a = t ^ s ^ (s >> 19);
        state.a
    }

    fn splitmix64(&mut self, state: &mut Splitmix64State) -> u64 {
        state.s += 0x9E3779B97f4A7C15;
        let mut result = state.s;
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        return result ^ (result >> 31);
    }

    // as an example; one could do this same thing for any of the other generators
    fn xorshift128_init(&mut self, seed: u64) -> Xorshift128State {
        let mut smstate = Splitmix64State { s: seed };

        let tmp = self.splitmix64(&mut smstate);
        let a = tmp as u32;
        let b = (tmp >> 32) as u32;

        let tmp = self.splitmix64(&mut smstate);
        let c = tmp as u32;
        let d = (tmp >> 32) as u32;

        Xorshift128State { a, b, c, d }
    }

    /* The state array must be initialized to not be all zero in the first four words */
    fn xorwow(state: &mut XorwowState) -> u32 {
        /* Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" */
        let mut t = state.e;
        let s = state.a;
        state.e = state.d;
        state.d = state.c;
        state.c = state.b;
        state.b = s;
        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        state.a = t;
        state.counter += 362437;
        return t + state.counter;
    }

    /* The state must be seeded so that there is at least one non-zero element in array */
    fn xorshift1024s(state: &mut Xorshift1024sState) -> u64 {
        let mut index = state.index + 1;
        let s = state.array[index - 1];
        index &= 15;
        let mut t = state.array[index];
        t ^= t << 31; // a
        t ^= t >> 11; // b
        t ^= s ^ (s >> 30); // c
        state.array[index] = t;
        state.index = index;
        return t * 1181783497276652981;
    }

    /* The state must be seeded so that it is not all zero */
    fn xorshift128p(state: &mut Xorshift128pState) -> u64 {
        let mut t = state.a;
        let s = state.b;
        state.a = s;
        t ^= t << 23; // a
        t ^= t >> 17; // b
        t ^= s ^ (s >> 26); // c
        state.b = t;
        return t + s;
    }

    fn rol64(&self, x: u64, k: usize) -> u64 {
        return (x << k) | (x >> (64 - k));
    }

    fn xoshiro256ss(&mut self, state: &mut Xoshiro256ssState) -> u64 {
        let mut s = state.s;
        let result = self.rol64(s[1] * 5, 7) * 9;
        let t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = self.rol64(s[3], 45);

        return result;
    }

    fn xoshiro256p(&mut self, state: &mut Xoshiro256pState) -> u64 {
        let mut s = state.s;
        let result = s[0] + s[3];
        let t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = self.rol64(s[3], 45);

        return result;
    }
}
