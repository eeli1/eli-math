use crate::random;
use std::mem;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

fn check_same_len(vec1: &Vector, vec2: &Vector) {
    if vec1.vec.len() != vec2.vec.len() {
        panic!(
            "the other vector has not the same len self.len() = {}, other.len() = {}",
            vec1.vec.len(),
            vec2.vec.len()
        );
    }
}

#[derive(PartialEq, Clone, Debug)]
/// this is a reper for `Vec<f32>`
///
/// the Vector implements many useful mathematical functions
pub struct Vector {
    vec: Vec<f32>,
}

impl Add for Vector {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        result.add_vec(&other);
        result
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, other: Self) {
        self.add_vec(&other);
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.clone();
        result.sub_vec(&other);
        result
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, other: Self) {
        self.sub_vec(&other);
    }
}

impl Mul for Vector {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = self.clone();
        result.mul_vec(&other);
        result
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, other: Self) {
        self.mul_vec(&other);
    }
}

impl Div for Vector {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut result = self.clone();
        result.div_vec(&other);
        result
    }
}

impl DivAssign for Vector {
    fn div_assign(&mut self, other: Self) {
        self.div_vec(&other);
    }
}

impl Vector {
    /// creates a new vector
    pub fn new(vec: Vec<f32>) -> Self {
        Self { vec }
    }

    /// creates a new [one hot] vector
    /// 
    /// [one hot]:https://en.wikipedia.org/wiki/One-hot
    /// 
    /// ## Example
    ///
    /// ```rust
    /// let vector = Vector::new_one_hot(2, 5);
    /// assert_eq!(vector.vec(), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    /// ```
    pub fn new_one_hot(index: usize, len: usize) -> Self {
        let mut vector = Self::new_zero(len);
        vector.set_index(index, 1.);
        vector
    }

    /// generates a vector of length `len` with random values between 0 and 1
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new_rand(4);
    /// assert_eq!(vector.vec(), vec![0.69186187, 0.3494884, 0.23957491, 0.06540034]);
    /// ```
    pub fn new_rand(len: usize) -> Self {
        let mut rand = random::Random::new();
        let mut vec = Vec::new();
        for _ in 0..len {
            vec.push(rand.f32());
        }
        Self { vec }
    }

    ///  generates a vector of length `len` with all values being 0.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new_zero(4);
    /// assert_eq!(vector.vec(), vec![0., 0., 0., 0.]);
    /// ```
    pub fn new_zero(len: usize) -> Self {
        Self { vec: vec![0.; len] }
    }

    /// applyes the lamda function to each value in the vector
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![0.7, 0.2, 0.3]);
    /// let step: Box<(dyn Fn(f32) -> f32 + 'static)> = Box::new(|x: f32| -> f32 {
    ///     if x > 0.5 {
    ///         1.
    ///     } else {
    ///         0.
    ///     }
    /// });
    /// vector.apply_func(&step);
    /// assert_eq!(vector.vec(), vec![1., 0., 0.]);
    /// ```
    pub fn apply_func(&mut self, lamda: &Box<(dyn Fn(f32) -> f32 + 'static)>) {
        for i in 0..self.len() {
            self.vec[i] = lamda(self.vec[i]);
        }
    }

    /// returns the angle in degrees between the 2 vectors
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![1., 0., 0.]);
    /// let vector2 = Vector::new(vec![0., 1., 0.]);
    /// assert_eq!(vector1.angle(&vector2), 90.);
    /// ```
    pub fn angle(&self, other: &Vector) -> f32 {
        self.rot(other) * (180. / std::f32::consts::PI)
    }

    /// returns the rotaion in radians between the 2 vectors
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![1., 0., 0.]);
    /// let vector2 = Vector::new(vec![0., 1., 0.]);
    /// assert_eq!(vector1.rot(&vector2), 1.5707964);
    /// ```
    pub fn rot(&self, other: &Vector) -> f32 {
        (self.dot_vec(other) / (self.mag() * other.mag())).acos()
    }

    /// returns the magnetude of the vector
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![2., 3., 5.]);
    /// assert_eq!(vector.mag(), ((2. * 2. + 3. * 3. + 5. * 5.) as f32).sqrt());
    /// ```
    pub fn mag(&self) -> f32 {
        let sqr_sum: f32 = self.vec.iter().map(|v| v * v).sum();
        sqr_sum.sqrt()
    }

    /// sets the magnetude of the vector to a spicific value
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.set_mag(4.);
    /// assert_eq!(vector.mag(), 4.);
    /// ```
    pub fn set_mag(&mut self, mag: f32) {
        self.mul_scalar(&(mag / self.mag()));
    }

    /// calculates the [Euclidean distance] between 2 vectors
    ///
    /// [Euclidean distance]:https://en.wikipedia.org/wiki/Euclidean_distance
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![2., 7., 1.]);
    /// let vector2 = Vector::new(vec![8., 2., 8.]);
    /// assert_eq!(vector1.dist(&vector2), 10.488089);
    /// ```
    pub fn dist(&self, other: &Vector) -> f32 {
        check_same_len(self, other);
        let mut res = 0.;
        for i in 0..self.vec.len() {
            res += (self.vec[i] - other.vec()[i]) * (self.vec[i] - other.vec()[i]);
        }
        res.sqrt()
    }

    /// Limit the magnitude of this vector to the value used for the `max` parameter
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.limit(2.);
    /// assert_eq!(vector.mag(), 2.);
    ///
    /// vector.limit(3.);
    /// assert_eq!(vector.mag(), 2.);
    /// ```
    pub fn limit(&mut self, max: f32) {
        if self.mag() > max {
            self.set_mag(max);
        }
    }

    /// normalizes the vetor same dirction but the magnetude is 1
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.unit();
    /// assert_eq!(vector.mag(), 1.);
    /// ```
    pub fn unit(&mut self) {
        self.div_scalar(&self.mag());
    }

    // vector math

    /// this returns the [cross product]
    ///
    /// [cross product]: https://en.wikipedia.org/wiki/Cross_product
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![1., 0., 0.]);
    /// let vector2 = Vector::new(vec![0., 1., 0.]);
    /// assert_eq!(vector1.cross_vec(&vector2), Vector::new(vec![0., 0., 1.]));
    /// ```  
    /// note this only works with 3 dimensional vectors
    pub fn cross_vec(&self, other: &Vector) -> Vector {
        if self.len() != 3 || other.len() != 3 {
            panic!("this only works with 3 dimensional vectors");
        }

        Vector::new(vec![
            self.index(1) * other.index(2) - self.index(2) * other.index(1),
            self.index(2) * other.index(0) - self.index(0) * other.index(2),
            self.index(0) * other.index(1) - self.index(1) * other.index(0),
        ])
    }

    /// returns the [dot product]
    ///
    /// [dot product]: https://en.wikipedia.org/wiki/Dot_product
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![2., 7., 1.]);    
    /// let vector2 = Vector::new(vec![8., 2., 8.]);
    /// assert_eq!(vector1.dot_vec(&vector2), 38.);
    /// ```
    /// note it panics if the vectors have not the same len  
    pub fn dot_vec(&self, other: &Vector) -> f32 {
        check_same_len(self, other);
        let mut res = 0.;
        for i in 0..self.vec.len() {
            res += self.vec[i] * other.vec()[i];
        }
        res
    }

    /// multiplies each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.mul_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. * 3., 2. * 1., 3. * 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn mul_vec(&mut self, other: &Vector) {
        check_same_len(self, other);
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] * other.vec[i];
        }
    }

    /// adds each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.add_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. + 3., 2. + 1., 3. + 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn add_vec(&mut self, other: &Vector) {
        check_same_len(self, other);
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] + other.vec[i];
        }
    }

    /// subtracts each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.sub_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. - 3., 2. - 1., 3. - 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn sub_vec(&mut self, other: &Vector) {
        check_same_len(self, other);
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] - other.vec[i];
        }
    }

    /// divides each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.div_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. / 3., 2. / 1., 3. / 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn div_vec(&mut self, other: &Vector) {
        check_same_len(self, other);
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] / other.vec[i];
        }
    }

    /// multiplies each component from the vector with a scalar value and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.mul_scalar(&2.);
    /// assert_eq!(vector, Vector::new(vec![2. * 2., 3. * 2., 5. * 2.]));
    /// ```
    pub fn mul_scalar(&mut self, scalar: &f32) {
        self.vec = self.vec.iter().map(|v| v * scalar).collect();
    }

    /// divides each component from the vector with a scalar value and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.div_scalar(&2.);
    /// assert_eq!(vector, Vector::new(vec![2. / 2., 3. / 2., 5. / 2.]));
    /// ```
    pub fn div_scalar(&mut self, scalar: &f32) {
        self.vec = self.vec.iter().map(|v| v / scalar).collect();
    }

    /// adds each component from the vector with a scalar value and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.add_scalar(&2.);
    /// assert_eq!(vector, Vector::new(vec![2. + 2., 3. + 2., 5. + 2.]));
    /// ```
    pub fn add_scalar(&mut self, scalar: &f32) {
        self.vec = self.vec.iter().map(|v| v + scalar).collect();
    }

    /// subtracts each component from the vector with a scalar value and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.sub_scalar(&2.);
    /// assert_eq!(vector, Vector::new(vec![2. - 2., 3. - 2., 5. - 2.]));
    /// ```
    pub fn sub_scalar(&mut self, scalar: &f32) {
        self.vec = self.vec.iter().map(|v| v - scalar).collect();
    }

    /// getter for the internal Vec<f32> representation
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 1., 6.]);
    /// assert_eq!(vector.vec(), vec![2., 1., 6.]);
    /// ```
    pub fn vec(&self) -> Vec<f32> {
        self.vec.clone()
    }

    /// the returns the length of the vec
    /// or in mathematical terms the [dimensions] of the vector
    ///
    /// [dimensions]: https://en.wikipedia.org/wiki/Dimension_(vector_space)
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![4., 3., 5.]);    
    /// assert_eq!(vector.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    /// returns the value at the given index
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![1., 3., 6.]);
    /// assert_eq!(vector.index(1), 3.);
    /// ```  
    pub fn index(&self, index: usize) -> f32 {
        if index > self.len() {
            panic!(
                "array out of bouns max len is {} input is {}",
                index,
                self.len()
            );
        }
        self.vec[index]
    }

    /// sets the value of the vector at the specifide index
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.set_index(1, 10.);
    /// assert_eq!(vector.vec(), vec![2.0, 10.0, 5.0]);
    /// ```
    pub fn set_index(&mut self, index: usize, val: f32) {
        if index > self.len() {
            panic!(
                "array out of bouns max len is {} input is {}",
                index,
                self.len()
            );
        }
        self.vec[index] = val;
    }

    /// returns the sum of the elements
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![3., 1., 3., 1.]);
    /// assert_eq!(vector.sum(), 8.);
    /// ```
    pub fn sum(&self) -> f32 {
        self.vec.iter().sum()
    }
}

#[cfg(feature = "gpu")]
impl Vector {
    /// this return a vector of bytes representing the vector
    ///
    /// this is useful for the *GPU* because the interface only uses bytes
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![2., 1., 6.]);
    /// assert_eq!(
    ///    vector.bytes(),
    ///    vec![0, 0, 64, 64, 0, 0, 0, 64, 0, 0, 128, 63, 0, 0, 192, 64]
    /// );
    /// ```
    /// note the fist `f32` is the len of the vector
    pub fn bytes(&self) -> Vec<u8> {
        let size = (1 + self.vec.len()) * mem::size_of::<f32>();
        let mut bytes = Vec::<u8>::with_capacity(size);

        let push_f32_bytes = |num: f32, bytes: &mut Vec<u8>| {
            for b in num.to_ne_bytes().to_vec() {
                bytes.push(b);
            }
        };

        push_f32_bytes(self.vec.len() as f32, &mut bytes);

        self.vec
            .iter()
            .for_each(|&val| push_f32_bytes(val, &mut bytes));
        bytes
    }
}
