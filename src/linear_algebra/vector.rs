use crate::random;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(PartialEq, Clone, Debug)]
/// this is a reper for `Vec<f32>`
///
/// the Vector implements many useful mathematical functions
pub struct Vector {
    vec: Vec<f32>,
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.vec())
    }
}

impl Add for Vector {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        result.add_vec(&other).unwrap();
        result
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, other: Self) {
        self.add_vec(&other).unwrap();
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.clone();
        result.sub_vec(&other).unwrap();
        result
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, other: Self) {
        self.sub_vec(&other).unwrap();
    }
}

impl Mul for Vector {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = self.clone();
        result.mul_vec(&other).unwrap();
        result
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, other: Self) {
        self.mul_vec(&other).unwrap();
    }
}

impl Div for Vector {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut result = self.clone();
        result.div_vec(&other).unwrap();
        result
    }
}

impl DivAssign for Vector {
    fn div_assign(&mut self, other: Self) {
        self.div_vec(&other).unwrap();
    }
}

impl Vector {
    /// creates a new vector
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![1., 2., 3.]);
    /// assert_eq!(vector.vec(), vec![1., 2., 3.]);
    /// ```
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
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new_one_hot(2, 5).unwrap();
    /// assert_eq!(vector.vec(), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    /// ```
    pub fn new_one_hot(index: usize, len: usize) -> Result<Self, String> {
        let mut vector = Self::new_zero(len);
        vector.set_index(index, 1.)?;
        Ok(vector)
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

    /// returns the index with the largest element in the vector
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![3., 2., 10., 4.]);
    /// assert_eq!(vector.argmax(), 2);
    /// ```
    pub fn argmax(&self) -> usize {
        let mut index = 0;
        let mut largest = self.vec[0];
        for (i, &val) in self.vec.iter().enumerate() {
            if val > largest {
                largest = val;
                index = i;
            }
        }
        index
    }

    /// retruns the mean of the vector
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![3., 2., 10., 4.]);
    /// assert_eq!(vector.mean(), 4.75);
    /// ```
    pub fn mean(&self) -> f32 {
        self.sum() / (self.len() as f32)
    }

    /// returns the angle in degrees between the 2 vectors
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![1., 0., 0.]);
    /// let vector2 = Vector::new(vec![0., 1., 0.]);
    /// assert_eq!(vector1.angle(&vector2), Ok(90.));
    /// ```
    pub fn angle(&self, other: &Vector) -> Result<f32, String> {
        Ok(self.rot(other)? * (180. / std::f32::consts::PI))
    }

    /// returns the rotaion in radians between the 2 vectors
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![1., 0., 0.]);
    /// let vector2 = Vector::new(vec![0., 1., 0.]);
    /// assert_eq!(vector1.rot(&vector2), Ok(1.5707964));
    /// ```
    pub fn rot(&self, other: &Vector) -> Result<f32, String> {
        Ok((self.dot_vec(other)? / (self.mag() * other.mag())).acos())
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
    /// assert_eq!(vector1.dist(&vector2), Ok(10.488089));
    /// ```
    pub fn dist(&self, other: &Vector) -> Result<f32, String> {
        let res = self.dist_sq(other)?;
        Ok(res.sqrt())
    }

    /// calculates the [Euclidean distance] squared between 2 vectors
    ///
    /// [Euclidean distance]:https://en.wikipedia.org/wiki/Euclidean_distance
    ///   
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![2., 7., 1.]);
    /// let vector2 = Vector::new(vec![8., 2., 8.]);
    /// assert_eq!(vector1.dist_sq(&vector2), Ok(110.0));
    /// ```
    pub fn dist_sq(&self, other: &Vector) -> Result<f32, String> {
        check_same_len(self, other)?;
        let mut res = 0.;
        for i in 0..self.vec.len() {
            res += (self.vec[i] - other.vec()[i]) * (self.vec[i] - other.vec()[i]);
        }
        Ok(res)
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
    /// assert_eq!(vector1.cross_vec(&vector2), Ok(Vector::new(vec![0., 0., 1.])));
    /// ```  
    /// note this only works with 3 dimensional vectors
    pub fn cross_vec(&self, other: &Vector) -> Result<Vector, String> {
        if self.len() != 3 || other.len() != 3 {
            Err(format!("this only works with 3 dimensional vectors"))
        } else {
            let self_0 = self.index(0)?;
            let other_0 = other.index(0)?;
            let self_1 = self.index(1)?;
            let other_1 = other.index(1)?;
            let self_2 = self.index(2)?;
            let other_2 = other.index(2)?;

            Ok(Vector::new(vec![
                self_1 * other_2 - self_2 * other_1,
                self_2 * other_0 - self_0 * other_2,
                self_0 * other_1 - self_1 * other_0,
            ]))
        }
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
    /// assert_eq!(vector1.dot_vec(&vector2), Ok(38.));
    /// ```
    /// note it panics if the vectors have not the same len  
    pub fn dot_vec(&self, other: &Vector) -> Result<f32, String> {
        check_same_len(self, other)?;
        let mut res = 0.;
        for i in 0..self.vec.len() {
            res += self.vec[i] * other.vec()[i];
        }
        Ok(res)
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
    pub fn mul_vec(&mut self, other: &Vector) -> Result<(), String> {
        check_same_len(self, other)?;
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] * other.vec[i];
        }
        Ok(())
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
    pub fn add_vec(&mut self, other: &Vector) -> Result<(), String> {
        check_same_len(self, other)?;
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] + other.vec[i];
        }
        Ok(())
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
    pub fn sub_vec(&mut self, other: &Vector) -> Result<(), String> {
        check_same_len(self, other)?;
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] - other.vec[i];
        }
        Ok(())
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
    pub fn div_vec(&mut self, other: &Vector) -> Result<(), String> {
        check_same_len(self, other)?;
        for i in 0..other.len() {
            self.vec[i] = self.vec[i] / other.vec[i];
        }
        Ok(())
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
    /// assert_eq!(vector.index(1), Ok(3.));
    /// ```  
    pub fn index(&self, index: usize) -> Result<f32, String> {
        in_bouns(index, self.len())?;
        Ok(self.vec[index])
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
    pub fn set_index(&mut self, index: usize, val: f32) -> Result<(), String> {
        in_bouns(index, self.len())?;
        self.vec[index] = val;
        Ok(())
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

fn in_bouns(index: usize, len: usize) -> Result<(), String> {
    if index > len {
        Err(format!(
            "array out of bouns max len is {} input is {}",
            index, len
        ))
    } else {
        Ok(())
    }
}

fn check_same_len(vec1: &Vector, vec2: &Vector) -> Result<(), String> {
    if vec1.len() != vec2.len() {
        Err(format!(
            "the other vector has not the same len self.len() = {}, other.len() = {}",
            vec1.len(),
            vec2.len()
        ))
    } else {
        Ok(())
    }
}

#[cfg(feature = "gpu")]
use std::mem;

#[cfg(feature = "gpu")]
impl Vector {
    /// return a vector constructed form bytes
    ///
    /// the first 4 byts give the size (as f32) the rest is the conten (also f32)
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Vector;
    /// let bytes = vec![0, 0, 64, 64, 0, 0, 0, 64, 0, 0, 128, 63, 0, 0, 192, 64];
    /// let vector = Vector::new_bytes(bytes.clone()).unwrap();
    /// assert_eq!(vector.vec(), vec![2., 1., 6.]);
    /// ```
    pub fn new_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let mut vec = Vec::new();
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "bytes.len() have to be divisibel by 4 (the len was {})",
                bytes.len()
            ));
        }

        for i in 0..(bytes.len() / 4) {
            let byte_4 = [
                bytes[i * 4 + 0],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ];
            vec.push(f32::from_ne_bytes(byte_4));
        }

        let size = vec[0];
        vec.remove(0);

        if size != vec.len() as f32 {
            return Err(format!(
                "unexpected size the expected size was {} and the actual size is {}",
                size,
                vec.len()
            ));
        }

        Ok(Self { vec })
    }

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
