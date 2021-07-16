use std::mem;

#[derive(PartialEq, Clone, Debug)]
/// this is a reper for `Vec<f32>`
///
/// the Vector implements many useful mathematical functions
pub struct Vector {
    vec: Vec<f32>,
}

impl Vector {
    /// creates a new vector
    pub fn new(vec: Vec<f32>) -> Self {
        Self { vec }
    }

    /// this returns the [`cross product`]
    ///
    /// [`cross product`]: https://en.wikipedia.org/wiki/Cross_product
    ///
    /// ## Example
    ///
    /// ```rust
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

    /// returns the value at the given index
    ///
    /// ## Example
    ///
    /// ```rust
    /// let vector = Vector::new(vec![1., 3., 6.]);
    /// assert_eq!(vector.index(1), 3.);
    /// ```  
    pub fn index(&self, index: usize) -> f32 {
        self.vec()[index]
    }

    /// returns the angle in degrees between the 2 vectors
    ///   
    /// ## Example
    ///
    /// ```rust
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
    /// let vector = Vector::new(vec![2., 3., 5.]);
    /// assert_eq!(vector.mag(), (2. * 2. + 3. * 3. + 5. * 5.).sqrt());
    /// ```
    pub fn mag(&self) -> f32 {
        let sqr_sum: f32 = self.vec.iter().map(|v| v * v).sum();
        sqr_sum.sqrt()
    }

    /// normalizes the vetor same dirction but the magnetude is 1
    ///
    /// ## Example
    ///
    /// ```rust
    /// let vector = Vector::new(vec![2., 3., 5.]);
    /// vector.unit();
    /// assert_eq!(vector.mag(), 1);
    /// ```
    pub fn unit(&mut self) {
        self.div_scalar(&self.mag());
    }

    /// the returns the length of the vec
    /// or in mathematical terms the [`dimensions`] of the vector
    ///
    /// [`dimensions`]: https://en.wikipedia.org/wiki/Dimension_(vector_space)
    ///
    /// ## Example
    ///
    /// ```rust
    /// let vector = Vector::new(vec![4., 3., 5.]);    
    /// assert_eq!(vector.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    /// returns the [`dot product`]
    ///
    /// [`dot product`]: https://en.wikipedia.org/wiki/Dot_product
    ///
    /// ## Example
    ///
    /// ```rust
    /// let vector1 = Vector::new(vec![2., 7., 1.]);    
    /// let vector2 = Vector::new(vec![8., 2., 8.]);
    /// assert_eq!(vector1.dot_vec(vector2), 38.);
    /// ```
    /// note it panics if the vectors have not the same len  
    pub fn dot_vec(&self, other: &Vector) -> f32 {
        if self.vec.len() == other.len() {
            let mut res = 0.;
            for i in 0..self.vec.len() {
                res += self.vec[i] * other.vec()[i];
            }
            res
        } else {
            panic!(
                "the other vector has not the same len self.len() = {}, other.len() = {}",
                self.len(),
                other.len()
            );
        }
    }

    /// multiplies each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.mul_vec(vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. * 3., 2. * 1., 3. * 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn mul_vec(&mut self, other: &Vector) {
        if self.vec.len() == other.len() {
            let mut vec = Vec::with_capacity(self.vec.len());
            for i in 0..self.vec.len() {
                vec.push(self.vec[i] * other.vec()[i]);
            }
            self.vec = vec;
        } else {
            panic!(
                "the other vector has not the same len self.len() = {}, other.len() = {}",
                self.len(),
                other.len()
            );
        }
    }

    /// adds each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.mul_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. + 3., 2. + 1., 3. + 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn add_vec(&mut self, other: &Vector) {
        if self.vec.len() == other.len() {
            let mut vec = Vec::with_capacity(self.vec.len());
            for i in 0..self.vec.len() {
                vec.push(self.vec[i] + other.vec()[i]);
            }
            self.vec = vec;
        } else {
            panic!(
                "the other vector has not the same len self.len() = {}, other.len() = {}",
                self.len(),
                other.len()
            );
        }
    }

    /// subtracts each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.mul_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. - 3., 2. - 1., 3. - 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn sub_vec(&mut self, other: &Vector) {
        if self.vec.len() == other.len() {
            let mut vec = Vec::with_capacity(self.vec.len());
            for i in 0..self.vec.len() {
                vec.push(self.vec[i] - other.vec()[i]);
            }
            self.vec = vec;
        } else {
            panic!(
                "the other vector has not the same len self.len() = {}, other.len() = {}",
                self.len(),
                other.len()
            );
        }
    }

    /// divides each component from the vector with the component of the other vector and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.div_vec(&vector2);
    /// assert_eq!(vector1, Vector::new(vec![0. / 3., 2. / 1., 3. / 3.]));
    /// ```
    /// note it panics if the vectors have not the same len
    pub fn div_vec(&mut self, other: &Vector) {
        if self.vec.len() == other.len() {
            let mut vec = Vec::with_capacity(self.vec.len());
            for i in 0..self.vec.len() {
                vec.push(self.vec[i] / other.vec()[i]);
            }
            self.vec = vec;
        } else {
            panic!(
                "the other vector has not the same len self.len() = {}, other.len() = {}",
                self.len(),
                other.len()
            );
        }
    }

    /// multiplies each component from the vector with a scalar value and stors the result in this vector   
    ///
    /// ## Example
    ///
    /// ```rust
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
    /// let mut vector = Vector::new(vec![2., 1., 6.]);
    /// assert_eq!(vector.vec(), vec![2., 1., 6.]);
    /// ```
    pub fn vec(&self) -> Vec<f32> {
        self.vec.clone()
    }

    /// this return a vector of bytes representing the vector
    ///
    /// this is useful for the *GPU* because the interface only uses bytes
    ///
    /// ## Example
    ///
    /// ```rust
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index() {
        let vector = Vector::new(vec![1., 3., 6.]);
        assert_eq!(vector.index(1), 3.);
    }

    #[test]
    fn cross_vec() {
        let vector1 = Vector::new(vec![1., 0., 0.]);
        let vector2 = Vector::new(vec![0., 1., 0.]);
        assert_eq!(vector1.cross_vec(&vector2), Vector::new(vec![0., 0., 1.]));
    }

    #[test]
    #[should_panic(expected = "this only works with 3 dimensional vectors")]
    fn cross_vec_panic() {
        let vector1 = Vector::new(vec![1., 0., 0., 2.]);
        let vector2 = Vector::new(vec![0., 1., 0., 2.]);
        let _ = vector1.cross_vec(&vector2);
    }

    #[test]
    fn angle() {
        let vector1 = Vector::new(vec![1., 0., 0.]);
        let vector2 = Vector::new(vec![0., 1., 0.]);
        assert_eq!(vector1.angle(&vector2), 90.);
    }

    #[test]
    fn rot() {
        let vector1 = Vector::new(vec![1., 0., 0.]);
        let vector2 = Vector::new(vec![0., 1., 0.]);
        assert_eq!(vector1.rot(&vector2), 1.5707964);
    }

    #[test]
    fn bytes() {
        let vector = Vector::new(vec![2., 1., 6.]);
        assert_eq!(
            vector.bytes(),
            vec![0, 0, 64, 64, 0, 0, 0, 64, 0, 0, 128, 63, 0, 0, 192, 64]
        );
    }

    #[test]
    fn vec() {
        let vector = Vector::new(vec![2., 1., 6.]);
        assert_eq!(vector.vec(), vec![2., 1., 6.]);
    }

    #[test]
    fn mul_scalar() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.mul_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. * 2., 3. * 2., 5. * 2.]));
    }

    #[test]
    fn div_scalar() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.div_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. / 2., 3. / 2., 5. / 2.]));
    }

    #[test]
    fn add_scalar() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.add_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. + 2., 3. + 2., 5. + 2.]));
    }

    #[test]
    fn sub_scalar() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.sub_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. - 2., 3. - 2., 5. - 2.]));
    }

    #[test]
    fn mag() {
        let vector = Vector::new(vec![2., 3., 5.]);
        assert_eq!(vector.mag(), ((2 * 2 + 3 * 3 + 5 * 5) as f32).sqrt());
    }

    #[test]
    fn unit() {
        let mut vector = Vector::new(vec![3., 3., 5.]);
        let dot = vector.mag();
        let temp = vector.clone();
        vector.unit();
        assert_eq!(vector.mag(), 1.);

        vector.mul_scalar(&dot);
        assert_eq!(temp, vector);
    }

    #[test]
    fn len() {
        let vector = Vector::new(vec![4., 3., 5.]);
        assert_eq!(vector.len(), 3);
    }

    #[test]
    fn dot_vec() {
        let vector1 = Vector::new(vec![2., 7., 1.]);
        let vector2 = Vector::new(vec![8., 2., 8.]);
        assert_eq!(vector1.dot_vec(&vector2), 38.);
    }

    #[test]
    #[should_panic(
        expected = "the other vector has not the same len self.len() = 3, other.len() = 4"
    )]
    fn dot_vec_panic() {
        let vector1 = Vector::new(vec![2., 7., 1.]);
        let vector2 = Vector::new(vec![8., 2., 8., 1.]);
        vector1.dot_vec(&vector2);
    }

    #[test]
    fn mul_vec() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3.]);
        vector1.mul_vec(&vector2);
        assert_eq!(vector1, Vector::new(vec![0. * 3., 2. * 1., 3. * 3.]));
    }

    #[test]
    #[should_panic(
        expected = "the other vector has not the same len self.len() = 3, other.len() = 4"
    )]
    fn mul_vec_panic() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3., 1.]);
        vector1.mul_vec(&vector2);
    }

    #[test]
    fn add_vec() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3.]);
        vector1.add_vec(&vector2);
        assert_eq!(vector1, Vector::new(vec![0. + 3., 2. + 1., 3. + 3.]));
    }

    #[test]
    #[should_panic(
        expected = "the other vector has not the same len self.len() = 3, other.len() = 4"
    )]
    fn add_vec_panic() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3., 1.]);
        vector1.add_vec(&vector2);
    }

    #[test]
    fn sub_vec() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3.]);
        vector1.sub_vec(&vector2);
        assert_eq!(vector1, Vector::new(vec![0. - 3., 2. - 1., 3. - 3.]));
    }

    #[test]
    #[should_panic(
        expected = "the other vector has not the same len self.len() = 3, other.len() = 4"
    )]
    fn sub_vec_panic() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3., 1.]);
        vector1.sub_vec(&vector2);
    }

    #[test]
    fn div_vec() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3.]);
        vector1.div_vec(&vector2);
        assert_eq!(vector1, Vector::new(vec![0. / 3., 2. / 1., 3. / 3.]));
    }

    #[test]
    #[should_panic(
        expected = "the other vector has not the same len self.len() = 3, other.len() = 4"
    )]
    fn div_vec_panic() {
        let mut vector1 = Vector::new(vec![0., 2., 3.]);
        let vector2 = Vector::new(vec![3., 1., 3., 1.]);
        vector1.mul_vec(&vector2);
    }
}
