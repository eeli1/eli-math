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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
    /// let vector = Vector::new(vec![2., 3., 5.]);
    /// assert_eq!(vector.mag(), ((2. * 2. + 3. * 3. + 5. * 5.) as f32).sqrt());
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
    /// use eli_math::linear_algebra::Vector;
    /// let mut vector = Vector::new(vec![2., 3., 5.]);
    /// vector.unit();
    /// assert_eq!(vector.mag(), 1.);
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![2., 7., 1.]);    
    /// let vector2 = Vector::new(vec![8., 2., 8.]);
    /// assert_eq!(vector1.dot_vec(&vector2), 38.);
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
    /// use eli_math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.mul_vec(&vector2);
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
    /// use eli_math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.add_vec(&vector2);
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
    /// use eli_math::linear_algebra::Vector;
    /// let mut vector1 = Vector::new(vec![0., 2., 3.]);
    /// let vector2 = Vector::new(vec![3., 1., 3.]);
    /// vector1.sub_vec(&vector2);
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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
    /// use eli_math::linear_algebra::Vector;
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

// ----------------------------------------------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------------------------------------------- //



#[derive(PartialEq, Clone, Debug)]
pub struct Matrix {
    cols: usize,
    rows: usize,
    matrix_flatt: Vec<f32>,
    is_transpose: bool,
}

impl Matrix {
    /// this return a vector of bytes representing the matrix
    ///
    /// this is useful for the *GPU* because the interface only uses bytes
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![2., 3.], vec![7., 4.]]);
    /// assert_eq!(
    ///     matrix.bytes(),
    ///     vec![0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 224, 64, 0, 0, 128, 64]
    /// );
    /// ```
    /// note the fist and seconde `f32` is the rows and cols of the matrix
    pub fn bytes(&self) -> Vec<u8> {
        let size = (2 + self.matrix_flatt.len()) * mem::size_of::<f32>();
        let mut bytes = Vec::<u8>::with_capacity(size);

        let push_f32_bytes = |num: f32, bytes: &mut Vec<u8>| {
            for b in num.to_ne_bytes().to_vec() {
                bytes.push(b);
            }
        };

        push_f32_bytes(self.rows() as f32, &mut bytes);
        push_f32_bytes(self.cols() as f32, &mut bytes);

        self.matrix_flatt()
            .iter()
            .for_each(|&val| push_f32_bytes(val, &mut bytes));
        bytes
    }

    /// getter for the internal matrix_flatt representation
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
    /// assert_eq!(matrix.matrix_flatt(), vec![2., 3., 5., 7., 1., 4.]);
    /// ```
    pub fn matrix_flatt(&self) -> Vec<f32> {
        if self.is_transpose {
            let mut matrix_flatt = Vec::with_capacity(self.cols * self.rows);
            for i in 0..self.rows {
                for val in self.row(i).vec() {
                    matrix_flatt.push(val);
                }
            }
            matrix_flatt
        } else {
            self.matrix_flatt.clone()
        }
    }

    /// return the length of the columns
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.cols(), 2);
    /// ```
    pub fn cols(&self) -> usize {
        if self.is_transpose {
            self.rows
        } else {
            self.cols
        }
    }

    /// return the length of the rows
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.rows(), 3);
    /// ```
    pub fn rows(&self) -> usize {
        if self.is_transpose {
            self.cols
        } else {
            self.rows
        }
    }

    /// getter for the transpose
    pub fn is_transpose(&self) -> bool {
        self.is_transpose
    }

    /// converts 2d vec in to matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// ```
    /// crates matrix that looks like this:
    /// 
    /// [3.0, 2.0, 4.0]
    /// [4.0, 5.0, 6.0]
    /// 
    pub fn new(vec: Vec<Vec<f32>>) -> Self {
        let cols = vec.len();
        let rows = vec[0].len();

        let mut flatt: Vec<f32> = Vec::with_capacity(cols * rows);

        vec.iter().for_each(|col| {
            if col.len() != rows {
                panic!("wrong row shape expected {}, got {}", rows, col.len())
            }
            col.iter().for_each(|&x| flatt.push(x))
        });

        Self {
            cols: cols,
            rows: rows,
            matrix_flatt: flatt,
            is_transpose: false,
        }
    }

    /// [`transposes`] matrix flips rows and cols
    ///
    /// [`transposes`]: https://en.wikipedia.org/wiki/Transpose
    pub fn transpose(&mut self) {
        self.is_transpose = !self.is_transpose;
    }

    fn get_col(&self, col: usize) -> Vector {
        if self.cols < col + 1 {
            panic!("index out of bounds max col {}", self.cols - 1)
        }

        let mut result: Vec<f32> = Vec::with_capacity(self.rows);
        for i in (col * self.rows)..((1 + col) * self.rows) {
            result.push(self.matrix_flatt[i].clone());
        }

        Vector::new(result)
    }

    fn get_row(&self, row: usize) -> Vector {
        if self.rows < row + 1 {
            panic!("index out of bounds max row {}", self.rows - 1)
        }

        let mut result: Vec<f32> = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            result.push(self.matrix_flatt[i * self.rows + row].clone());
        }

        Vector::new(result)
    }

    /// return column from matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// use eli_math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.col(0), Vector::new(vec![3., 2., 4.]));
    /// ```
    pub fn col(&self, col: usize) -> Vector {
        if self.is_transpose {
            self.get_row(col)
        } else {
            self.get_col(col)
        }
    }

    /// return row from matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// use eli_math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.row(0), Vector::new(vec![3., 4.]));
    /// ```
    pub fn row(&self, row: usize) -> Vector {
        if self.is_transpose {
            self.get_col(row)
        } else {
            self.get_row(row)
        }
    }

    /// return index(row, col) from matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.index(0, 1), 2.);
    /// ```
    pub fn index(&self, mut row: usize, mut col: usize) -> f32 {
        if self.is_transpose {
            let temp = row;
            row = col;
            col = temp;
        }

        if self.rows < row + 1 {
            panic!("index out of bounds max row {}", self.rows - 1)
        }
        if self.cols < col + 1 {
            panic!("index out of bounds max col {}", self.cols - 1)
        }

        self.matrix_flatt[row * self.rows + col]
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
    /// matrix.mul_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. * 2., 3. * 2., 5. * 2.],
    ///         vec![7. * 2., 1. * 2., 4. * 2.]
    ///     ])
    /// );
    /// ```
    pub fn mul_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt = self.matrix_flatt.iter().map(|x| x * scalar).collect();
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
    /// matrix.add_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. + 2., 3. + 2., 5. + 2.],
    ///         vec![7. + 2., 1. + 2., 4. + 2.]
    ///     ])
    /// );
    /// ```
    pub fn add_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt = self.matrix_flatt.iter().map(|x| x + scalar).collect();
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
    /// matrix.div_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. / 2., 3. / 2., 5. / 2.],
    ///         vec![7. / 2., 1. / 2., 4. / 2.]
    ///     ])
    /// );
    /// ```
    pub fn div_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt = self.matrix_flatt.iter().map(|x| x / scalar).collect();
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
    /// matrix.sub_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. - 2., 3. - 2., 5. - 2.],
    ///         vec![7. - 2., 1. - 2., 4. - 2.]
    ///     ])
    /// );
    /// ```
    pub fn sub_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt = self.matrix_flatt.iter().map(|x| x - scalar).collect();
    }

    /// dot product vector with matrix
    pub fn dot_vec(&self, vector: &Vector) -> Vector {
        let vec = vector.vec();
        if vec.len() != self.rows {
            panic!(
                "wrong vector shape expected {}, got {}",
                self.rows,
                vec.len()
            )
        }

        let mut result: Vec<f32> = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            result.push(
                self.col(i)
                    .vec()
                    .iter()
                    .enumerate()
                    .map(|(j, x)| vec[j] * x)
                    .sum(),
            );
        }
        Vector::new(result)
    }

    pub fn add_vec(&mut self, vector: &Vector) {
        todo!();
    }

    pub fn sub_vec(&mut self, vector: &Vector) {
        todo!();
    }

    pub fn mul_vec(&mut self, vector: &Vector) {
        todo!();
    }

    pub fn div_vec(&mut self, vector: &Vector) {
        todo!();
    }

    pub fn add_mat(&mut self, other: &Matrix) {
        todo!();
    }

    pub fn sub_mat(&mut self, other: &Matrix) {
        todo!();
    }

    pub fn div_mat(&mut self, other: &Matrix) {
        todo!();
    }

    pub fn mul_mat(&mut self, other: &Matrix) {
        todo!();
    }

    pub fn dot_mat(&self) {
        todo!();
    }

    /// returns the [`determinant`] of this matrix
    ///
    /// [`determinant`]: https://en.wikipedia.org/wiki/Determinant
    ///
    /// ## Example
    ///
    /// #```rust
    /// use eli_math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.], vec![1., 4., 5.]]);
    /// assert_eq!(matrix.det(), 49.);
    /// #```
    /// note the matrix has to be a [`square matrix`]
    ///
    /// [`square matrix`]: https://en.wikipedia.org/wiki/Square_matrix
    pub fn det(&self) -> f32 {
        if self.cols() != self.rows() {
            panic!("the matrix has to be a square matrix");
        }
        todo!();
    }

    /// this returns the [`eigenvalues`] of this matrix
    ///
    /// [`eigenvalues`]: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
    ///
    /// ## Example
    ///
    /// ```rust
    ///
    /// ```
    /// note the matrix has to be a [`square matrix`]
    ///
    /// [`square matrix`]: https://en.wikipedia.org/wiki/Square_matrix
    pub fn eigen_val(&self) -> f32 {
        if self.cols() != self.rows() {
            panic!("the matrix has to be a square matrix");
        }
        todo!();
    }

    pub fn eigen_vec(&self) -> Vector {
        if self.cols() != self.rows() {
            panic!("the matrix has to be a square matrix");
        }
        todo!();
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_vec() {
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
    fn bytes_vec() {
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
    fn mul_scalar_vec() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.mul_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. * 2., 3. * 2., 5. * 2.]));
    }

    #[test]
    fn div_scalar_vec() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.div_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. / 2., 3. / 2., 5. / 2.]));
    }

    #[test]
    fn add_scalar_vec() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.add_scalar(&2.);
        assert_eq!(vector, Vector::new(vec![2. + 2., 3. + 2., 5. + 2.]));
    }

    #[test]
    fn sub_scalar_vec() {
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

// ----------------------------------------------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------------------------------------------- //

    #[test]
    #[ignore]
    fn det() {
        let matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.], vec![1., 4., 5.]]);
        assert_eq!(matrix.det(), 49.);
    }

    #[test]
    #[should_panic(expected = "the matrix has to be a square matrix")]
    fn det_panic() {
        let matrix = Matrix::new(vec![
            vec![2., -3., 1.],
            vec![2., 0., -1.],
            vec![1., 4., 5.],
            vec![2., 0., -1.],
        ]);
        assert_eq!(matrix.det(), 49.);
    }

    #[test]
    fn bytes() {
        let matrix = Matrix::new(vec![vec![2., 3.], vec![7., 4.]]);
        assert_eq!(
            matrix.bytes(),
            vec![0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 224, 64, 0, 0, 128, 64]
        );
    }

    #[test]
    #[ignore]
    fn matrix_flatt() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        assert_eq!(matrix.matrix_flatt(), vec![2., 3., 5., 7., 1., 4.]);
        matrix.transpose();
        assert_eq!(matrix.matrix_flatt(), vec![2., 7., 3., 1., 5., 4.]);
    }

    #[test]
    #[should_panic(expected = "wrong row shape expected 3, got 4")]
    fn new() {
        let _ = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4., 1.]]);
    }

    #[test]
    fn mul_scalar() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        matrix.mul_scalar(&2.);
        assert_eq!(
            matrix,
            Matrix::new(vec![
                vec![2. * 2., 3. * 2., 5. * 2.],
                vec![7. * 2., 1. * 2., 4. * 2.]
            ])
        );
    }

    #[test]
    fn div_scalar() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        matrix.div_scalar(&2.);
        assert_eq!(
            matrix,
            Matrix::new(vec![
                vec![2. / 2., 3. / 2., 5. / 2.],
                vec![7. / 2., 1. / 2., 4. / 2.]
            ])
        );
    }

    #[test]
    fn add_scalar() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        matrix.add_scalar(&2.);
        assert_eq!(
            matrix,
            Matrix::new(vec![
                vec![2. + 2., 3. + 2., 5. + 2.],
                vec![7. + 2., 1. + 2., 4. + 2.]
            ])
        );
    }

    #[test]
    fn sub_scalar() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        matrix.sub_scalar(&2.);
        assert_eq!(
            matrix,
            Matrix::new(vec![
                vec![2. - 2., 3. - 2., 5. - 2.],
                vec![7. - 2., 1. - 2., 4. - 2.]
            ])
        );
    }

    #[test]
    fn transpose() {
        let mut matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.is_transpose(), false);
        matrix.transpose();
        assert_eq!(matrix.is_transpose(), true);
        matrix.transpose();
        assert_eq!(matrix.is_transpose(), false);
    }

    #[test]
    fn col() {
        let mut matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.col(0), Vector::new(vec![3., 2., 4.]));
        assert_eq!(matrix.col(1), Vector::new(vec![4., 5., 6.]));

        matrix.transpose();

        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.col(0), Vector::new(vec![3., 4.]));
        assert_eq!(matrix.col(1), Vector::new(vec![2., 5.]));
        assert_eq!(matrix.col(2), Vector::new(vec![4., 6.]));
    }

    #[test]
    fn row() {
        let mut matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.row(0), Vector::new(vec![3., 4.]));
        assert_eq!(matrix.row(1), Vector::new(vec![2., 5.]));
        assert_eq!(matrix.row(2), Vector::new(vec![4., 6.]));

        matrix.transpose();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.row(0), Vector::new(vec![3., 2., 4.]));
        assert_eq!(matrix.row(1), Vector::new(vec![4., 5., 6.]));
    }

    #[test]
    #[should_panic(expected = "index out of bounds max row 2")]
    fn row_panic() {
        let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        let _ = matrix.row(3);
    }

    #[test]
    #[should_panic(expected = "index out of bounds max col 1")]
    fn col_panic() {
        let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        let _ = matrix.col(2);
    }

    #[test]
    #[ignore]
    fn index_mat() {
        let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.index(0, 0), 3.);
        assert_eq!(matrix.index(0, 1), 2.);
        assert_eq!(matrix.index(0, 2), 4.);
        assert_eq!(matrix.index(1, 0), 4.);
        assert_eq!(matrix.index(1, 1), 5.);
        assert_eq!(matrix.index(1, 2), 6.);
    }

    #[test]
    fn dot_mat() {
        let matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
        assert_eq!(
            matrix.dot_vec(&Vector::new(vec![2., 1., 0.])),
            Vector::new(vec![1., -3.])
        )
    }

    #[test]
    #[should_panic(expected = "wrong vector shape expected 3, got 2")]
    fn dot_vec_mat_panic() {
        let matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
        assert_eq!(
            matrix.dot_vec(&Vector::new(vec![2., 1.])),
            Vector::new(vec![1., -3.])
        )
    }
}
