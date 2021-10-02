use crate::linear_algebra::Vector;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Debug)]
pub struct Matrix {
    cols: usize,
    rows: usize,
    matrix_flatt: Vector,
    is_transpose: bool,
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.cols() == other.cols()
            && self.rows() == other.rows()
            && self.matrix_flatt() == other.matrix_flatt()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.cols() {
            writeln!(f, "{}", self.col(i).unwrap())?;
        }
        Ok(())
    }
}

impl Add for Matrix {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        result.add_mat(&other);
        result
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, other: Self) {
        self.add_mat(&other);
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.clone();
        result.sub_mat(&other);
        result
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, other: Self) {
        self.sub_mat(&other);
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = self.clone();
        result.mul_mat(&other);
        result
    }
}

impl MulAssign for Matrix {
    fn mul_assign(&mut self, other: Self) {
        self.mul_mat(&other);
    }
}

impl Div for Matrix {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut result = self.clone();
        result.div_mat(&other);
        result
    }
}

impl DivAssign for Matrix {
    fn div_assign(&mut self, other: Self) {
        self.div_mat(&other);
    }
}

impl Matrix {
    /// converts 2d vec in to matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]).unwrap();
    /// ```
    /// crates matrix that looks like this:
    ///
    /// [3.0, 2.0, 4.0]
    /// [4.0, 5.0, 6.0]
    ///
    pub fn new(vec: Vec<Vec<f32>>) -> Result<Self, String> {
        let cols = vec.len();
        let rows = vec[0].len();

        let mut flatt: Vec<f32> = Vec::with_capacity(cols * rows);

        for col in vec {
            if col.len() != rows {
                return Err(format!(
                    "wrong row shape expected {}, got {}",
                    rows,
                    col.len()
                ));
            }
            col.iter().for_each(|&x| flatt.push(x))
        }

        Ok(Self {
            cols: cols,
            rows: rows,
            matrix_flatt: Vector::new(flatt),
            is_transpose: false,
        })
    }

    /// returns the Matrix of the [outer product] with the vectors
    ///
    /// [outer product]:https://en.wikipedia.org/wiki/Outer_product
    ///
    ///  ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let vector1 = Vector::new(vec![2., 4., 3.]);
    /// let vector2 = Vector::new(vec![2., 7., 9.]);
    /// let matrix = Matrix::new_outer(&vector1,&vector2);
    /// assert_eq!(matrix, Matrix::new_flatt(vec![4.0, 14.0, 18.0, 8.0, 28.0, 36.0, 6.0, 21.0, 27.0], 3, 3));
    /// ```
    pub fn new_outer(vector1: &Vector, vector2: &Vector) -> Result<Self, String> {
        let mut vec = Vec::new();
        for i in 0..vector1.len() {
            let mut temp = Vec::new();
            for j in 0..vector2.len() {
                let index_1 = vector1.index(i)?;
                let index_2 = vector2.index(j)?;
                temp.push(index_1 * index_2);
            }
            vec.push(temp);
        }

        Self::new(vec)
    }

    /// generats a matrix from a 1D Vector
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new_flatt(vec![3., 2., 4., 4., 5., 6.], 2, 3).unwrap();
    /// assert_eq!(matrix.matrix_flatt(), Ok(Vector::new(vec![3., 2., 4., 4., 5., 6.])));
    /// ```
    pub fn new_flatt(matrix_flatt: Vec<f32>, cols: usize, rows: usize) -> Result<Self, String> {
        if cols * rows != matrix_flatt.len() {
            Err(format!(
                "cols * rows = {} has to be the same len as the matrix_flatt = {}",
                cols * rows,
                matrix_flatt.len()
            ))
        } else {
            Ok(Self {
                cols,
                rows,
                matrix_flatt: Vector::new(matrix_flatt),
                is_transpose: false,
            })
        }
    }

    /// generates a matrix of size `cols` and `rows` with random values between 0 and 1
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new_rand(2, 3);
    /// assert_eq!(
    ///     matrix.matrix_flatt(),
    ///     Ok(Vector::new(vec![
    ///        0.69186187,
    ///        0.3494884,
    ///        0.23957491,
    ///        0.06540034,
    ///        0.5443042,
    ///        0.013656098,
    ///    ]))
    /// );
    /// ```
    pub fn new_rand(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            matrix_flatt: Vector::new_rand(cols * rows),
            is_transpose: false,
        }
    }

    /// generates a matrix of size `cols` and `rows` with all values being 0.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new_zero(2, 3);
    /// assert_eq!(matrix.matrix_flatt().unwrap(), Vector::new(vec![0., 0., 0., 0., 0., 0.]));
    /// ```
    pub fn new_zero(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            matrix_flatt: Vector::new_zero(cols * rows),
            is_transpose: false,
        }
    }

    /// getter for the internal matrix_flatt representation
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    /// assert_eq!(matrix.matrix_flatt().unwrap(), Vector::new(vec![2., 3., 5., 7., 1., 4.]));
    /// ```
    pub fn matrix_flatt(&self) -> Result<Vector, String> {
        if self.is_transpose {
            let mut matrix_flatt = Vec::with_capacity(self.cols * self.rows);
            for i in 0..self.rows {
                for val in self.col(i)?.vec() {
                    matrix_flatt.push(val);
                }
            }
            Ok(Vector::new(matrix_flatt))
        } else {
            Ok(self.matrix_flatt.clone())
        }
    }

    /// return index(row, col) from matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]).unwrap();
    /// assert_eq!(matrix.index(0, 1), Ok(2.));
    /// ```
    pub fn index(&self, mut row: usize, mut col: usize) -> Result<f32, String> {
        if self.is_transpose {
            let temp = row;
            row = col;
            col = temp;
        }

        if self.rows < row {
            Err(format!("index out of bounds max row {}", self.rows - 1))
        } else if self.cols < col {
            Err(format!("index out of bounds max col {}", self.cols - 1))
        } else {
            let index = row * self.rows + col;
            self.matrix_flatt.index(index)
        }
    }

    /// sets the value of the matrix at the specifide index row col
    ///
    /// ## Example
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    /// matrix.set_index(0, 1, 10.);
    /// assert_eq!(matrix.matrix_flatt(), Ok(Vector::new(vec![2.0, 10.0, 5.0, 7.0, 1.0, 4.0])));
    /// ```
    pub fn set_index(&mut self, mut row: usize, mut col: usize, val: f32) -> Option<String> {
        if self.is_transpose {
            let temp = row;
            row = col;
            col = temp;
        }

        if self.rows < row + 1 {
            Some(format!("index out of bounds max row {}", self.rows - 1))
        } else if self.cols < col + 1 {
            Some(format!("index out of bounds max col {}", self.cols - 1))
        } else {
            let index = row * self.rows + col;
            self.matrix_flatt.set_index(index, val)
        }
    }

    /// return the length of the columns
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]).unwrap();
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
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]).unwrap();
    /// assert_eq!(matrix.rows(), 3);
    /// ```
    pub fn rows(&self) -> usize {
        if self.is_transpose {
            self.cols
        } else {
            self.rows
        }
    }

    /// return column from matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]).unwrap();
    /// assert_eq!(matrix.col(0), Ok(Vector::new(vec![3., 2., 4.])));
    /// ```
    pub fn col(&self, col: usize) -> Result<Vector, String> {
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
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]).unwrap();
    /// assert_eq!(matrix.row(0), Ok(Vector::new(vec![3., 4.])));
    /// ```
    pub fn row(&self, row: usize) -> Result<Vector, String> {
        if self.is_transpose {
            self.get_col(row)
        } else {
            self.get_row(row)
        }
    }

    /// returns true if the matrix is a [square matrix]  
    ///
    /// that means if it has as much rows as cols
    ///
    /// [square matrix]:https://en.wikipedia.org/wiki/Square_matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2.], vec![4., 5.]]).unwrap();
    /// assert_eq!(matrix.is_square(), true);
    /// ```
    pub fn is_square(&self) -> bool {
        self.cols() == self.rows()
    }

    /// a getter for the [transpose]
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 2.], vec![4., 5.]]).unwrap();
    /// assert_eq!(matrix.is_transpose(), false);
    /// ```
    ///
    /// [transpose]: https://en.wikipedia.org/wiki/Transpose
    pub fn is_transpose(&self) -> bool {
        self.is_transpose
    }

    /// [transposes] matrix flips rows and cols
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![3., 2.], vec![4., 5.]]).unwrap();
    /// matrix.transpose();
    /// assert_eq!(matrix.is_transpose(), true);
    /// ```
    ///
    /// [transposes]: https://en.wikipedia.org/wiki/Transpose
    pub fn transpose(&mut self) {
        self.is_transpose = !self.is_transpose;
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    /// matrix.mul_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. * 2., 3. * 2., 5. * 2.],
    ///         vec![7. * 2., 1. * 2., 4. * 2.]
    ///     ]).unwrap()
    /// );
    /// ```
    pub fn mul_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt.mul_scalar(scalar);
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    /// matrix.add_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. + 2., 3. + 2., 5. + 2.],
    ///         vec![7. + 2., 1. + 2., 4. + 2.]
    ///     ]).unwrap()
    /// );
    /// ```
    pub fn add_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt.add_scalar(scalar);
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    /// matrix.div_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. / 2., 3. / 2., 5. / 2.],
    ///         vec![7. / 2., 1. / 2., 4. / 2.]
    ///     ]).unwrap()
    /// );
    /// ```
    pub fn div_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt.div_scalar(scalar);
    }

    /// multiplies each component from the matrix with a scalar value and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    /// matrix.sub_scalar(&2.);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![
    ///         vec![2. - 2., 3. - 2., 5. - 2.],
    ///         vec![7. - 2., 1. - 2., 4. - 2.]
    ///     ]).unwrap()
    /// );
    /// ```
    pub fn sub_scalar(&mut self, scalar: &f32) {
        self.matrix_flatt.sub_scalar(scalar);
    }

    /// computes the dot product between the vector and this matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]).unwrap();
    /// assert_eq!(
    ///     matrix.dot_vec(&Vector::new(vec![2., 1., 0.])),
    ///     Ok(Vector::new(vec![1., -3.]))
    /// );
    /// ```
    pub fn dot_vec(&self, vector: &Vector) -> Result<Vector, String> {
        let vec = vector.vec();
        if let Some(err) = check_vector(self, vector) {
            return Err(err);
        }

        let mut result: Vec<f32> = Vec::with_capacity(self.cols());
        for i in 0..self.cols() {
            let col = self.col(i)?;
            let sum = col.vec().iter().enumerate().map(|(j, x)| vec[j] * x).sum();
            result.push(sum);
        }
        Ok(Vector::new(result))
    }

    /// adds each component from the vector with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let vector = Vector::new(vec![2., 4., 6.]);
    /// matrix.add_vec(&vector);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![6.0, 0.0, -1.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn add_vec(&mut self, vector: &Vector) -> Option<String> {
        if let Some(err) = check_vector(self, vector) {
            return Some(err);
        }
        for row in 0..self.rows() - 1 {
            for col in 0..self.cols() - 1 {
                let val;
                match (vector.index(row), self.index(row, col)) {
                    (Ok(vector), Ok(mat)) => val = mat + vector,
                    (_, Err(err)) => return Some(err),
                    (Err(err), _) => return Some(err),
                };

                if let Some(err) = self.set_index(row, col, val) {
                    return Some(err);
                }
            }
        }
        None
    }

    /// subtracts each component from the vector with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let vector = Vector::new(vec![2., 4., 6.]);
    /// matrix.sub_vec(&vector);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![vec![0.0, -3.0, 1.0], vec![-2.0, 0.0, -1.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn sub_vec(&mut self, vector: &Vector) -> Option<String> {
        if let Some(err) = check_vector(self, vector) {
            return Some(err);
        }
        for row in 0..self.rows() - 1 {
            for col in 0..self.cols() - 1 {
                let val;
                match (vector.index(row), self.index(row, col)) {
                    (Ok(vector), Ok(mat)) => val = mat - vector,
                    (_, Err(err)) => return Some(err),
                    (Err(err), _) => return Some(err),
                };

                if let Some(err) = self.set_index(row, col, val) {
                    return Some(err);
                }
            }
        }
        None
    }

    /// multiplys each component from the vector with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let vector = Vector::new(vec![2., 4., 6.]);
    /// matrix.mul_vec(&vector);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![8.0, 0.0, -1.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn mul_vec(&mut self, vector: &Vector) -> Option<String> {
        if let Some(err) = check_vector(self, vector) {
            return Some(err);
        }
        for row in 0..self.rows() - 1 {
            for col in 0..self.cols() - 1 {
                let val;
                match (vector.index(row), self.index(row, col)) {
                    (Ok(vector), Ok(mat)) => val = mat * vector,
                    (_, Err(err)) => return Some(err),
                    (Err(err), _) => return Some(err),
                };

                if let Some(err) = self.set_index(row, col, val) {
                    return Some(err);
                }
            }
        }
        None
    }

    /// divides each component from the vector with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let vector = Vector::new(vec![2., 4., 6.]);
    /// matrix.div_vec(&vector);
    /// assert_eq!(
    ///     matrix,
    ///     Matrix::new(vec![vec![1.0, -3.0, 1.0], vec![0.5, 0.0, -1.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn div_vec(&mut self, vector: &Vector) -> Option<String> {
        if let Some(err) = check_vector(self, vector) {
            return Some(err);
        }
        for row in 0..self.rows() - 1 {
            for col in 0..self.cols() - 1 {
                let val;
                match (vector.index(row), self.index(row, col)) {
                    (Ok(vector), Ok(mat)) => val = mat / vector,
                    (_, Err(err)) => return Some(err),
                    (Err(err), _) => return Some(err),
                };

                if let Some(err) = self.set_index(row, col, val) {
                    return Some(err);
                }
            }
        }
        None
    }

    /// adds each component from the matrix with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    ///
    /// matrix1.add_mat(&matrix2);
    /// assert_eq!(
    ///     matrix1,
    ///     Matrix::new(vec![vec![4.0, 0.0, 6.0], vec![9.0, 1.0, 3.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn add_mat(&mut self, other: &Matrix) -> Option<String> {
        if let Some(err) = check_matrix(self, other) {
            return Some(err);
        }

        let self_flatt = match self.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        let other_flatt = match other.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        self.matrix_flatt = self_flatt + other_flatt;
        self.is_transpose = false;
        self.cols = other.cols();
        self.rows = other.rows();
        None
    }

    /// subtracts each component from the matrix with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    ///
    /// matrix1.sub_mat(&matrix2);
    /// assert_eq!(
    ///   matrix1,
    ///   Matrix::new(vec![vec![0.0, -6.0, -4.0], vec![-5.0, -1.0, -5.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn sub_mat(&mut self, other: &Matrix) -> Option<String> {
        if let Some(err) = check_matrix(self, other) {
            return Some(err);
        }
        let self_flatt = match self.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        let other_flatt = match other.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        self.matrix_flatt = self_flatt - other_flatt;
        self.is_transpose = false;
        self.cols = other.cols();
        self.rows = other.rows();
        None
    }

    /// divides each component from the matrix with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    ///
    /// matrix1.div_mat(&matrix2);
    /// assert_eq!(
    ///     matrix1,
    ///     Matrix::new(vec![vec![1.0, -1.0, 0.2], vec![0.2857143, 0.0, -0.25]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn div_mat(&mut self, other: &Matrix) -> Option<String> {
        if let Some(err) = check_matrix(self, other) {
            return Some(err);
        }
        let self_flatt = match self.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        let other_flatt = match other.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        self.matrix_flatt = self_flatt / other_flatt;
        self.is_transpose = false;
        self.cols = other.cols();
        self.rows = other.rows();
        None
    }

    /// multiples each component from the matrix with the component of the other matrix and stors the result in this matrix   
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]).unwrap();
    /// let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]).unwrap();
    ///
    /// matrix1.mul_mat(&matrix2);
    /// assert_eq!(
    ///   matrix1,
    ///   Matrix::new(vec![vec![4.0, -9.0, 5.0], vec![14.0, 0.0, -4.0]]).unwrap()
    /// );
    /// ```
    /// note it panics if the matrices have not the same rows and cols
    pub fn mul_mat(&mut self, other: &Matrix) -> Option<String> {
        if let Some(err) = check_matrix(self, other) {
            return Some(err);
        }
        let self_flatt = match self.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        let other_flatt = match other.matrix_flatt() {
            Ok(flatt) => flatt,
            Err(err) => return Some(err),
        };
        self.matrix_flatt = self_flatt * other_flatt;
        self.is_transpose = false;
        self.cols = other.cols();
        self.rows = other.rows();
        None
    }

    /// returns the [determinant] of this matrix
    ///
    /// [determinant]: https://en.wikipedia.org/wiki/Determinant
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![1., 2.], vec![3., 4.]]).unwrap();
    /// assert_eq!(matrix.det(), Ok(-2.));
    /// ```
    ///  note the matrix has to be a [square matrix]
    ///
    /// [square matrix]: https://en.wikipedia.org/wiki/Square_matrix
    pub fn det(&self) -> Result<f32, String> {
        if let Some(err) = check_square(self) {
            return Err(err);
        }
        if self.rows() == 2 {
            Ok(self.index(0, 0)? * self.index(1, 1)? - self.index(0, 1)? * self.index(1, 0)?)
        } else {
            let mut sign = 1.;
            let mut sum = 0.;

            for col in 0..self.cols() {
                let sub = self.finde_sub(0, col)?;
                sum += sub.det()? * sign * self.index(0, col)?;
                sign *= -1.;
            }

            Ok(sum)
        }
    }

    /// this returns the [eigenvalues] of this matrix
    ///
    /// [eigenvalues]: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
    ///
    /// ## Example
    ///
    /// ```rust
    ///
    /// ```
    /// note the matrix has to be a [square matrix]
    ///
    /// [square matrix]: https://en.wikipedia.org/wiki/Square_matrix
    pub fn eigen_val(&self) -> Result<f32, String> {
        if let Some(err) = check_square(self) {
            return Err(err);
        }
        todo!();
    }

    pub fn eigen_vec(&self) -> Result<Vector, String> {
        if let Some(err) = check_square(self) {
            return Err(err);
        }
        todo!();
    }

    pub fn dot_mat(&self, other: &Matrix) -> Option<String> {
        if let Some(err) = check_matrix(self, other) {
            return Some(err);
        }
        todo!();
    }

    pub fn inv(&mut self) -> Option<String> {
        if let Some(err) = check_square(self) {
            return Some(err);
        }
        let det = match self.det() {
            Ok(det) => det,
            Err(err) => return Some(err),
        };

        if det == 0. {
            return Some(format!("the determinant of the matrix can't be 0"));
        }
        todo!();
    }

    /// applyes the lamda function to each value in the matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new(vec![vec![0.7, 0.2, 0.3], vec![0.5, 0.6, 0.1]]).unwrap();
    /// let step: Box<(dyn Fn(f32) -> f32 + 'static)> = Box::new(|x: f32| -> f32 {
    ///     if x > 0.5 {
    ///         1.
    ///     } else {
    ///         0.
    ///     }
    /// });
    /// matrix.apply_func_val(&step);
    /// assert_eq!(matrix.matrix_flatt().unwrap().vec(), vec![1., 0., 0., 0., 1., 0.]);
    /// ```
    pub fn apply_func_val(&mut self, lamda: &Box<(dyn Fn(f32) -> f32 + 'static)>) {
        self.matrix_flatt.apply_func(lamda);
    }

    /// returns a vector of the sumed rows
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// use math::linear_algebra::Vector;
    /// let matrix = Matrix::new(vec![vec![3., 1.], vec![5., 3.]]).unwrap();
    /// assert_eq!(matrix.sum_vec(), Ok(Vector::new(vec![8., 4.])));
    /// ```
    pub fn sum_vec(&self) -> Result<Vector, String> {
        let mut vec = Vec::new();
        for i in 0..self.rows() {
            let sum = self.row(i)?.sum();
            vec.push(sum);
        }
        Ok(Vector::new(vec))
    }

    /// returns the sum of the elements
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![3., 1.], vec![5., 3.]]).unwrap();
    /// assert_eq!(matrix.sum(), 12.);
    /// ```
    pub fn sum(&self) -> f32 {
        self.matrix_flatt.sum()
    }

    // finds the sub matrix is user for the determinant
    fn finde_sub(&self, row: usize, col: usize) -> Result<Self, String> {
        let mut flatt = Vec::with_capacity((self.cols() - 1) * (self.rows() - 1));

        for i in 0..self.cols() {
            for j in 0..self.rows() {
                if !(i == col || j == row) {
                    flatt.push(self.index(i, j)?);
                }
            }
        }
        Self::new_flatt(flatt, self.cols() - 1, self.rows() - 1)
    }

    fn get_row(&self, row: usize) -> Result<Vector, String> {
        if self.rows < row + 1 {
            Err(format!("index out of bounds max row {}", self.rows - 1))
        } else {
            let mut result: Vec<f32> = Vec::with_capacity(self.cols);
            for i in 0..self.cols {
                let index = self.matrix_flatt.index(i * self.rows + row)?;
                result.push(index);
            }

            Ok(Vector::new(result))
        }
    }

    fn get_col(&self, col: usize) -> Result<Vector, String> {
        if self.cols < col + 1 {
            Err(format!("index out of bounds max col {}", self.cols - 1))
        } else {
            let mut result: Vec<f32> = Vec::with_capacity(self.rows);
            for i in (col * self.rows)..((1 + col) * self.rows) {
                let index = self.matrix_flatt.index(i)?;
                result.push(index);
            }

            Ok(Vector::new(result))
        }
    }
}

fn check_square(mat: &Matrix) -> Option<String> {
    if !mat.is_square() {
        Some(format!("the matrix has to be a square matrix"))
    } else if mat.rows() == 1 {
        Some(format!("the matrix has to have more then one row"))
    } else {
        None
    }
}

fn check_vector(mat: &Matrix, vec: &Vector) -> Option<String> {
    if vec.len() != mat.rows() {
        Some(format!(
            "wrong vector shape expected {}, got {}",
            mat.rows,
            vec.len()
        ))
    } else {
        None
    }
}

fn check_matrix(mat1: &Matrix, mat2: &Matrix) -> Option<String> {
    if mat1.rows() != mat2.rows() {
        Some(format!(
            "wrong row shape expected {}, got {}",
            mat1.rows, mat2.rows
        ))
    } else if mat1.cols() != mat2.cols() {
        Some(format!(
            "wrong col shape expected {}, got {}",
            mat1.cols, mat2.cols
        ))
    } else {
        None
    }
}

#[cfg(feature = "gpu")]
use crate::random;
#[cfg(feature = "gpu")]
use std::mem;

#[cfg(feature = "gpu")]
impl Matrix {
    /// this return a vector of bytes representing the matrix
    ///
    /// this is useful for the *GPU* because the interface only uses bytes
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::linear_algebra::Matrix;
    /// let matrix = Matrix::new(vec![vec![2., 3.], vec![7., 4.]]);
    /// assert_eq!(
    ///     matrix.bytes(),
    ///     vec![0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 224, 64, 0, 0, 128, 64]
    /// );
    /// ```
    /// note the fist and seconde `f32` is the rows and cols of the matrix
    pub fn bytes(&self) -> Vec<u8> {
        let size = (2 + self.rows() * self.cols()) * mem::size_of::<f32>();
        let mut bytes = Vec::<u8>::with_capacity(size);

        for b in (self.rows() as f32).to_ne_bytes().to_vec() {
            bytes.push(b);
        }
        for b in (self.cols() as f32).to_ne_bytes().to_vec() {
            bytes.push(b);
        }

        // `skip(4)` because the first 4 bytes is the len of the vector (f32 = 4bytes)
        for &b in self.matrix_flatt().bytes().iter().skip(4) {
            bytes.push(b);
        }
        bytes
    }
}
