use super::*;
use std::mem;

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
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// ```
    /// crates matrix that looks like this:
    /// ```
    /// 3. 2. 4.
    /// 4. 5. 6.
    /// ```
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

    fn get_col(&self, col: usize) -> vector::Vector {
        if self.cols < col + 1 {
            panic!("index out of bounds max col {}", self.cols - 1)
        }

        let mut result: Vec<f32> = Vec::with_capacity(self.rows);
        for i in (col * self.rows)..((1 + col) * self.rows) {
            result.push(self.matrix_flatt[i].clone());
        }

        vector::Vector::new(result)
    }

    fn get_row(&self, row: usize) -> vector::Vector {
        if self.rows < row + 1 {
            panic!("index out of bounds max row {}", self.rows - 1)
        }

        let mut result: Vec<f32> = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            result.push(self.matrix_flatt[i * self.rows + row].clone());
        }

        vector::Vector::new(result)
    }

    /// return column from matrix
    ///
    /// ## Example
    ///
    /// ```rust
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.col(0), vec![3., 2., 4.]);
    /// ```
    pub fn col(&self, col: usize) -> vector::Vector {
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
    /// let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
    /// assert_eq!(matrix.row(0), vec![3., 4.]);
    /// ```
    pub fn row(&self, row: usize) -> vector::Vector {
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
    pub fn dot_vec(&self, vector: &vector::Vector) -> vector::Vector {
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
        vector::Vector::new(result)
    }

    pub fn add_vec(&mut self, vector: &vector::Vector) {
        todo!();
    }

    pub fn sub_vec(&mut self, vector: &vector::Vector) {
        todo!();
    }

    pub fn mul_vec(&mut self, vector: &vector::Vector) {
        todo!();
    }

    pub fn div_vec(&mut self, vector: &vector::Vector) {
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
    /// ```rust
    /// let matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.], vec![1., 4., 5.]]);
    /// assert_eq!(matrix.det(), 49);
    /// ```
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

    pub fn eigen_vec(&self) -> vector::Vector {
        if self.cols() != self.rows() {
            panic!("the matrix has to be a square matrix");
        }
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(matrix.col(0), vector::Vector::new(vec![3., 2., 4.]));
        assert_eq!(matrix.col(1), vector::Vector::new(vec![4., 5., 6.]));

        matrix.transpose();

        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.col(0), vector::Vector::new(vec![3., 4.]));
        assert_eq!(matrix.col(1), vector::Vector::new(vec![2., 5.]));
        assert_eq!(matrix.col(2), vector::Vector::new(vec![4., 6.]));
    }

    #[test]
    fn row() {
        let mut matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.row(0), vector::Vector::new(vec![3., 4.]));
        assert_eq!(matrix.row(1), vector::Vector::new(vec![2., 5.]));
        assert_eq!(matrix.row(2), vector::Vector::new(vec![4., 6.]));

        matrix.transpose();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.row(0), vector::Vector::new(vec![3., 2., 4.]));
        assert_eq!(matrix.row(1), vector::Vector::new(vec![4., 5., 6.]));
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
    fn index() {
        let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.index(0, 0), 3.);
        assert_eq!(matrix.index(0, 1), 2.);
        assert_eq!(matrix.index(0, 2), 4.);
        assert_eq!(matrix.index(1, 0), 4.);
        assert_eq!(matrix.index(1, 1), 5.);
        assert_eq!(matrix.index(1, 2), 6.);
    }

    #[test]
    fn dot_vec() {
        let matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
        assert_eq!(
            matrix.dot_vec(&vector::Vector::new(vec![2., 1., 0.])),
            vector::Vector::new(vec![1., -3.])
        )
    }

    #[test]
    #[should_panic(expected = "wrong vector shape expected 3, got 2")]
    fn dot_vec_panic() {
        let matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
        assert_eq!(
            matrix.dot_vec(&vector::Vector::new(vec![2., 1.])),
            vector::Vector::new(vec![1., -3.])
        )
    }
}
