#[cfg(test)]
mod tests_vec {
    use eli_math::linear_algebra::Vector;

    #[test]
    fn new_rand() {
        let vector = Vector::new_rand(4);
        assert_eq!(
            vector.vec(),
            vec![0.69186187, 0.3494884, 0.23957491, 0.06540034]
        );
    }

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

    // ----------------------------------------------------------------------------------------------------------------------------------------------------- //
    // ----------------------------------------------------------------------------------------------------------------------------------------------------- //
    // ----------------------------------------------------------------------------------------------------------------------------------------------------- //
}
#[cfg(test)]
mod tests_mat {
    use eli_math::linear_algebra::Matrix;
    use eli_math::linear_algebra::Vector;

    #[test]
    fn new_rand() {
        let matrix = Matrix::new_rand(3, 4);
        assert_eq!(
            matrix.matrix_flatt(),
            vec![
                0.69186187,
                0.3494884,
                0.23957491,
                0.06540034,
                0.5443042,
                0.013656098,
                0.4336478,
                0.8349666,
                0.10932327,
                0.52898574,
                0.4612443,
                0.3579495,
            ]
        );

        let matrix = Matrix::new_rand(2, 3);
        assert_eq!(
            matrix.matrix_flatt(),
            vec![
                0.69186187,
                0.3494884,
                0.23957491,
                0.06540034,
                0.5443042,
                0.013656098,
            ]
        );
    }
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
