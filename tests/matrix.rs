#[cfg(test)]
mod tests {
    use math::linear_algebra::Matrix;
    use math::linear_algebra::Vector;

    #[test]
    fn det() {
        // examples from https://www.mathsisfun.com/algebra/matrix-determinant.html

        let matrix = Matrix::new(vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(matrix.det(), -2.);

        let matrix = Matrix::new(vec![vec![3., 8.], vec![4., 6.]]);
        assert_eq!(matrix.det(), -14.);

        let matrix = Matrix::new(vec![vec![4., 6.], vec![3., 8.]]);
        assert_eq!(matrix.det(), 14.);

        let matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.], vec![1., 4., 5.]]);
        assert_eq!(matrix.det(), -46.);

        let matrix = Matrix::new(vec![vec![6., 1., 1.], vec![4., -2., 5.], vec![2., 8., 7.]]);
        assert_eq!(matrix.det(), -316.);

        let matrix = Matrix::new(vec![
            vec![6., 1., 1., 4.],
            vec![4., -2., 5., -7.],
            vec![2., 8., 7., 3.],
            vec![4., 1., 4., 2.],
        ]);
        assert_eq!(matrix.det(), -3748.);
    }

    #[test]
    fn name() {
        let matrix = Matrix::new_flatt(vec![3., 2., 4., 4., 5., 6.], 2, 3);
        assert_eq!(matrix.matrix_flatt().vec(), vec![3., 2., 4., 4., 5., 6.]);
    }

    #[test]
    fn add_vec() {
        let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let vector = Vector::new(vec![2., 4., 6.]);
        matrix.add_vec(&vector);
        assert_eq!(
            matrix,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![6.0, 0.0, -1.0]])
        );

        matrix.transpose();
        let vector = Vector::new(vec![-2., 6.]);
        matrix.add_vec(&vector);
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![2.0, 4.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn sub_vec() {
        let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let vector = Vector::new(vec![2., 4., 6.]);
        matrix.sub_vec(&vector);
        assert_eq!(
            matrix,
            Matrix::new(vec![vec![0.0, -3.0, 1.0], vec![-2.0, 0.0, -1.0]])
        );

        matrix.transpose();
        let vector = Vector::new(vec![-2., 6.]);
        matrix.sub_vec(&vector);
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![2.0, 0.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn mul_vec() {
        let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let vector = Vector::new(vec![2., 4., 6.]);
        matrix.mul_vec(&vector);
        assert_eq!(
            matrix,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![8.0, 0.0, -1.0]])
        );

        matrix.transpose();
        let vector = Vector::new(vec![-2., 6.]);
        matrix.mul_vec(&vector);
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![-8.0, -16.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn div_vec() {
        let mut matrix = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let vector = Vector::new(vec![2., 4., 6.]);
        matrix.div_vec(&vector);
        assert_eq!(
            matrix,
            Matrix::new(vec![vec![1.0, -3.0, 1.0], vec![0.5, 0.0, -1.0]])
        );

        matrix.transpose();
        let vector = Vector::new(vec![-2., 6.]);
        matrix.div_vec(&vector);
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![-0.5, -0.25, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn add_mat() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1.add_mat(&matrix2);
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![9.0, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1.add_mat(&matrix2);
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![6.0, 5.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn sub_mat() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1.sub_mat(&matrix2);
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![0.0, -3.0, 1.0], vec![-5.0, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1.sub_mat(&matrix2);
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![-2.0, -1.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn mul_mat() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1.mul_mat(&matrix2);
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![14.0, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1.mul_mat(&matrix2);
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![8.0, -56.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn div_mat() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1.div_mat(&matrix2);
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![1.0, -3.0, 1.0], vec![0.2857143, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1.div_mat(&matrix2);
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![0.5, -0.071428575, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn add_assign() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1 += matrix2;
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![9.0, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1 += matrix2;
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![6.0, 5.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn sub_assign() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1 -= matrix2;
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![0.0, -3.0, 1.0], vec![-5.0, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1 -= matrix2;
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![-2.0, -1.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn mul_assign() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1 *= matrix2;
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![14.0, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1 *= matrix2;
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![8.0, -56.0, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn div_assign() {
        let mut matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        matrix1 /= matrix2;
        assert_eq!(
            matrix1,
            Matrix::new(vec![vec![1.0, -3.0, 1.0], vec![0.2857143, 0.0, -1.0]])
        );

        matrix1.transpose();
        let matrix2 = Matrix::new(vec![vec![2., -4.], vec![7., 1.], vec![-3., 5.]]);
        matrix1 /= matrix2;
        assert_eq!(
            matrix1.matrix_flatt().vec(),
            vec![0.5, -0.071428575, -3.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn add() {
        let matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        assert_eq!(
            matrix1 + matrix2,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![9.0, 0.0, -1.0]])
        );
    }

    #[test]
    fn sub() {
        let matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        assert_eq!(
            matrix1 - matrix2,
            Matrix::new(vec![vec![0.0, -3.0, 1.0], vec![-5.0, 0.0, -1.0]])
        );
    }

    #[test]
    fn mul() {
        let matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        assert_eq!(
            matrix1 * matrix2,
            Matrix::new(vec![vec![4.0, -3.0, 1.0], vec![14.0, 0.0, -1.0]])
        );
    }

    #[test]
    fn div() {
        let matrix1 = Matrix::new(vec![vec![2., -3., 1.], vec![2., 0., -1.]]);
        let matrix2 = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);

        assert_eq!(
            matrix1 / matrix2,
            Matrix::new(vec![vec![1.0, -3.0, 1.0], vec![0.2857143, 0.0, -1.0]])
        );
    }

    #[test]
    fn set_index() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        assert_eq!(matrix.matrix_flatt().vec(), vec![2., 3., 5., 7., 1., 4.]);
        matrix.set_index(0, 1, 10.);
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![2.0, 10.0, 5.0, 7.0, 1.0, 4.0]
        );
        matrix.transpose();
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![2.0, 7.0, 10.0, 1.0, 5.0, 4.0]
        );
        matrix.set_index(0, 1, 10.);
        assert_eq!(
            matrix.matrix_flatt().vec(),
            vec![2.0, 10.0, 10.0, 1.0, 5.0, 4.0]
        );
    }

    #[test]
    fn matrix_flatt() {
        let mut matrix = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4.]]);
        assert_eq!(matrix.matrix_flatt().vec(), vec![2., 3., 5., 7., 1., 4.]);
        matrix.transpose();
        assert_eq!(matrix.matrix_flatt().vec(), vec![2., 7., 3., 1., 5., 4.]);
    }

    #[test]
    fn index() {
        let mut matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.index(0, 0), 3.);
        assert_eq!(matrix.index(0, 1), 2.);
        assert_eq!(matrix.index(0, 2), 4.);
        assert_eq!(matrix.index(1, 0), 4.);
        assert_eq!(matrix.index(1, 1), 5.);
        assert_eq!(matrix.index(1, 2), 6.);

        matrix.transpose();
        assert_eq!(matrix.index(0, 0), 3.);
        assert_eq!(matrix.index(0, 1), 4.);
        assert_eq!(matrix.index(1, 0), 2.);
        assert_eq!(matrix.index(1, 1), 5.);
        assert_eq!(matrix.index(2, 0), 4.);
        assert_eq!(matrix.index(2, 1), 6.);
    }

    #[test]
    fn new_zero() {
        let matrix = Matrix::new_zero(2, 3);
        assert_eq!(matrix.matrix_flatt().vec(), vec![0., 0., 0., 0., 0., 0.]);
    }

    #[test]
    fn new_rand() {
        let matrix = Matrix::new_rand(3, 4);
        assert_eq!(
            matrix.matrix_flatt().vec(),
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
            matrix.matrix_flatt().vec(),
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

    #[cfg(feature = "gpu")]
    #[test]
    fn bytes() {
        let matrix = Matrix::new(vec![vec![2., 3.], vec![7., 4.]]);
        assert_eq!(
            matrix.bytes(),
            vec![0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 224, 64, 0, 0, 128, 64]
        );
    }

    #[test]
    #[should_panic(expected = "wrong row shape expected 3, got 4")]
    fn new() {
        let _ = Matrix::new(vec![vec![2., 3., 5.], vec![7., 1., 4., 1.]]);
    }

    #[test]
    fn is_square() {
        let matrix = Matrix::new(vec![vec![3., 2., 4.], vec![4., 5., 6.]]);
        assert_eq!(matrix.is_square(), false);
        let matrix = Matrix::new(vec![vec![3., 2.], vec![4., 5.]]);
        assert_eq!(matrix.is_square(), true);
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
    fn dot_vec() {
        let mut matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
        assert_eq!(
            matrix.dot_vec(&Vector::new(vec![2., 1., 0.])),
            Vector::new(vec![1., -3.])
        );

        matrix.transpose();
        assert_eq!(
            matrix.dot_vec(&Vector::new(vec![2., 1.])),
            Vector::new(vec![2.0, -5.0, 5.0])
        );
    }

    #[test]
    #[should_panic(expected = "wrong vector shape expected 3, got 2")]
    fn dot_vec_panic() {
        let matrix = Matrix::new(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
        assert_eq!(
            matrix.dot_vec(&Vector::new(vec![2., 1.])),
            Vector::new(vec![1., -3.])
        );
    }

    #[test]
    fn apply_func_val() {
        let mut matrix = Matrix::new(vec![vec![0.7, 0.2, 0.3], vec![0.5, 0.6, 0.1]]);
        let step: Box<(dyn Fn(f32) -> f32 + 'static)> = Box::new(|x: f32| -> f32 {
            if x > 0.5 {
                1.
            } else {
                0.
            }
        });
        matrix.apply_func_val(&step);
        assert_eq!(matrix.matrix_flatt().vec(), vec![1., 0., 0., 0., 1., 0.]);
    }

    #[test]
    fn sum() {
        let matrix = Matrix::new(vec![vec![3., 1.], vec![5., 3.]]);
        assert_eq!(matrix.sum(), 12.);
    }

    #[test]
    fn sum_vec() {
        let matrix = Matrix::new(vec![vec![3., 1.], vec![5., 3.]]);
        assert_eq!(matrix.sum_vec(), Vector::new(vec![8.0, 4.0]));
    }
}
