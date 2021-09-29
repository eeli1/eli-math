#[cfg(test)]
mod tests {
    use math::linear_algebra::Vector;

    #[test]
    fn set_index() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.set_index(1, 10.);
        assert_eq!(vector.vec(), vec![2.0, 10.0, 5.0]);
    }

    #[test]
    fn dist() {
        let vector1 = Vector::new(vec![2., 7., 1.]);
        let vector2 = Vector::new(vec![8., 2., 8.]);
        assert_eq!(vector1.dist(&vector2), 10.488089);
    }

    #[test]
    fn limit() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.limit(2.);
        assert_eq!(vector.mag(), 2.);

        vector.limit(3.);
        assert_eq!(vector.mag(), 2.);
    }

    #[test]
    fn set_mag() {
        let mut vector = Vector::new(vec![2., 3., 5.]);
        vector.set_mag(4.);
        assert_eq!(vector.mag(), 4.);
        assert_eq!(vector.vec(), vec![1.2977713, 1.946657, 3.2444284]);
    }

    #[test]
    fn new_zero() {
        let vector = Vector::new_zero(4);
        assert_eq!(vector.vec(), vec![0., 0., 0., 0.]);
    }

    #[test]
    fn ops_add() {
        let vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        let result = Vector::new(vec![8.0, 9.0, 7.0]);

        assert_eq!(vector1 + vector2, result);
    }

    #[test]
    fn ops_sub() {
        let vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        let result = Vector::new(vec![-4.0, 3.0, -1.0]);

        assert_eq!(vector1 - vector2, result);
    }

    #[test]
    fn ops_mul() {
        let vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        let result = Vector::new(vec![12.0, 18.0, 12.0]);

        assert_eq!(vector1 * vector2, result);
    }

    #[test]
    fn ops_div() {
        let vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        let result = Vector::new(vec![0.33333334, 2.0, 0.75]);

        assert_eq!(vector1 / vector2, result);
    }

    #[test]
    fn add_assign() {
        let mut vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        vector1 += vector2;
        assert_eq!(vector1, Vector::new(vec![8.0, 9.0, 7.0]));
    }

    #[test]
    fn sub_assign() {
        let mut vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        vector1 -= vector2;
        assert_eq!(vector1, Vector::new(vec![-4.0, 3.0, -1.0]));
    }

    #[test]
    fn mul_assign() {
        let mut vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        vector1 *= vector2;
        assert_eq!(vector1, Vector::new(vec![12.0, 18.0, 12.0]));
    }

    #[test]
    fn div_assign() {
        let mut vector1 = Vector::new(vec![2., 6., 3.]);
        let vector2 = Vector::new(vec![6., 3., 4.]);
        vector1 /= vector2;
        assert_eq!(vector1, Vector::new(vec![0.33333334, 2.0, 0.75]));
    }

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

    #[cfg(feature = "gpu")]
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

    #[test]
    fn apply_func() {
        let mut vector = Vector::new(vec![0.7, 0.2, 0.3]);
        let step: Box<(dyn Fn(f32) -> f32 + 'static)> = Box::new(|x: f32| -> f32 {
            if x > 0.5 {
                1.
            } else {
                0.
            }
        });
        vector.apply_func(&step);
        assert_eq!(vector.vec(), vec![1., 0., 0.]);
    }

    #[test]
    fn sum() {
        let vector = Vector::new(vec![3., 1., 3., 1.]);
        assert_eq!(vector.sum(), 8.);
    }

    #[test]
    fn new_one_hot() {
        let vector = Vector::new_one_hot(2, 5);
        assert_eq!(vector.vec(), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn argmax() {
        let vector = Vector::new(vec![3., 2., 10., 4.]);
        assert_eq!(vector.argmax(), 2);
    }

    #[test]
    fn mean() {
        let vector = Vector::new(vec![3., 2., 10., 4.]);
        assert_eq!(vector.mean(), 4.75);
    }
}
