#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use math::gpu::Gpu;

    #[test]
    fn new() {
        let gpu = pollster::block_on(Gpu::new());
        assert_eq!(gpu.is_ok(), true);
    }
}
