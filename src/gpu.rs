use std::borrow::Cow;
use std::fs;
use wgpu::util::DeviceExt;

pub struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    /// creats new gpu instance
    ///
    /// ## Example
    ///
    /// ```rust
    /// use math::gpu::Gpu;
    /// let gpu = pollster::block_on(Gpu::new());
    /// assert_eq!(gpu.is_ok(), true);
    /// ```
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            let device_result = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await;

            match device_result {
                Ok((device, queue)) => Ok(Self {
                    adapter,
                    device,
                    queue,
                }),
                Err(err) => Err(format!("{:?}", err)),
            }
        } else {
            Err("the adapter couldn't be initialized".to_string())
        }
    }

    /// with this function, you can upload arbitrary data (as bytes) to the GPU and execute it whit a (wgsl) compute shader
    /// the output is also in bytes that you can then map to your specific data type
    ///
    /// ## Example
    ///
    /// #```rust
    /// let gpu = gpu::Gpu::new().await;
    ///
    /// let vec1 = vec![1., 2., 3.];
    /// let vec2 = vec![3., 2., 1.];
    ///
    /// let size = 3 * std::mem::size_of::<f32>();
    ///
    /// let bytes = gpu
    ///     .execute(
    ///         vec![bytemuck::cast_slice(&vec1), bytemuck::cast_slice(&vec2)],
    ///         size,
    ///         "./src/add_vec.wgsl",
    ///         vec![size as u32, 1, 1],
    ///     )
    ///     .await;
    ///
    /// let result: Vec<f32> = bytes
    ///     .chunks_exact(4)
    ///     .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
    ///    .collect();
    /// #```
    ///
    /// !! note !!
    /// that the last binding is always the result buffer
    /// the entry point in the shader is always "main"
    pub async fn execute(
        &self,
        in_data: Vec<&[u8]>,
        out_size: usize,
        shader: &str,
        cells: Vec<u32>,
    ) -> Vec<u8> {
        // init pipeline and add shader
        let pipeline = self.create_pipeline(shader);

        // setup buffer
        let mut data_buffers = Vec::with_capacity(in_data.len() + 1);

        let mut in_data_buffer_value = Vec::with_capacity(in_data.len());
        for (i, data) in in_data.iter().enumerate() {
            let storage_buffer = self.create_storage_buffer(
                data,
                Some(format!("Storage Buffer {}", i).as_ref()),
                false,
            );
            in_data_buffer_value.push(storage_buffer);
        }

        in_data_buffer_value
            .iter()
            .for_each(|data| data_buffers.push(data));

        let result_buffer = self.create_storage_buffer(
            bytemuck::cast_slice(&vec![0 as u8; out_size]),
            Some("Storage Buffer Result"),
            true,
        );

        data_buffers.push(&result_buffer);

        // create bindigs
        let bind_group = self.create_bind_group(data_buffers, &pipeline);

        // retrieve data
        let bytes = self
            .get_bytes(&result_buffer, out_size, &pipeline, &bind_group, cells)
            .await;
        bytes
    }

    fn load_cs_mod(&self, file_path: &str) -> wgpu::ShaderModule {
        let cs_module = self
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    fs::read_to_string(file_path).unwrap().as_str(),
                )),
            });
        cs_module
    }

    fn create_pipeline(&self, shader: &str) -> wgpu::ComputePipeline {
        let cs_module = self.load_cs_mod(shader);

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point: "main",
            })
    }

    fn create_bind_group(
        &self,
        data_buffers: Vec<&wgpu::Buffer>,
        pipeline: &wgpu::ComputePipeline,
    ) -> wgpu::BindGroup {
        let mut entries = Vec::new();
        for (index, buffer) in data_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
        }

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });

        bind_group
    }

    async fn get_bytes(
        &self,
        source: &wgpu::Buffer,
        size: usize,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        cells: Vec<u32>,
    ) -> Vec<u8> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch(cells[0], cells[1], cells[2]); // Number of cells to run, the (x,y,z) size of item being processed
        }

        encoder.copy_buffer_to_buffer(&source, 0, &staging_buffer, 0, size as wgpu::BufferAddress);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let data = buffer_slice.get_mapped_range();
            let result = data.to_vec();

            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    fn create_storage_buffer(
        &self,
        bytes: &[u8],
        name: Option<&str>,
        is_output: bool,
    ) -> wgpu::Buffer {
        let mut flags = wgpu::BufferUsages::STORAGE;
        if is_output {
            flags |= wgpu::BufferUsages::COPY_SRC;
        }

        let storage_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: name,
                contents: bytes,
                usage: flags,
            });
        storage_buffer
    }
}
