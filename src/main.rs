use math::linear_algebra::Vector;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
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

    pub fn init_compute_pipeline(&self, shader: &str) -> wgpu::ComputePipeline {
        let cs_module = self
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    std::fs::read_to_string(shader).unwrap().as_str(),
                )),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &cs_module,
                    entry_point: "main",
                });

        compute_pipeline
    }

    pub fn result_buffer(&self, size: wgpu::BufferAddress) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

async fn run() {
    let vector = Vector::new(vec![1., 1., 1., 1., 1., 1.]);
    let out = execute_gpu(vector).await.unwrap();
    println!("output: {}", out);
}

async fn execute_gpu(vector: Vector) -> Option<Vector> {
    let gpu = Gpu::new().await.unwrap();
    execute_gpu_inner(&gpu, vector).await
}

fn create_encoder(
    gpu: &Gpu,
    size: u64,
    compute_pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
) -> wgpu::CommandEncoder {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(compute_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(size as u32, 1, 1);
    }
    encoder
}

async fn execute_gpu_inner(gpu: &Gpu, vector: Vector) -> Option<Vector> {
    let vec_size = vector.bytes()[..4].to_vec();

    let contents = &vector.bytes()[4..];
    let slice_size = contents.len();
    let size = slice_size as wgpu::BufferAddress;

    let storage_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

    let compute_pipeline = gpu.init_compute_pipeline("src/shader.wgsl");

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let result_buffer = gpu.result_buffer(size);
    let mut encoder = create_encoder(gpu, size, &compute_pipeline, &bind_group);
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &result_buffer, 0, size);
    gpu.queue.submit(Some(encoder.finish()));

    let buffer_slice = result_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    gpu.device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        let mut bytes = vec_size;
        data.to_vec().iter().for_each(|&x| bytes.push(x));
        let result = Vector::new_bytes(bytes).unwrap();

        drop(data);
        result_buffer.unmap();

        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    pollster::block_on(run());
}
