[[block]]
struct Vector {
    // size: f32;
    data: [[stride(4)]] array<f32>;
}; 

[[group(0), binding(0)]]
var<storage, read_write> vector: Vector;

fn add_1(input: f32) -> f32{
    return input + 1.;
}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    vector.data[global_id.x] = add_1(vector.data[global_id.x]);
}
