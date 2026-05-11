// Reduce sum-of-squares para gradient clipping global.
//
// Lê um buffer de gradientes `g` (bytes f32 stored as u32 via atomic CAS no
// backward), computa sum(g[i]²) por workgroup em LDS, e atomic-adiciona ao
// accumulator global `norm_sq_acc` (1-element atomic<u32>, bitcast f32).
//
// Dispatchado uma vez POR tensor de gradiente (7×). Acumulator é zerado
// antes do primeiro dispatch via shader `zero_acc`.

struct ParamsNorm {
    length: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: ParamsNorm;
@group(0) @binding(1) var<storage, read> g: array<u32>;            // bitcast f32
@group(0) @binding(2) var<storage, read_write> norm_sq_acc: array<atomic<u32>>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn sumsq_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = lid.x;
    var local_sum: f32 = 0.0;
    if (gid.x < params.length) {
        let v = bitcast<f32>(g[gid.x]);
        local_sum = v * v;
    }
    partial[tid] = local_sum;
    workgroupBarrier();

    // Reduce binária em LDS (256→1).
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 atomic-add ao accumulator global.
    if (tid == 0u) {
        let val = partial[0];
        if (val == 0.0) { return; }
        var old_u = atomicLoad(&norm_sq_acc[0]);
        loop {
            let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
            let res = atomicCompareExchangeWeak(&norm_sq_acc[0], old_u, new_u);
            if (res.exchanged) { break; }
            old_u = res.old_value;
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn zero_acc() {
    atomicStore(&norm_sq_acc[0], 0u);
}
