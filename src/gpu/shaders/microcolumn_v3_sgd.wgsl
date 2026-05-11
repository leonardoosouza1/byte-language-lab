// SGD shader genérico: w[i] -= scale * bitcast<f32>(g[i]).
//
// Os grad buffers são `array<atomic<u32>>` (escritos via atomic_add_f32 CAS no
// backward). Aqui lemos com atomicLoad e bitcast f32 sem novas writes — não
// precisa atomic.
//
// Dispatchado uma vez POR tensor (rebindando os buffers w/g). `length` controla
// quantos elementos processar (último workgroup pode ser parcial).

struct ParamsSgd {
    length: u32,
    scale_bits: u32,  // bitcast<f32>; permite passar scale como u32 no uniform
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ParamsSgd;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn sgd_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }
    let scale = bitcast<f32>(params.scale_bits);
    let g_val = bitcast<f32>(atomicLoad(&g[i]));
    w[i] = w[i] - scale * g_val;
}

// Zera grads (storage<atomic<u32>>) — usado entre steps.
@compute @workgroup_size(256, 1, 1)
fn zero_grads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }
    atomicStore(&g[i], 0u);
}
