// smg_sgd.hip — SGD step + grad zeroing para SMG v3.
//
// Port de `src/gpu/shaders/microcolumn_v3_sgd.wgsl`.
// Alvo: RDNA2 gfx1031. Compilar com hipcc --genco --offload-arch=gfx1031 ...
//
// Dois entry points:
//   sgd_main    — w[i] -= scale * f32_bits(g[i]);  depois zera g[i]=0
//   zero_grads  — zera todos os g[i] sem step (entre backward calls)
//
// Kernarg layout (sgd_main):
//   offset  0: ParamsSgd   (16 B, 4×u32)
//   offset 16: w           (8 B ptr, float*)
//   offset 24: g           (8 B ptr, uint32_t* — gradientes como bit-pattern f32)
//   explicit: 32 B + hidden args ≈ 88 B

#include <hip/hip_runtime.h>
#include <math.h>

struct ParamsSgd {
    uint32_t length;
    uint32_t scale_bits;   // bitcast f32; permite passar scale como u32
    uint32_t _pad0, _pad1;
};

__device__ __forceinline__ float u32_as_f32(uint32_t u) {
    float f; __builtin_memcpy(&f, &u, sizeof(float)); return f;
}

// ---------------------------------------------------------------------------
// sgd_main: w[i] -= scale * g[i]; depois g[i] = 0 (zero inline).
// Combinar em 1 kernel = 1 round-trip a menos vs. dispatch separado.
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(256)
sgd_main(
    ParamsSgd               params,
    float* __restrict__     w,     // parâmetros (inplace update)
    uint32_t* __restrict__  g      // gradiente (bit-pattern f32; zerado após leitura)
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.length) return;

    const float scale = u32_as_f32(params.scale_bits);
    const float g_val = u32_as_f32(g[i]);
    w[i] -= scale * g_val;
    // Zera gradiente inline — poupa um dispatch separado de zero_grads.
    g[i] = 0u;
}

// ---------------------------------------------------------------------------
// zero_grads: zera todos os g[i] — usado quando queremos zero sem step
// (ex.: início do treino ou descarte de batch).
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(256)
zero_grads(
    ParamsSgd               params,
    float* __restrict__     w,     // não usado; mantido para layout idêntico ao sgd_main
    uint32_t* __restrict__  g
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.length) return;
    g[i] = 0u;
    (void)w;
}
