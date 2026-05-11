// smg_backward.hip — Backward pass do SMG v3 (BPTT T=48 ticks).
//
// Port de `src/gpu/shaders/microcolumn_v3_backward.wgsl`.
// Alvo: RDNA2 gfx1031. Compilar com:
//   hipcc --genco --offload-arch=gfx1031 -mwavefrontsize32 -mno-cumode -O3 -ffast-math
//
// Layout:
//   Grid  : (batch_size, 1, 1)
//   Block : (128, 1, 1)
//   LDS   : d_state(16KB) + d_pre_all(16KB) + d_logits(4KB) = 36KB
//
// f32 Atomic add via CAS-loop:
//   RDNA2 NÃO tem `global_atomic_add_f32` nativo (só em RDNA3+/CDNA2+).
//   Padrão: atomicCAS loop sobre bit-pattern u32.
//   Ver docs/research/memory-model.md §3 (CAS loop confirmado gfx1031).
//
// Kernarg layout (explícito):
//   offset  0: ParamsBwd  (16 B)
//   offset 16: t_f        ptr (8 B)
//   offset 24: lat_w      ptr (8 B)
//   offset 32: w_out      ptr (8 B)
//   offset 40: states_all ptr (8 B)
//   offset 48: logits_in  ptr (8 B)
//   offset 56: windows    ptr (8 B)
//   offset 64: targets    ptr (8 B)
//   offset 72: g_t_f      ptr (8 B)
//   offset 80: g_i_f      ptr (8 B)
//   offset 88: g_b_f      ptr (8 B)
//   offset 96: g_pos      ptr (8 B)
//   offset 104: g_lat_w   ptr (8 B)
//   offset 112: g_w_out   ptr (8 B)
//   offset 120: g_b_out   ptr (8 B)
//   explicit total: 128 B + hidden args ≈ 184 B

#include <hip/hip_runtime.h>
#include <math.h>

#define K              32u
#define N_FAMILIES     8u
#define FSMS_PER_CELL  8u
#define N_CELLS        16u
#define GRID_SIDE      4u
#define FLAT_DIM       4096u
#define ALPHA          0.1f

struct ParamsBwd {
    uint32_t vocab_size;
    uint32_t t_steps;
    uint32_t batch_size;
    uint32_t _pad;
};

// ---------------------------------------------------------------------------
// Atomic f32 add via CAS-loop (RDNA2 não tem global_atomic_add_f32 nativo).
// Buffers de gradiente são uint32_t* (bit-pattern de f32 via atomicCAS).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float u32_as_f32(uint32_t u) {
    float f; __builtin_memcpy(&f, &u, sizeof(float)); return f;
}
__device__ __forceinline__ uint32_t f32_as_u32(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, sizeof(uint32_t)); return u;
}

__device__ __forceinline__ void atomic_add_f32(uint32_t* __restrict__ addr, float val) {
    if (val == 0.0f) return;
    uint32_t old = *addr;
    uint32_t assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, f32_as_u32(u32_as_f32(assumed) + val));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// Kernel principal
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(128, 2)
backward_main(
    ParamsBwd        params,
    const float* __restrict__    t_f,       // [N_FAMILIES, K, K]
    const float* __restrict__    lat_w,     // [N_FAMILIES, K, K]
    const float* __restrict__    w_out,     // [FLAT_DIM, vocab_size]
    const float* __restrict__    states_all,// [B, T+1, FLAT_DIM]
    const float* __restrict__    logits_in, // [B, V]
    const uint32_t* __restrict__ windows,   // [B, T]
    const uint32_t* __restrict__ targets,   // [B]
    uint32_t* __restrict__ g_t_f,           // grad t_f   (atomic u32)
    uint32_t* __restrict__ g_i_f,           // grad i_f
    uint32_t* __restrict__ g_b_f,           // grad b_f
    uint32_t* __restrict__ g_pos,           // grad pos_bias
    uint32_t* __restrict__ g_lat_w,         // grad lat_w
    uint32_t* __restrict__ g_w_out,         // grad w_out
    uint32_t* __restrict__ g_b_out          // grad b_out
) {
    // ---- LDS: 36 KB (d_state 16KB + d_pre_all 16KB + d_logits 4KB) ----
    __shared__ float d_state[FLAT_DIM];    // 4096 f32 = 16 KB
    __shared__ float d_pre_all[FLAT_DIM];  // 4096 f32 = 16 KB
    __shared__ float d_logits[1024];       // V ≤ 1024; 4 KB

    const uint32_t sample = blockIdx.x;
    if (sample >= params.batch_size) return;

    const uint32_t tid  = threadIdx.x;
    const uint32_t cell = tid / FSMS_PER_CELL;
    const uint32_t fsm  = tid % FSMS_PER_CELL;
    const uint32_t f    = fsm;
    const uint32_t v    = params.vocab_size;
    const uint32_t ts   = params.t_steps;
    const uint32_t cf_base = (cell * FSMS_PER_CELL + fsm) * K;

    const uint32_t row    = cell / GRID_SIDE;
    const uint32_t col    = cell % GRID_SIDE;
    const uint32_t n_up   = ((row + GRID_SIDE - 1u) % GRID_SIDE) * GRID_SIDE + col;
    const uint32_t n_down = ((row + 1u) % GRID_SIDE) * GRID_SIDE + col;
    const uint32_t n_left = row * GRID_SIDE + (col + GRID_SIDE - 1u) % GRID_SIDE;
    const uint32_t n_rght = row * GRID_SIDE + (col + 1u) % GRID_SIDE;

    const uint32_t t_off   = f * K * K;
    const uint32_t b_off   = f * K;
    const uint32_t pos_off = cell * K;
    const uint32_t lat_off = f * K * K;

    // ===== HEAD: softmax + d_logits + g_b_out + g_w_out + d_state =====
    //
    // Thread 0 calcula d_logits[0..V] inteiro (sequencial, 1 vez por sample).
    if (tid == 0u) {
        const uint32_t logits_off = sample * v;
        float mx = -1e30f;
        for (uint32_t j = 0; j < v; j++) {
            float lv = logits_in[logits_off + j];
            if (lv > mx) mx = lv;
        }
        float s = 0.0f;
        for (uint32_t j = 0; j < v; j++) {
            float e = expf(logits_in[logits_off + j] - mx);
            d_logits[j] = e;
            s += e;
        }
        const uint32_t tgt = min(targets[sample], v - 1u);
        for (uint32_t j = 0; j < v; j++) {
            float p = d_logits[j] / s;
            if (j == tgt) p -= 1.0f;
            d_logits[j] = p;
        }
    }
    __syncthreads();

    // g_b_out[j] += d_logits[j] — todos threads cobrem chunks de V.
    {
        const uint32_t per_thread_v = (v + 127u) / 128u;
        const uint32_t j_start = tid * per_thread_v;
        const uint32_t j_end   = min(j_start + per_thread_v, v);
        for (uint32_t j = j_start; j < j_end; j++) {
            atomic_add_f32(&g_b_out[j], d_logits[j]);
        }
    }

    // g_w_out[i, j] += states_all[sample, T, i] * d_logits[j]
    // d_state[i] (semente BPTT) = sum_j w_out[i,j] * d_logits[j]
    {
        const uint32_t s_final_off = (sample * (ts + 1u) + ts) * FLAT_DIM;
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t i   = cf_base + k;
            const float    s_i = states_all[s_final_off + i];
            float          d_flat_i = 0.0f;
            const uint32_t w_row = i * v;
            for (uint32_t j = 0; j < v; j++) {
                const float dl = d_logits[j];
                d_flat_i += w_out[w_row + j] * dl;
                atomic_add_f32(&g_w_out[w_row + j], s_i * dl);
            }
            d_state[i] = d_flat_i;
        }
    }
    __syncthreads();

    // ===== BPTT: walk ticks reverse =====
    for (uint32_t step_inv = 0u; step_inv < ts; step_inv++) {
        const uint32_t step = ts - 1u - step_inv;
        const uint32_t raw        = windows[sample * ts + step];
        const uint32_t input_byte = min(raw, v - 1u);
        const uint32_t i_off      = (f * v + input_byte) * K;
        const uint32_t s_in_off   = (sample * (ts + 1u) + step) * FLAT_DIM;
        const uint32_t s_out_off  = (sample * (ts + 1u) + step + 1u) * FLAT_DIM;

        float s_in_local[K], s_out_local[K], dsn_local[K];
        for (uint32_t k = 0; k < K; k++) {
            s_in_local[k]  = states_all[s_in_off  + cf_base + k];
            s_out_local[k] = states_all[s_out_off + cf_base + k];
            dsn_local[k]   = d_state[cf_base + k];
        }

        // ---- Fase 1: d_pre, params I/b/pos, T_f, recurrence-bw ----
        float d_pre_local[K], d_state_prev_local[K];
        for (uint32_t k = 0; k < K; k++) {
            d_state_prev_local[k] = (1.0f - ALPHA) * dsn_local[k];
            const float tp = (s_out_local[k] - (1.0f - ALPHA) * s_in_local[k]) / ALPHA;
            d_pre_local[k] = ALPHA * (1.0f - tp * tp) * dsn_local[k];
        }

        for (uint32_t k = 0; k < K; k++) {
            atomic_add_f32(&g_i_f[i_off  + k], d_pre_local[k]);
            atomic_add_f32(&g_b_f[b_off  + k], d_pre_local[k]);
            atomic_add_f32(&g_pos[pos_off + k], d_pre_local[k]);
        }

        // Publicar d_pre_local em LDS para fase 2/3.
        for (uint32_t k = 0; k < K; k++) d_pre_all[cf_base + k] = d_pre_local[k];

        // T_f outer-product (g_t_f) + recurrence-bw (d_state_prev += T_f^T @ d_pre).
        for (uint32_t j = 0; j < K; j++) {
            const float    sj      = s_in_local[j];
            const uint32_t row_off = t_off + j * K;
            float          acc     = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                const float dp = d_pre_local[k];
                atomic_add_f32(&g_t_f[row_off + k], sj * dp);
                acc += t_f[row_off + k] * dp;
            }
            d_state_prev_local[j] += acc;
        }
        __syncthreads();  // d_pre_all visível para fase 2/3

        // ---- Fase 2/3: pool 4 vizinhos + g_lat_w + lat_w-bw ----
        float pool_local[K];
        for (uint32_t k = 0; k < K; k++) {
            pool_local[k] =
                  d_pre_all[(n_up   * FSMS_PER_CELL + fsm) * K + k]
                + d_pre_all[(n_down * FSMS_PER_CELL + fsm) * K + k]
                + d_pre_all[(n_left * FSMS_PER_CELL + fsm) * K + k]
                + d_pre_all[(n_rght * FSMS_PER_CELL + fsm) * K + k];
        }
        for (uint32_t j = 0; j < K; j++) {
            const float    sj      = s_in_local[j];
            const uint32_t row_off = lat_off + j * K;
            float          acc     = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                const float pk = pool_local[k];
                atomic_add_f32(&g_lat_w[row_off + k], sj * pk);
                acc += lat_w[row_off + k] * pk;
            }
            d_state_prev_local[j] += acc;
        }
        __syncthreads();

        // d_state_prev → LDS (cada thread escreve seu próprio slot).
        for (uint32_t k = 0; k < K; k++) d_state[cf_base + k] = d_state_prev_local[k];
        __syncthreads();
    }
}
