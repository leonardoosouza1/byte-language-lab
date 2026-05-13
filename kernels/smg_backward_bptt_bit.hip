// smg_backward_bptt.hip — BPTT backward sem atomics para g_t_f/g_lat_w/g_w_out.
//
// SPLIT de smg_backward.hip em 2 fases para eliminar 14M atomics/passo:
//
//   Fase 1 (este kernel): propagação BPTT.
//     - Computa d_pre[B, T, FLAT_DIM] (para grad_acc usar depois)
//     - Computa d_logits[B, V]        (para grad_acc calcular g_w_out)
//     - Escrita atômica apenas dos tensores pequenos:
//         g_i_f     [8, V, K]    — contendido por token (baixo conflito, ~3/elem)
//         g_nibble_h [8, 16, K] — high nibble: ~192/elem (alto mas correcto)
//         g_nibble_l [8, 16, K] — low  nibble: ~192/elem (alto mas correcto)
//         g_b_out   [V]          — 256 elementos, 1 WG por elem.
//   Fase 2 (smg_grad_acc.hip): g_t_f, g_lat_w, g_w_out via GEMM-like batch-reduce.
//
// Contention de nibble (POC):
//   16 bytes → mesmo high_nibble × batch=64 × T=48 = 49152 writes para 16 entradas
//   → ~3072 writes/entrada → CAS-loop correcto mas potencialmente lento.
//   Para produção: mover nibble para grad_acc_nibble (como b_f/pos → grad_acc_b_f/pos).
//
// LDS layout com padding (STATE_STRIDE = K+1 = 33) para eliminar bank conflicts 32-way:
//   state[tid * STATE_STRIDE + k]  ← bank = (tid × 33 + k) % 32 = (tid + k) % 32 ≠ conflito
//
// Kernarg layout (112B → 120B com nibble):
//   offset  0: ParamsBwd   (16 B)
//   offset 16: t_f         (8 B)
//   offset 24: lat_w       (8 B)
//   offset 32: w_out       (8 B)
//   offset 40: states_all  (8 B)
//   offset 48: logits_in   (8 B)
//   offset 56: windows     (8 B)
//   offset 64: targets     (8 B)
//   offset 72: g_i_f       (8 B) — atomic u32
//   offset 80: g_nibble_h  (8 B) — atomic u32, NEW
//   offset 88: g_nibble_l  (8 B) — atomic u32, NEW
//   offset 96: g_b_out     (8 B) — atomic u32
//   offset 104: d_pre_out  (8 B)
//   offset 112: d_logits_out (8 B)
//   total: 120 B
//
// Layout da grade:
//   Grid  : (batch_size, 1, 1)
//   Block : (128, 1, 1)

#include <hip/hip_runtime.h>
#include <math.h>

#define K              32u
#define N_FAMILIES     8u
#define FSMS_PER_CELL  8u
#define N_CELLS        16u
#define GRID_SIDE      4u
#define FLAT_DIM       4096u
#define ALPHA          0.1f
#define STATE_STRIDE   33u   // K + 1: elimina bank conflicts 32-way em LDS

struct ParamsBwd {
    uint32_t vocab_size;
    uint32_t t_steps;
    uint32_t batch_size;
    uint32_t _pad;
};

__device__ __forceinline__ float u32_as_f32(uint32_t u) {
    float f; __builtin_memcpy(&f, &u, sizeof(float)); return f;
}
__device__ __forceinline__ uint32_t f32_as_u32(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, sizeof(uint32_t)); return u;
}
__device__ __forceinline__ void atomic_add_f32(uint32_t* __restrict__ addr, float val) {
    if (val == 0.0f) return;
    uint32_t old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, f32_as_u32(u32_as_f32(assumed) + val));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// LDS padded — d_state e d_pre_buf com STATE_STRIDE=33 por thread.
// 128 × 33 × 4 = 16,896 bytes por array; 2 arrays = 33,792 bytes.
// d_logits: V ≤ 1024; 4 KB (sem conflito: broadcast por j).
// Total LDS: 33,792 + 4,096 = ~37.9 KB < 64 KB.
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(128, 1)
backward_bptt_bit(
    ParamsBwd        params,
    const float* __restrict__    t_f,        // [N_FAMILIES, K, K]
    const float* __restrict__    lat_w,      // [N_FAMILIES, K, K]
    const float* __restrict__    w_out,      // [FLAT_DIM, vocab_size]
    const float* __restrict__    states_all, // [B, T+1, FLAT_DIM]
    const float* __restrict__    logits_in,  // [B, V]
    const uint32_t* __restrict__ windows,    // [B, T]
    const uint32_t* __restrict__ targets,    // [B]
    uint32_t* __restrict__ g_i_f,            // atomic u32 [8, V, K]   — baixo conflito
    uint32_t* __restrict__ g_nibble_h,       // atomic u32 [8, 16, K]  — high nibble grads
    uint32_t* __restrict__ g_nibble_l,       // atomic u32 [8, 16, K]  — low  nibble grads
    uint32_t* __restrict__ g_b_out,          // atomic u32 [V]          — muito baixo conflito
    float* __restrict__    d_pre_out,        // [B, T, FLAT_DIM]  (NÃO atômico)
    float* __restrict__    d_logits_out      // [B, V]             (NÃO atômico)
) {
    // LDS padded (bank-conflict-free)
    __shared__ float d_state_lds[128 * STATE_STRIDE];   // 16,896 bytes
    __shared__ float d_pre_lds[128 * STATE_STRIDE];     // 16,896 bytes
    __shared__ float d_logits[1024];                    // ≤ V * 4 = 4,096 bytes

    const uint32_t sample = blockIdx.x;
    if (sample >= params.batch_size) return;

    const uint32_t tid  = threadIdx.x;
    const uint32_t cell = tid / FSMS_PER_CELL;
    const uint32_t fsm  = tid % FSMS_PER_CELL;
    const uint32_t f    = fsm;
    const uint32_t v    = params.vocab_size;
    const uint32_t ts   = params.t_steps;

    // Índice LDS padded para este thread.
    const uint32_t lds_base = tid * STATE_STRIDE;
    // Índice flat no espaço de parâmetros (para gradientes de parâmetros).
    const uint32_t cf_base  = tid * K;   // offset sem padding (para leitura de states_all etc.)

    const uint32_t row    = cell / GRID_SIDE;
    const uint32_t col    = cell % GRID_SIDE;
    const uint32_t n_up   = ((row + GRID_SIDE - 1u) % GRID_SIDE) * GRID_SIDE + col;
    const uint32_t n_down = ((row + 1u) % GRID_SIDE) * GRID_SIDE + col;
    const uint32_t n_left = row * GRID_SIDE + (col + GRID_SIDE - 1u) % GRID_SIDE;
    const uint32_t n_rght = row * GRID_SIDE + (col + 1u) % GRID_SIDE;

    // Índices LDS dos vizinhos (com padding).
    const uint32_t up_lds   = (n_up   * FSMS_PER_CELL + fsm) * STATE_STRIDE;
    const uint32_t down_lds = (n_down * FSMS_PER_CELL + fsm) * STATE_STRIDE;
    const uint32_t left_lds = (n_left * FSMS_PER_CELL + fsm) * STATE_STRIDE;
    const uint32_t rght_lds = (n_rght * FSMS_PER_CELL + fsm) * STATE_STRIDE;

    const uint32_t t_off   = f * K * K;

    // ---- HEAD: softmax 256-way + BIT-LEVEL BCE (marginal) ----
    // σ_i = sum_{j: bit_i(j)=1} P(byte=j)
    // c_i = (σ_i - t_i) / (σ_i*(1-σ_i))
    // K   = sum_i c_i * σ_i
    // per_byte[j] = sum_i c_i * 1[bit_i(j)=1]
    // d_logits[j] = P(byte=j) * (per_byte[j] - K)
    if (tid == 0u) {
        const uint32_t logits_off = sample * v;
        const uint32_t tgt_byte   = targets[sample] & 0xFFu;

        // 1) softmax → d_logits[j] = P(byte=j)
        float mx = -1e30f;
        for (uint32_t j = 0; j < v; j++) if (logits_in[logits_off + j] > mx) mx = logits_in[logits_off + j];
        float s = 0.0f;
        for (uint32_t j = 0; j < v; j++) { d_logits[j] = expf(logits_in[logits_off + j] - mx); s += d_logits[j]; }
        const float inv_s = 1.0f / s;
        for (uint32_t j = 0; j < v; j++) d_logits[j] *= inv_s;

        // 2) σ_i marginal por bit (i=0..7, bit 0 = LSB no convention bit_i(j) = (j>>(7-i))&1)
        float sigma[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t j = 0; j < v; j++) {
            const float pj = d_logits[j];
            #pragma unroll
            for (uint32_t i = 0; i < 8; i++) {
                if ((j >> (7u - i)) & 1u) sigma[i] += pj;
            }
        }

        // 3) c_i = (σ_i - t_i) / (σ_clamped*(1-σ_clamped))
        //    Clamp σ em [0.05, 0.95] ANTES do denominador pra evitar explosão
        //    quando bit prob saturada. Mantém c_i ∈ [-21, 21] aprox.
        //    Escala 1/8 pra normalizar pelo número de bits (mantém grad ~O(1)
        //    como CE).
        float c[8];
        #pragma unroll
        for (uint32_t i = 0; i < 8; i++) {
            const float t_i = (float)((tgt_byte >> (7u - i)) & 1u);
            const float si  = sigma[i];
            const float sc  = fmaxf(0.05f, fminf(si, 0.95f));
            const float den = sc * (1.0f - sc);
            c[i] = (si - t_i) / den * 0.125f;  // /8 pra mesma escala que CE
        }

        // 4) K = sum_i c_i * σ_i
        float kconst = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8; i++) kconst += c[i] * sigma[i];

        // 5) d_logits[j] = P(byte=j) * (per_byte[j] - K)
        for (uint32_t j = 0; j < v; j++) {
            float per_byte = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 8; i++) {
                if ((j >> (7u - i)) & 1u) per_byte += c[i];
            }
            d_logits[j] = d_logits[j] * (per_byte - kconst);
        }
    }
    __syncthreads();

    // g_b_out: small tensor, keep atomic (256 elements, low-overhead).
    {
        const uint32_t per_v = (v + 127u) / 128u;
        const uint32_t j0 = tid * per_v, j1 = min(j0 + per_v, v);
        for (uint32_t j = j0; j < j1; j++) atomic_add_f32(&g_b_out[j], d_logits[j]);
    }

    // d_state_lds = sum_j w_out[cf_base+k, j] * d_logits[j].
    // (g_w_out omitido aqui — calculado em grad_acc via GEMM)
    {
        const uint32_t s_final_off = (sample * (ts + 1u) + ts) * FLAT_DIM;
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t i     = cf_base + k;
            const uint32_t w_row = i * v;
            float d_flat_i = 0.0f;
            for (uint32_t j = 0; j < v; j++) {
                d_flat_i += w_out[w_row + j] * d_logits[j];
            }
            d_state_lds[lds_base + k] = d_flat_i;
            (void)s_final_off;
        }
    }
    __syncthreads();

    // Escrita d_logits_out para uso em grad_acc (cálculo de g_w_out).
    {
        const uint32_t per_v = (v + 127u) / 128u;
        const uint32_t j0 = tid * per_v, j1 = min(j0 + per_v, v);
        for (uint32_t j = j0; j < j1; j++) d_logits_out[sample * v + j] = d_logits[j];
    }
    __syncthreads();

    // ---- BPTT: ticks reversos ----
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
            dsn_local[k]   = d_state_lds[lds_base + k];
        }

        // d_pre_local (registadores) + d_state_prev_local.
        float d_pre_local[K], d_state_prev_local[K];
        for (uint32_t k = 0; k < K; k++) {
            d_state_prev_local[k] = (1.0f - ALPHA) * dsn_local[k];
            const float tp = (s_out_local[k] - (1.0f - ALPHA) * s_in_local[k]) / ALPHA;
            d_pre_local[k] = ALPHA * (1.0f - tp * tp) * dsn_local[k];
        }

        // g_i_f (65536 elem, baixo conflito ~3 ops/elem).
        for (uint32_t k = 0; k < K; k++) {
            atomic_add_f32(&g_i_f[i_off + k], d_pre_local[k]);
        }

        // g_nibble_h + g_nibble_l: mesmo gradiente que g_i_f.
        // grad flui para AMBOS i_f e nibble (embedding aditivo).
        // Contention mais alta (16 bytes/nibble × B×T) mas correcto para POC.
        {
            const uint32_t hn     = input_byte >> 4u;         // 0..15
            const uint32_t ln     = input_byte & 0x0Fu;       // 0..15
            const uint32_t nh_off = (f * 16u + hn) * K;
            const uint32_t nl_off = (f * 16u + ln) * K;
            for (uint32_t k = 0; k < K; k++) {
                atomic_add_f32(&g_nibble_h[nh_off + k], d_pre_local[k]);
                atomic_add_f32(&g_nibble_l[nl_off + k], d_pre_local[k]);
            }
        }

        // Publica d_pre_local em LDS para fase lateral.
        for (uint32_t k = 0; k < K; k++) d_pre_lds[lds_base + k] = d_pre_local[k];
        __syncthreads();

        // d_state_prev: recurrência T_f^T @ d_pre (sem atomic).
        for (uint32_t j = 0; j < K; j++) {
            const uint32_t row_off = t_off + j * K;
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) acc += t_f[row_off + k] * d_pre_local[k];
            d_state_prev_local[j] += acc;
        }
        __syncthreads();  // d_pre_lds visível para fase lateral

        // Lateral: pool 4 vizinhos (usa d_pre_lds padded — sem bank conflict).
        float pool_local[K];
        for (uint32_t k = 0; k < K; k++) {
            pool_local[k] = d_pre_lds[up_lds + k] + d_pre_lds[down_lds + k]
                          + d_pre_lds[left_lds + k] + d_pre_lds[rght_lds + k];
        }
        // lat_w backward (sem atomic para g_lat_w — grad_acc cuida disso).
        for (uint32_t j = 0; j < K; j++) {
            const uint32_t row_off = t_off + j * K;  // t_off = lat_off (mesmo stride)
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) acc += lat_w[row_off + k] * pool_local[k];
            d_state_prev_local[j] += acc;
        }

        // Commit d_state_prev → LDS.
        for (uint32_t k = 0; k < K; k++) d_state_lds[lds_base + k] = d_state_prev_local[k];

        // Escreve d_pre_local → VRAM d_pre_out[sample, step, cf_base+k].
        const uint32_t dpre_off = (sample * ts + step) * FLAT_DIM + cf_base;
        for (uint32_t k = 0; k < K; k++) d_pre_out[dpre_off + k] = d_pre_local[k];

        __syncthreads();
    }
}
