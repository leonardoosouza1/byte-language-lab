// smg_forward.hip — Forward pass do SMG v3 (Stacked Microcolumn Grid).
//
// Port de `src/gpu/shaders/microcolumn_v3_forward.wgsl` para HIP C++.
// Alvo: RDNA2 gfx1030 (compatível com RX 6750 XT via HSA_OVERRIDE_GFX_VERSION=10.3.0).
// Compilar com:
//   hipcc --genco --offload-arch=gfx1030 -O3 -ffast-math
//
// Layout da grade:
//   Grid  : (batch_size, 1, 1) — 1 workgroup por sample.
//   Block : (128, 1, 1)        — 128 threads = N_CELLS × FSMS_PER_CELL.
//   LDS   : state_curr(16KB) + lat_proj(16KB) = 32KB por workgroup.
//           Fits 2 workgroups/CU nos 40 CUs = 80 workgroups in-flight.
//
// Kernarg layout (explícito, ABI v5):
//   offset  0: ParamsFwd  (16 B, 4×u32)
//   offset 16: t_f        (8 B ptr, float*)
//   offset 24: i_f        (8 B ptr, float*)   [N_FAMILIES, vocab_size, K]
//   offset 32: nibble_h   (8 B ptr, float*)   [N_FAMILIES, 16, K] — high nibble geom.
//   offset 40: nibble_l   (8 B ptr, float*)   [N_FAMILIES, 16, K] — low  nibble geom.
//   offset 48: b_f        (8 B ptr, float*)
//   offset 56: pos_bias   (8 B ptr, float*)
//   offset 64: lat_w      (8 B ptr, float*)
//   offset 72: w_out      (8 B ptr, float*)
//   offset 80: b_out      (8 B ptr, float*)
//   offset 88: windows    (8 B ptr, uint32_t*)
//   offset 96: logits_out (8 B ptr, float*)
//   offset 104: states_out (8 B ptr, float*)
//   explicit total: 112 B
//
// Geometria de nibble:
//   byte b decompõe-se em (high_nibble = b>>4, low_nibble = b&0xF).
//   Bytes no mesmo high_nibble partilham zona ASCII (ex: 0x4_ = A-O maiúsc.).
//   Bytes com mesmo low_nibble têm a mesma "posição" na zona.
//   embed(b) = i_f[b] + nibble_h[b>>4] + nibble_l[b&0xF]
//   → 'A'(0x41) e 'a'(0x61) diferem em nibble_h[4] vs nibble_h[6]
//     mas partilham nibble_l[1] → inductive bias de case grátis.

#include <hip/hip_runtime.h>
#include <math.h>

#define K              32u
#define N_FAMILIES     8u
#define FSMS_PER_CELL  8u
#define N_CELLS        16u
#define GRID_SIDE      4u
#define FLAT_DIM       4096u       // N_CELLS * FSMS_PER_CELL * K
#define ALPHA          0.1f

struct ParamsFwd {
    uint32_t vocab_size;
    uint32_t t_steps;
    uint32_t batch_size;
    uint32_t cache_states;   // 0 = inference, 1 = training (write states_out)
};

extern "C" __global__ void
__launch_bounds__(128, 1)   // maxThreadsPerBlock=128, minBlocksPerSM=1
// NOTE: nibble embedding aumentou VGPRs para 128/wave. Com wave32 e 256 VGPRs/SIMD32,
// só cabe 1 workgroup/CU (2 waves/SIMD32 × 2 SIMD32 = 4 waves = 128 threads).
// minBlocksPerSM=2 exigiria 8 waves mas só há 4 → código inválido.
// Para batch=64 e 40 CUs, 1 block/CU é suficiente (64 WGs enchem os 40 CUs).
forward_main(
    ParamsFwd        params,
    const float* __restrict__    t_f,       // [N_FAMILIES, K, K]
    const float* __restrict__    i_f,       // [N_FAMILIES, vocab_size, K]
    const float* __restrict__    nibble_h,  // [N_FAMILIES, 16, K] — high nibble (zona ASCII)
    const float* __restrict__    nibble_l,  // [N_FAMILIES, 16, K] — low  nibble (posição)
    const float* __restrict__    b_f,       // [N_FAMILIES, K]
    const float* __restrict__    pos_bias,  // [N_CELLS, K]
    const float* __restrict__    lat_w,     // [N_FAMILIES, K, K]
    const float* __restrict__    w_out,     // [FLAT_DIM, vocab_size]
    const float* __restrict__    b_out,     // [vocab_size]
    const uint32_t* __restrict__ windows,   // [batch_size, t_steps]
    float* __restrict__          logits_out,// [batch_size, vocab_size]
    float* __restrict__          states_out // [batch_size, t_steps+1, FLAT_DIM]
) {
    // ---- LDS: 16 KB + 16 KB = 32 KB ----
    __shared__ float state_curr[FLAT_DIM];
    __shared__ float lat_proj[FLAT_DIM];

    const uint32_t sample = blockIdx.x;
    if (sample >= params.batch_size) return;

    const uint32_t tid  = threadIdx.x;
    const uint32_t cell = tid / FSMS_PER_CELL;   // 0..15
    const uint32_t fsm  = tid % FSMS_PER_CELL;   // 0..7
    const uint32_t f    = fsm;
    const uint32_t v    = params.vocab_size;
    const uint32_t ts   = params.t_steps;

    const uint32_t cf_base = (cell * FSMS_PER_CELL + fsm) * K;

    // Toroidal 4-neighbors (constante: só depende de `cell`).
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

    // ---- Inicializa estado a zero ----
    for (uint32_t k = 0; k < K; k++) state_curr[cf_base + k] = 0.0f;

    if (params.cache_states == 1u) {
        const uint32_t s0 = (sample * (ts + 1u) + 0u) * FLAT_DIM;
        for (uint32_t k = 0; k < K; k++) states_out[s0 + cf_base + k] = 0.0f;
    }
    __syncthreads();

    // ---- Loop de ticks ----
    for (uint32_t step = 0; step < ts; step++) {
        const uint32_t raw        = windows[sample * ts + step];
        const uint32_t input_byte = min(raw, v - 1u);
        const uint32_t i_off      = (f * v + input_byte) * K;

        // Decomposição em nibbles para geometric embedding.
        // high_nibble = zona ASCII (0x4_=maiúsc., 0x6_=minúsc., 0x3_=dígitos, ...)
        // low_nibble  = posição dentro da zona (A=0x1, B=0x2, ..., 'a'=0x1, 'b'=0x2...)
        const uint32_t hn     = input_byte >> 4u;          // 0..15
        const uint32_t ln     = input_byte & 0x0Fu;        // 0..15
        const uint32_t nh_off = (f * 16u + hn) * K;       // offset em nibble_h
        const uint32_t nl_off = (f * 16u + ln) * K;       // offset em nibble_l

        // Fase 1: lat_proj[cf] = state_curr[cf] @ lat_w[f]
        for (uint32_t k = 0; k < K; k++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < K; j++) {
                acc += state_curr[cf_base + j] * lat_w[lat_off + j * K + k];
            }
            lat_proj[cf_base + k] = acc;
        }
        __syncthreads();

        // Fase 2: pre + leaky integrator → próximo estado (em registradores)
        // embed(byte) = i_f[byte] + nibble_h[byte>>4] + nibble_l[byte&0xF]
        // → 'A' e 'a' partilham nibble_l[0x1] → inductive bias de case grátis.
        float new_state[K];
        for (uint32_t k = 0; k < K; k++) {
            float pre = i_f[i_off + k]
                      + nibble_h[nh_off + k]   // zona ASCII (geom. high nibble)
                      + nibble_l[nl_off + k]   // posição na zona (geom. low nibble)
                      + b_f[b_off + k] + pos_bias[pos_off + k];
            // Recurrence intra-FSM
            for (uint32_t j = 0; j < K; j++) {
                pre += state_curr[cf_base + j] * t_f[t_off + j * K + k];
            }
            // Lateral: soma lat_proj dos 4 vizinhos
            pre += lat_proj[(n_up   * FSMS_PER_CELL + fsm) * K + k];
            pre += lat_proj[(n_down * FSMS_PER_CELL + fsm) * K + k];
            pre += lat_proj[(n_left * FSMS_PER_CELL + fsm) * K + k];
            pre += lat_proj[(n_rght * FSMS_PER_CELL + fsm) * K + k];

            new_state[k] = (1.0f - ALPHA) * state_curr[cf_base + k] + ALPHA * tanhf(pre);
        }
        __syncthreads();

        // Commit: cada thread escreve no seu próprio slot (sem corrida).
        for (uint32_t k = 0; k < K; k++) state_curr[cf_base + k] = new_state[k];

        if (params.cache_states == 1u) {
            const uint32_t s_off = (sample * (ts + 1u) + step + 1u) * FLAT_DIM;
            for (uint32_t k = 0; k < K; k++) states_out[s_off + cf_base + k] = new_state[k];
        }
        __syncthreads();
    }

    // ---- Decoder: logits[j] = b_out[j] + sum_i state[i] * w_out[i,j] ----
    // 128 threads cobrem `v` slots em chunks contíguos.
    const uint32_t per_thread = (v + 127u) / 128u;
    const uint32_t j_start    = tid * per_thread;
    const uint32_t j_end      = min(j_start + per_thread, v);
    for (uint32_t j = j_start; j < j_end; j++) {
        float acc = b_out[j];
        for (uint32_t i = 0; i < FLAT_DIM; i++) {
            acc += state_curr[i] * w_out[i * v + j];
        }
        logits_out[sample * v + j] = acc;
    }
}
