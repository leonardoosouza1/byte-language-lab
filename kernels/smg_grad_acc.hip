// smg_grad_acc.hip — Fase 2 BPTT: acumulação de gradientes sem atomics pesados.
//
// Consome as saídas de backward_bptt (Fase 1) e acumula os gradientes
// de alta contenção sem nenhum atomic pesado, via GEMM-like batch-reduce.
//
// Entrada:
//   d_pre_out   [B, T, FLAT_DIM]    — d_pre por sample/passo (do backward_bptt)
//   d_logits_out[B, V]               — d_logits por sample   (do backward_bptt)
//   states_all  [B, T+1, FLAT_DIM]  — estados forward salvos
//
// Saída (bit-pattern IEEE-754 de f32 em uint32_t*, compatível com sgd_main):
//   g_t_f   [N_FAMILIES, K, K]  — gradiente da transição recorrente
//   g_lat_w [N_FAMILIES, K, K]  — gradiente da conexão lateral
//   g_w_out [FLAT_DIM, V]       — gradiente do decoder
//   g_b_f   [N_FAMILIES, K]     — bias FSM
//   g_pos   [N_CELLS, K]        — positional bias
//
// NOTA: os buffers g_t_f, g_lat_w, g_b_f, g_pos são zerados pelo sgd_main ao fim
// de cada step. Os kernels que usam atomic_add_f32 dependem desse zero inicial.
//
// ---- Kernel grad_acc_tf_latw ----
//
//   Grid:  (N_FAMILIES × N_CELLS, 1, 1) = (128, 1, 1)
//   Block: (128, 1, 1)
//
//   blockIdx.x = f × N_CELLS + c → cada WG cobre um par (f, c) único.
//   Loop interno: B × T apenas (c é fixo no WG — sem loop sobre N_CELLS).
//   Escrita final: atomic_add_f32 (16-way por família em 8.192 elementos → ≪1ms).
//
//   Benefício: 128 WGs × 128 threads = 16.384 threads em 40 CUs.
//   (antes: 8 WGs = 20% utilização; agora: 128 WGs ≈ 100% utilização)
//
//   Fórmula (split por célula; redução via atomic_add):
//     WG(f,c): partial_tf[m]   = Σ_{b,t} d_pre[b,t,c,f,k] × s_in[b,t,c,f,j]
//     WG(f,c): partial_latw[m] = Σ_{b,t} d_pre[b,t,c,f,k] × Σ_{c'∈nbr(c)} s_in[b,t,c',f,j]
//     g_t_f[f,j,k]   += partial_tf[m]   (atomic, 16-way)
//     g_lat_w[f,j,k] += partial_latw[m] (atomic, 16-way)
//
//   Padrão de memória (por WG, por (b,t)):
//     d_pre_out[base + cf_off + k_fixed]: 1 cache line (broadcast inter-wave, K=32 floats).
//     states_all[base + cf_off + j]:      1 cache line (32 j's via wave-broadcast × 8 m's).
//     states_all[base + nbr_* + j]:       4 × 1 cache line (vizinhos fixos por WG).
//     Total por (b,t): 6 cache lines = 768 B.
//
//   Optimização de registo: d_pre_out é carregado UMA vez por (b,t) (k constante).
//   Topologia toroidal calculada UMA vez por WG (c é fixo no WG).
//
// ---- Kernel grad_acc_wout ----
//
//   Grid:  (ceil(FLAT_DIM × V / 128), 1, 1)
//   Block: (128, 1, 1)
//
//   Cada thread (i, j) computa:
//     g_w_out[i,j] = Σ_b states_all[b, T, i] × d_logits_out[b, j]
//
//   Já bem paralelizado (8192 WGs para V=256, FLAT_DIM=4096). Sem alteração.
//
// ---- Kernels grad_acc_b_f e grad_acc_pos (128 WGs — full utilização) ----
//
//   Antes: 8 / 16 WGs → 20% / 40% utilização dos 40 CUs.
//   Depois: 128 WGs (N_FAMILIES × N_CELLS) → ~100% utilização.
//
//   grad_acc_b_f: WG(f,c) = Σ_{b,t} d_pre[b,t,c,f,k]
//     atomic_add g_b_f[f,k]: 16-way em 256 elementos → negligível.
//
//   grad_acc_pos: WG(c,f) = Σ_{b,t} d_pre[b,t,c,f,k]
//     atomic_add g_pos[c,k]: 8-way em 512 elementos → negligível.
//
// ---- Kernarg layouts (sem hidden args — não usa blockDim.x / gridDim.x) ----
//
//   grad_acc_tf_latw:
//     offset  0: ParamsGradAcc (16 B)
//     offset 16: d_pre_out     ptr (8 B)
//     offset 24: states_all    ptr (8 B)
//     offset 32: g_t_f         ptr (8 B)
//     offset 40: g_lat_w       ptr (8 B)
//     explicit: 48 B
//
//   grad_acc_wout:
//     offset  0: ParamsGradAcc (16 B)
//     offset 16: states_all    ptr (8 B)
//     offset 24: d_logits_out  ptr (8 B)
//     offset 32: g_w_out       ptr (8 B)
//     explicit: 40 B
//
//   grad_acc_b_f:
//     offset  0: ParamsGradAcc (16 B)
//     offset 16: d_pre_out     ptr (8 B)
//     offset 24: g_b_f         ptr (8 B)
//     explicit: 32 B
//
//   grad_acc_pos:
//     offset  0: ParamsGradAcc (16 B)
//     offset 16: d_pre_out     ptr (8 B)
//     offset 24: g_pos         ptr (8 B)
//     explicit: 32 B

#include <hip/hip_runtime.h>

#define K              32u
#define N_FAMILIES     8u
#define FSMS_PER_CELL  8u
#define N_CELLS        16u
#define GRID_SIDE      4u
#define FLAT_DIM       4096u

struct ParamsGradAcc {
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

// CAS-loop float atomic add.
// Usado em grad_acc_tf_latw / b_f / pos para reduzir over células.
// Os buffers de destino são zerados pelo sgd_main ao fim de cada step.
// Contention: máximo 16-way (N_CELLS=16 WGs por família) → aceitável.
__device__ __forceinline__ void atomic_add_f32(uint32_t* __restrict__ addr, float val) {
    if (val == 0.0f) return;
    uint32_t old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed,
                        f32_as_u32(u32_as_f32(assumed) + val));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// grad_acc_tf_latw — computa g_t_f e g_lat_w por batch-reduce.
//
// Grid: (N_FAMILIES × N_CELLS, 1, 1) = (128, 1, 1).
//   blockIdx.x = f × N_CELLS + c  →  f = blockIdx.x / N_CELLS, c = blockIdx.x % N_CELLS.
// Block: (128, 1, 1).
//   tid: 128 threads, 8 pares (j,k) cada (k_fixed = tid & 31 constante por thread).
//
// Cada WG acumula Σ_{b,t} para o seu (f,c) fixo e faz atomic_add no resultado.
// 128 WGs vs 8 WGs anterior → 16× mais paralelismo, todos os 40 CUs activos.
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(128, 1)
grad_acc_tf_latw(
    ParamsGradAcc             params,
    const float* __restrict__ d_pre_out,    // [B, T, FLAT_DIM]
    const float* __restrict__ states_all,   // [B, T+1, FLAT_DIM]
    uint32_t*    __restrict__ g_t_f,        // [N_FAMILIES, K, K] — atomic_add target
    uint32_t*    __restrict__ g_lat_w       // [N_FAMILIES, K, K] — atomic_add target
) {
    const uint32_t fc  = blockIdx.x;             // WG index 0..127
    const uint32_t f   = fc / N_CELLS;            // família 0..7
    const uint32_t c   = fc % N_CELLS;            // célula  0..15
    const uint32_t tid = threadIdx.x;             // 0..127
    const uint32_t B   = params.batch_size;
    const uint32_t T   = params.t_steps;

    // k é constante para todos os 8 pares deste thread (optimização de registo).
    const uint32_t k_fixed = tid & 31u;

    // Acumuladores em registadores: 8 pares (j, k_fixed) por thread.
    float acc_tf  [8] = {};
    float acc_latw[8] = {};

    // Topologia toroidal 4×4 — calculada UMA vez (c é fixo no WG).
    const uint32_t row   = c / GRID_SIDE;
    const uint32_t col   = c % GRID_SIDE;
    const uint32_t n_up  = ((row + GRID_SIDE - 1u) % GRID_SIDE) * GRID_SIDE + col;
    const uint32_t n_dn  = ((row + 1u)             % GRID_SIDE) * GRID_SIDE + col;
    const uint32_t n_lt  = row * GRID_SIDE + (col + GRID_SIDE - 1u) % GRID_SIDE;
    const uint32_t n_rt  = row * GRID_SIDE + (col + 1u)             % GRID_SIDE;

    // Offsets CF fixos dentro de FLAT_DIM (calculados UMA vez por WG).
    const uint32_t cf_off    = (c   * FSMS_PER_CELL + f) * K;
    const uint32_t nbr_up_cf = (n_up * FSMS_PER_CELL + f) * K;
    const uint32_t nbr_dn_cf = (n_dn * FSMS_PER_CELL + f) * K;
    const uint32_t nbr_lt_cf = (n_lt * FSMS_PER_CELL + f) * K;
    const uint32_t nbr_rt_cf = (n_rt * FSMS_PER_CELL + f) * K;

    for (uint32_t b = 0; b < B; b++) {
        const uint32_t b_dpre_off = b * T;
        const uint32_t b_sin_off  = b * (T + 1u);

        for (uint32_t t = 0; t < T; t++) {
            const uint32_t base_dpre = (b_dpre_off + t) * FLAT_DIM;
            const uint32_t base_sin  = (b_sin_off  + t) * FLAT_DIM;

            // d_pre[b,t,c,f,k_fixed]: 1 leitura por (b,t); coalesced (k=0..31 por wave).
            // Broadcast inter-wave: todos os 4 waves lêem a mesma cache line.
            const float dpk = d_pre_out[base_dpre + cf_off + k_fixed];

            // Offsets dos vizinhos (fixos por WG; apenas base_sin varia com b,t).
            const uint32_t sin_up = base_sin + nbr_up_cf;
            const uint32_t sin_dn = base_sin + nbr_dn_cf;
            const uint32_t sin_lt = base_sin + nbr_lt_cf;
            const uint32_t sin_rt = base_sin + nbr_rt_cf;
            const uint32_t sin_c  = base_sin + cf_off;

            // 8 pares (j, k_fixed): j varia por m, k_fixed constante.
            // Wave 0: j=0,4,8,...,28 (m=0..7); wave 1: j=1,5,...,29; etc.
            // Reads de states_all[sin_c + j]: broadcast intra-wave (1 cache line).
            for (uint32_t m = 0; m < 8u; m++) {
                const uint32_t pair = tid + m * 128u;
                const uint32_t j    = pair >> 5u;   // pair / K

                acc_tf[m]   += dpk * states_all[sin_c + j];
                acc_latw[m] += dpk * (states_all[sin_up + j]
                                    + states_all[sin_dn + j]
                                    + states_all[sin_lt + j]
                                    + states_all[sin_rt + j]);
            }
        }
    }

    // Escrita via atomic_add: 16 WGs (um por célula) contribuem para cada (f,j,k).
    // Contention: 16-way em K×K×N_FAMILIES = 8.192 elementos → ≈4µs de overhead.
    // g_t_f / g_lat_w são zerados pelo sgd_main antes deste kernel.
    const uint32_t f_off = f * K * K;
    for (uint32_t m = 0; m < 8u; m++) {
        const uint32_t pair = tid + m * 128u;
        const uint32_t j    = pair >> 5u;
        const uint32_t idx  = f_off + j * K + k_fixed;
        atomic_add_f32(&g_t_f  [idx], acc_tf  [m]);
        atomic_add_f32(&g_lat_w[idx], acc_latw[m]);
    }
}

// ---------------------------------------------------------------------------
// grad_acc_wout — computa g_w_out[FLAT_DIM, V] = states_final^T × d_logits.
//
// Grid:  (ceil(FLAT_DIM × V / 128), 1, 1)
// Block: (128, 1, 1)
//
// Thread gid = blockIdx.x × 128 + threadIdx.x.
//   i = gid / V  ∈ [0, FLAT_DIM)
//   j = gid % V  ∈ [0, V)
//   g_w_out[i,j] = Σ_{b=0}^{B-1} states_all[b, T, i] × d_logits_out[b, j]
//
// Já bem paralelizado (8192 WGs para V=256, FLAT_DIM=4096). Sem alteração.
// Acesso (V=256, B=64):
//   states_all[(b*(T+1)+T)*FLAT_DIM + i] → broadcast ✓
//   d_logits_out[b*256 + 0..127]          → coalesced ✓
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(128, 1)
grad_acc_wout(
    ParamsGradAcc             params,
    const float* __restrict__ states_all,    // [B, T+1, FLAT_DIM]
    const float* __restrict__ d_logits_out,  // [B, V]
    uint32_t*    __restrict__ g_w_out        // [FLAT_DIM, V] bit-pattern f32
) {
    const uint32_t B = params.batch_size;
    const uint32_t V = params.vocab_size;
    const uint32_t T = params.t_steps;

    const uint32_t gid = blockIdx.x * 128u + threadIdx.x;
    if (gid >= FLAT_DIM * V) return;

    const uint32_t i = gid / V;
    const uint32_t j = gid % V;

    // Offset do estado final: states_all[b, T, i] = [(b*(T+1)+T)*FLAT_DIM + i].
    // i é constante por thread; stride entre amostras = (T+1)*FLAT_DIM.
    const uint32_t st_stride = (T + 1u) * FLAT_DIM;
    const uint32_t st_base   = T * FLAT_DIM + i;   // offset para b=0

    float acc = 0.0f;
    for (uint32_t b = 0; b < B; b++) {
        const float s = states_all[b * st_stride + st_base];
        const float d = d_logits_out[b * V + j];
        acc += s * d;
    }
    g_w_out[i * V + j] = f32_as_u32(acc);
}

// ---------------------------------------------------------------------------
// grad_acc_b_f — g_b_f[N_FAMILIES, K] via redução por (f, c) sem atomics pesados.
//
// Grid:  (N_FAMILIES × N_CELLS, 1, 1) = (128, 1, 1)
//   blockIdx.x = f × N_CELLS + c  →  f = blockIdx.x / N_CELLS, c = blockIdx.x % N_CELLS.
// Block: (K, 1, 1) = (32, 1, 1) — 1 wave/WG.
//
// WG(f,c): partial = Σ_{b,t} d_pre[b,t,(c×8+f)×K+k]
// atomic_add g_b_f[f,k] += partial   (16-way em 256 elementos → negligível).
//
// 128 WGs vs 8 WGs anterior:
//   Antes: 8 WGs × B×T×N_CELLS = 49.152 iters/WG; 20% utilização.
//   Depois: 128 WGs × B×T = 3.072 iters/WG; 100% utilização.
//
// Acesso a d_pre_out[base + k] com k=threadIdx.x:
//   32 threads acessam 32 floats consecutivos → coalesced (1 cache line).
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(32, 1)
grad_acc_b_f(
    ParamsGradAcc             params,
    const float* __restrict__ d_pre_out,  // [B, T, FLAT_DIM]
    uint32_t*    __restrict__ g_b_f       // [N_FAMILIES, K] — atomic_add target
) {
    const uint32_t fc = blockIdx.x;              // WG index 0..127
    const uint32_t f  = fc / N_CELLS;             // família 0..7
    const uint32_t c  = fc % N_CELLS;             // célula  0..15
    const uint32_t k  = threadIdx.x;              // k 0..31
    const uint32_t B  = params.batch_size;
    const uint32_t T  = params.t_steps;

    const uint32_t cf_off = (c * FSMS_PER_CELL + f) * K;

    float acc = 0.0f;
    for (uint32_t b = 0; b < B; b++) {
        const uint32_t b_off = b * T;
        for (uint32_t t = 0; t < T; t++) {
            // d_pre_out[b, t, (c*8+f)*K + k] — coalesced por k dentro do wave.
            acc += d_pre_out[(b_off + t) * FLAT_DIM + cf_off + k];
        }
    }
    // 16 WGs por família (N_CELLS=16) → 16-way CAS em N_FAMILIES×K = 256 elementos.
    // g_b_f zerado pelo sgd_main antes deste kernel.
    atomic_add_f32(&g_b_f[f * K + k], acc);
}

// ---------------------------------------------------------------------------
// grad_acc_pos — g_pos[N_CELLS, K] via redução por (c, f) sem atomics pesados.
//
// Grid:  (N_CELLS × N_FAMILIES, 1, 1) = (128, 1, 1)
//   blockIdx.x = c × N_FAMILIES + f  →  c = blockIdx.x / N_FAMILIES, f = blockIdx.x % N_FAMILIES.
// Block: (K, 1, 1) = (32, 1, 1) — 1 wave/WG.
//
// WG(c,f): partial = Σ_{b,t} d_pre[b,t,(c×8+f)×K+k]
// atomic_add g_pos[c,k] += partial   (8-way em 512 elementos → negligível).
//
// 128 WGs vs 16 WGs anterior:
//   Antes: 16 WGs × B×T×N_FAMILIES = 24.576 iters/WG; 40% utilização.
//   Depois: 128 WGs × B×T = 3.072 iters/WG; 100% utilização.
//
// Acesso a d_pre_out[base + k] com k=threadIdx.x:
//   32 threads acessam 32 floats consecutivos → coalesced (1 cache line).
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(32, 1)
grad_acc_pos(
    ParamsGradAcc             params,
    const float* __restrict__ d_pre_out,  // [B, T, FLAT_DIM]
    uint32_t*    __restrict__ g_pos       // [N_CELLS, K] — atomic_add target
) {
    const uint32_t cf = blockIdx.x;              // WG index 0..127
    const uint32_t c  = cf / N_FAMILIES;          // célula  0..15
    const uint32_t f  = cf % N_FAMILIES;          // família 0..7
    const uint32_t k  = threadIdx.x;              // k 0..31
    const uint32_t B  = params.batch_size;
    const uint32_t T  = params.t_steps;

    const uint32_t cf_off = (c * FSMS_PER_CELL + f) * K;

    float acc = 0.0f;
    for (uint32_t b = 0; b < B; b++) {
        const uint32_t b_off = b * T;
        for (uint32_t t = 0; t < T; t++) {
            // d_pre_out[b, t, (c*8+f)*K + k] — coalesced por k dentro do wave.
            acc += d_pre_out[(b_off + t) * FLAT_DIM + cf_off + k];
        }
    }
    // 8 WGs por célula (N_FAMILIES=8) → 8-way CAS em N_CELLS×K = 512 elementos.
    // g_pos zerado pelo sgd_main antes deste kernel.
    atomic_add_f32(&g_pos[c * K + k], acc);
}
