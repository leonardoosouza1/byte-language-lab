// Forward kernel for SMG v3 (microcolumn_v3.rs).
//
// Mapeamento RDNA2 (wave32, 40 CUs, 64KB LDS/CU):
//   * 1 workgroup = 1 sample. workgroup_size = 128 threads (= N_CELLS * FSMS_PER_CELL).
//     Em wave32 isso = 4 waves/workgroup. Cada thread possui (cell, fsm) único.
//   * Cada thread processa K=32 dims sequencialmente (loop curto, predicação OK).
//   * Workgroup memory (LDS): state_curr (16 KB) + lat_proj (16 KB) = 32 KB.
//     Fits 2 workgroups/CU → 80 workgroups in-flight nas 40 CUs.
//   * Estado novo (próximo tick) fica em registradores per-thread (array<f32, K>).
//
// Layouts (todos row-major):
//   t_f       : [N_FAMILIES, K, K]              (recurrence intra-FSM)
//   i_f       : [N_FAMILIES, vocab_size, K]     (input embedding)
//   b_f       : [N_FAMILIES, K]
//   pos_bias  : [N_CELLS, K]
//   lat_w     : [N_FAMILIES, K, K]              (lateral bus, shared 4 dirs)
//   w_out     : [FLAT_DIM, vocab_size]
//   b_out     : [vocab_size]
//   windows   : [batch_size, t_steps]           (u32 byte ids)
//   logits_out: [batch_size, vocab_size]

const K: u32 = 32u;
const N_FAMILIES: u32 = 8u;
const FSMS_PER_CELL: u32 = 8u;
const N_CELLS: u32 = 16u;
const GRID_SIDE: u32 = 4u;
const FLAT_DIM: u32 = 4096u;       // N_CELLS * FSMS_PER_CELL * K
const ALPHA: f32 = 0.1;

struct Params {
    vocab_size: u32,
    t_steps: u32,
    batch_size: u32,
    _pad: u32,
};

struct ParamsFwd {
    vocab_size: u32,
    t_steps: u32,
    batch_size: u32,
    cache_states: u32,   // 0 = inference (skip states_out), 1 = training (write states_out)
};

@group(0) @binding(0) var<uniform> params: ParamsFwd;
@group(0) @binding(1) var<storage, read> t_f: array<f32>;
@group(0) @binding(2) var<storage, read> i_f: array<f32>;
@group(0) @binding(3) var<storage, read> b_f: array<f32>;
@group(0) @binding(4) var<storage, read> pos_bias: array<f32>;
@group(0) @binding(5) var<storage, read> lat_w: array<f32>;
@group(0) @binding(6) var<storage, read> w_out: array<f32>;
@group(0) @binding(7) var<storage, read> b_out: array<f32>;
@group(0) @binding(8) var<storage, read> windows: array<u32>;
@group(0) @binding(9) var<storage, read_write> logits_out: array<f32>;
// Cache de estados para BPTT: [B, T+1, FLAT_DIM]. Só escrito quando params.cache_states == 1.
@group(0) @binding(10) var<storage, read_write> states_out: array<f32>;

// LDS: 16KB + 16KB = 32KB
var<workgroup> state_curr: array<f32, 4096>;
var<workgroup> lat_proj: array<f32, 4096>;

@compute @workgroup_size(128, 1, 1)
fn forward_main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let sample = wg_id.x;
    if (sample >= params.batch_size) {
        return;
    }
    let tid = lid.x;
    let cell = tid / FSMS_PER_CELL;   // 0..16
    let fsm = tid % FSMS_PER_CELL;    // 0..8
    let f = fsm;
    let v = params.vocab_size;
    let t_steps = params.t_steps;

    let cf_base = (cell * FSMS_PER_CELL + fsm) * K;

    // Toroidal 4-neighbors (constante: depende apenas de `cell`).
    let row = cell / GRID_SIDE;
    let col = cell % GRID_SIDE;
    let n_up = ((row + GRID_SIDE - 1u) % GRID_SIDE) * GRID_SIDE + col;
    let n_down = ((row + 1u) % GRID_SIDE) * GRID_SIDE + col;
    let n_left = row * GRID_SIDE + (col + GRID_SIDE - 1u) % GRID_SIDE;
    let n_right = row * GRID_SIDE + (col + 1u) % GRID_SIDE;

    let t_off = f * K * K;
    let b_off = f * K;
    let pos_off = cell * K;
    let lat_off = f * K * K;

    // Inicializa estado a zero.
    for (var k = 0u; k < K; k = k + 1u) {
        state_curr[cf_base + k] = 0.0;
    }
    // Cache: states_out[sample, 0, cf_base..] = 0
    if (params.cache_states == 1u) {
        let s0 = (sample * (t_steps + 1u) + 0u) * FLAT_DIM;
        for (var k = 0u; k < K; k = k + 1u) {
            states_out[s0 + cf_base + k] = 0.0;
        }
    }
    workgroupBarrier();

    // ---------------- Loop de ticks ----------------
    for (var step = 0u; step < t_steps; step = step + 1u) {
        let raw = windows[sample * t_steps + step];
        let input_byte = min(raw, v - 1u);
        let i_off = (f * v + input_byte) * K;

        // ---- Fase 1: lat_proj[c, f, k] = sum_j state_curr[c,f,j] * lat_w[f,j,k] ----
        for (var k = 0u; k < K; k = k + 1u) {
            var acc: f32 = 0.0;
            for (var j = 0u; j < K; j = j + 1u) {
                acc = acc + state_curr[cf_base + j] * lat_w[lat_off + j * K + k];
            }
            lat_proj[cf_base + k] = acc;
        }
        workgroupBarrier();

        // ---- Fase 2: pre + leaky integrator → próximo estado em registradores ----
        var new_state: array<f32, 32>;  // K=32; per-thread.
        for (var k = 0u; k < K; k = k + 1u) {
            var pre: f32 = i_f[i_off + k] + b_f[b_off + k] + pos_bias[pos_off + k];
            // Recurrence intra-FSM
            for (var j = 0u; j < K; j = j + 1u) {
                pre = pre + state_curr[cf_base + j] * t_f[t_off + j * K + k];
            }
            // Lateral: soma lat_proj dos 4 vizinhos
            pre = pre + lat_proj[(n_up * FSMS_PER_CELL + fsm) * K + k];
            pre = pre + lat_proj[(n_down * FSMS_PER_CELL + fsm) * K + k];
            pre = pre + lat_proj[(n_left * FSMS_PER_CELL + fsm) * K + k];
            pre = pre + lat_proj[(n_right * FSMS_PER_CELL + fsm) * K + k];

            new_state[k] = (1.0 - ALPHA) * state_curr[cf_base + k] + ALPHA * tanh(pre);
        }
        workgroupBarrier();

        // ---- Commit: cada thread escreve no seu próprio slot (sem corrida) ----
        for (var k = 0u; k < K; k = k + 1u) {
            state_curr[cf_base + k] = new_state[k];
        }
        // Cache: states_out[sample, step+1, cf_base..]
        if (params.cache_states == 1u) {
            let s_off = (sample * (t_steps + 1u) + step + 1u) * FLAT_DIM;
            for (var k = 0u; k < K; k = k + 1u) {
                states_out[s_off + cf_base + k] = new_state[k];
            }
        }
        workgroupBarrier();
    }

    // ---------------- Decoder: logits[j] = b_out[j] + sum_i state[i] * w_out[i,j] ----------------
    // 128 threads cobrem `vocab_size` slots de saída em chunks contíguos.
    // Cada thread varre todo state_curr (LDS) → custo: FLAT_DIM * (vocab/128) ops.
    let per_thread = (v + 127u) / 128u;
    let j_start = tid * per_thread;
    let j_end = min(j_start + per_thread, v);
    for (var j = j_start; j < j_end; j = j + 1u) {
        var acc: f32 = b_out[j];
        for (var i = 0u; i < FLAT_DIM; i = i + 1u) {
            acc = acc + state_curr[i] * w_out[i * v + j];
        }
        logits_out[sample * v + j] = acc;
    }
}
