// Backward kernel monolítico para SMG v3.
//
// 1 workgroup = 1 sample. 128 threads (cell, fsm). Cada workgroup:
//   1) Recomputa softmax dos logits do sample → probs.
//      d_logits[j] = probs[j] - one_hot(target)[j]
//   2) Acumula g_b_out, g_w_out (atomic-add f32 via CAS no global).
//   3) Computa d_flat[i] = sum_j w_out[i,j] * d_logits[j]   (em LDS).
//   4) Itera ticks (T..0) executando 3 fases do backward CPU:
//        Fase 1: d_pre, params I/b/pos, T_f, recurrence-bw  → atomic-add
//        Fase 2/3 (combinadas): pool d_pre dos 4 vizinhos inline,
//                               outer-product g_lat_w, lat_w-bw   → atomic-add
//      d_state_next ← d_state_prev (mesmo buffer LDS, escritas só no slot próprio).
//
// Trocas de design:
//   * Atomic CAS f32: contenção real só em g_w_out (4096*V slots) e g_b_out (V).
//     Outros grads são por-família/posição, baixa contenção em batch ≤ 64.
//   * d_state em LDS (16KB), d_pre_all em LDS (16KB) → 32KB → 2 wgroups/CU.
//   * Não materializa d_pool: phase 3 lê d_pre_all dos vizinhos diretamente.

const K: u32 = 32u;
const N_FAMILIES: u32 = 8u;
const FSMS_PER_CELL: u32 = 8u;
const N_CELLS: u32 = 16u;
const GRID_SIDE: u32 = 4u;
const FLAT_DIM: u32 = 4096u;
const ALPHA: f32 = 0.1;

struct ParamsBwd {
    vocab_size: u32,
    t_steps: u32,
    batch_size: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: ParamsBwd;

// Pesos read-only (necessários para backprop).
@group(0) @binding(1) var<storage, read> t_f: array<f32>;
@group(0) @binding(2) var<storage, read> lat_w: array<f32>;
@group(0) @binding(3) var<storage, read> w_out: array<f32>;

// Forward outputs / inputs (read-only).
@group(0) @binding(4) var<storage, read> states_all: array<f32>;   // [B, T+1, FLAT_DIM]
@group(0) @binding(5) var<storage, read> logits_in: array<f32>;    // [B, V]
@group(0) @binding(6) var<storage, read> windows: array<u32>;      // [B, T]
@group(0) @binding(7) var<storage, read> targets: array<u32>;      // [B]

// Gradientes acumulados (atomic<u32> com bitcast f32).
@group(0) @binding(8)  var<storage, read_write> g_t_f:    array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> g_i_f:    array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> g_b_f:    array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> g_pos:    array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> g_lat_w:  array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> g_w_out:  array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> g_b_out:  array<atomic<u32>>;

// Atomic float add via CAS-loop. WGSL não tem float atomics nativos, e a spec
// não permite passar ptr<storage> como parâmetro de função → 1 helper por buffer.
fn add_t_f(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_t_f[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_t_f[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}
fn add_i_f(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_i_f[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_i_f[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}
fn add_b_f(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_b_f[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_b_f[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}
fn add_pos(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_pos[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_pos[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}
fn add_lat_w(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_lat_w[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_lat_w[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}
fn add_w_out(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_w_out[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_w_out[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}
fn add_b_out(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_u = atomicLoad(&g_b_out[idx]);
    loop {
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let res = atomicCompareExchangeWeak(&g_b_out[idx], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}

// LDS workgroup-scope. Total ~36KB:
//   d_state    16KB  (d_state_next, mutado in-place para d_state_prev)
//   d_pre_all  16KB  (pré-ativação backward para todas as cells×fsms)
//   d_logits    4KB  (sustenta V até 1024)
var<workgroup> d_state: array<f32, 4096>;
var<workgroup> d_pre_all: array<f32, 4096>;
var<workgroup> d_logits: array<f32, 1024>;

@compute @workgroup_size(128, 1, 1)
fn backward_main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let sample = wg_id.x;
    if (sample >= params.batch_size) { return; }
    let tid = lid.x;
    let cell = tid / FSMS_PER_CELL;
    let fsm = tid % FSMS_PER_CELL;
    let f = fsm;
    let v = params.vocab_size;
    let t_steps = params.t_steps;
    let cf_base = (cell * FSMS_PER_CELL + fsm) * K;

    let row = cell / GRID_SIDE;
    let col = cell % GRID_SIDE;
    let n_up    = ((row + GRID_SIDE - 1u) % GRID_SIDE) * GRID_SIDE + col;
    let n_down  = ((row + 1u) % GRID_SIDE) * GRID_SIDE + col;
    let n_left  = row * GRID_SIDE + (col + GRID_SIDE - 1u) % GRID_SIDE;
    let n_right = row * GRID_SIDE + (col + 1u) % GRID_SIDE;

    let t_off = f * K * K;
    let b_off = f * K;
    let pos_off = cell * K;
    let lat_off = f * K * K;

    // ===== HEAD: softmax + d_logits + g_b_out + g_w_out + d_flat =====
    //
    // d_logits é grande (V); precisa estar visível para todas as threads.
    // Thread 0 calcula d_logits[0..V] inteiro em LDS (sequencial, 1 vez por sample).
    // V típico = 16..256, O(V) é trivial vs BPTT.
    if (tid == 0u) {
        let logits_off = sample * v;
        var mx: f32 = -1e30;
        for (var j = 0u; j < v; j = j + 1u) {
            let lv = logits_in[logits_off + j];
            if (lv > mx) { mx = lv; }
        }
        var s: f32 = 0.0;
        for (var j = 0u; j < v; j = j + 1u) {
            let e = exp(logits_in[logits_off + j] - mx);
            d_logits[j] = e;
            s = s + e;
        }
        let tgt = min(targets[sample], v - 1u);
        for (var j = 0u; j < v; j = j + 1u) {
            var p = d_logits[j] / s;
            if (j == tgt) { p = p - 1.0; }
            d_logits[j] = p;
        }
    }
    workgroupBarrier();

    // g_b_out[j] += d_logits[j]   — todos threads cobrem chunks de V.
    let per_thread_v = (v + 127u) / 128u;
    let j_start = tid * per_thread_v;
    let j_end = min(j_start + per_thread_v, v);
    for (var j = j_start; j < j_end; j = j + 1u) {
        add_b_out(j, d_logits[j]);
    }

    // g_w_out[i, j] += states_all[sample, T, i] * d_logits[j]
    // d_state[i] (semente do BPTT) = sum_j w_out[i,j] * d_logits[j]
    // Cada thread cobre i = cf_base..cf_base+K (seu próprio (cell,fsm)).
    let s_final_off = (sample * (t_steps + 1u) + t_steps) * FLAT_DIM;
    for (var k = 0u; k < K; k = k + 1u) {
        let i = cf_base + k;
        let s_i = states_all[s_final_off + i];
        var d_flat_i: f32 = 0.0;
        let w_row = i * v;
        for (var j = 0u; j < v; j = j + 1u) {
            let dl = d_logits[j];
            d_flat_i = d_flat_i + w_out[w_row + j] * dl;
            add_w_out(w_row + j, s_i * dl);
        }
        d_state[i] = d_flat_i;
    }
    workgroupBarrier();

    // ===== BPTT: walk ticks reverse =====
    for (var step_inv = 0u; step_inv < t_steps; step_inv = step_inv + 1u) {
        let step = t_steps - 1u - step_inv;
        let raw = windows[sample * t_steps + step];
        let input_byte = min(raw, v - 1u);
        let i_off = (f * v + input_byte) * K;
        let s_in_off  = (sample * (t_steps + 1u) + step) * FLAT_DIM;
        let s_out_off = (sample * (t_steps + 1u) + step + 1u) * FLAT_DIM;

        // Estado de entrada e saída deste tick para a (cell,fsm) própria, em registradores.
        var s_in_local: array<f32, 32>;
        var s_out_local: array<f32, 32>;
        var dsn_local: array<f32, 32>;
        for (var k = 0u; k < K; k = k + 1u) {
            s_in_local[k] = states_all[s_in_off + cf_base + k];
            s_out_local[k] = states_all[s_out_off + cf_base + k];
            dsn_local[k] = d_state[cf_base + k];
        }

        // ---- Fase 1: d_pre, params, recurrence-bw ----
        // d_pre[k] = α * (1 - tanh(pre)²) * dsn[k]
        // tanh(pre) recuperado de cache: tp = (s_out - (1-α)*s_in) / α
        var d_pre_local: array<f32, 32>;
        var d_state_prev_local: array<f32, 32>;  // contribuição do recurrence intra-FSM
        for (var k = 0u; k < K; k = k + 1u) {
            // residual: d_state_prev += (1-α) * dsn
            d_state_prev_local[k] = (1.0 - ALPHA) * dsn_local[k];
            let tp = (s_out_local[k] - (1.0 - ALPHA) * s_in_local[k]) / ALPHA;
            d_pre_local[k] = ALPHA * (1.0 - tp * tp) * dsn_local[k];
        }

        // Atomic adds para params I, b, pos.
        for (var k = 0u; k < K; k = k + 1u) {
            add_i_f(i_off + k, d_pre_local[k]);
            add_b_f(b_off + k, d_pre_local[k]);
            add_pos(pos_off + k, d_pre_local[k]);
        }
        // Publicar d_pre_local em LDS para fase 2/3.
        for (var k = 0u; k < K; k = k + 1u) {
            d_pre_all[cf_base + k] = d_pre_local[k];
        }

        // T_f outer-product (g_t_f) e recurrence-bw (d_state_prev += T_f^T @ d_pre).
        for (var j = 0u; j < K; j = j + 1u) {
            let sj = s_in_local[j];
            let row_off = t_off + j * K;
            var acc: f32 = 0.0;
            for (var k = 0u; k < K; k = k + 1u) {
                let dp = d_pre_local[k];
                add_t_f(row_off + k, sj * dp);
                acc = acc + t_f[row_off + k] * dp;
            }
            d_state_prev_local[j] = d_state_prev_local[j] + acc;
        }
        workgroupBarrier();  // d_pre_all visível p/ fase 2/3

        // ---- Fase 2/3 combinadas: pool dos 4 vizinhos + g_lat_w + lat_w-bw ----
        // pool[k] = soma de d_pre_all[neighbor, fsm, k] sobre os 4 vizinhos da CELL ATUAL.
        // (Por simetria 4-conn toroidal, vizinho-de-vizinho retorna a si mesmo.)
        var pool_local: array<f32, 32>;
        for (var k = 0u; k < K; k = k + 1u) {
            pool_local[k] =
                  d_pre_all[(n_up    * FSMS_PER_CELL + fsm) * K + k]
                + d_pre_all[(n_down  * FSMS_PER_CELL + fsm) * K + k]
                + d_pre_all[(n_left  * FSMS_PER_CELL + fsm) * K + k]
                + d_pre_all[(n_right * FSMS_PER_CELL + fsm) * K + k];
        }
        for (var j = 0u; j < K; j = j + 1u) {
            let sj = s_in_local[j];
            let row_off = lat_off + j * K;
            var acc: f32 = 0.0;
            for (var k = 0u; k < K; k = k + 1u) {
                let pk = pool_local[k];
                add_lat_w(row_off + k, sj * pk);
                acc = acc + lat_w[row_off + k] * pk;
            }
            d_state_prev_local[j] = d_state_prev_local[j] + acc;
        }
        workgroupBarrier();

        // d_state_prev_local → d_state[cf_base..] (cada thread escreve seu próprio slot).
        for (var k = 0u; k < K; k = k + 1u) {
            d_state[cf_base + k] = d_state_prev_local[k];
        }
        workgroupBarrier();
    }
}
