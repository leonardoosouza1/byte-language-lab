// build.rs — byte-language-lab
//
// Quando feature `gpu-aql` está ativa, compila os kernels HIP (`.hip`) em
// `.hsaco` (HIP Shader Archive Code Object) usando hipcc do ROCm.
//
// Ver `docs/migration-hsa-aql.md` para o racional arquitetural.

fn main() {
    // Só compila kernels se a feature gpu-aql está ativa.
    if std::env::var("CARGO_FEATURE_GPU_AQL").is_err() {
        return;
    }

    compile_smg_kernels();
}

fn compile_smg_kernels() {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm-7.2.1".to_string());
    let hipcc = format!("{}/bin/hipcc", rocm);
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Flags de compilação RDNA2.
    // NOTA: `-mwavefrontsize32` e `-mno-cumode` são flags de device-side válidos
    // mas hipcc no ROCm 7.x os repassa ao host clang++ quando usados com
    // `--genco`, causando erro. Para gfx1031 eles são o DEFAULT (RDNA2 sempre
    // compila em wave32 e WGP mode), então são dispensáveis.
    // Ver docs/research/rdna2-microarch.md §wavefronts.
    let flags = [
        "--genco",
        "--offload-arch=gfx1031",
        "-O3",
        "-ffast-math",
        "-std=c++17",
    ];

    // Kernels em ordem de dependência (forward antes de backward pois backward
    // é testado contra forward — mas para compilação a ordem não importa).
    //
    // smg_backward_bptt: Fase 1 do backward refactored (sem atomics para g_t_f/g_lat_w/g_w_out).
    // smg_grad_acc:      Fase 2 do backward refactored (GEMM-like batch-reduce sem atomics).
    let kernels = [
        "smg_forward",
        "smg_backward",
        "smg_backward_bptt",
        "smg_grad_acc",
        "smg_norm",
        "smg_sgd",
    ];

    for name in kernels {
        let src = format!("{}/kernels/{}.hip", manifest, name);
        let out = format!("{}/{}.hsaco", out_dir, name);

        println!("cargo:rerun-if-changed=kernels/{}.hip", name);

        let status = std::process::Command::new(&hipcc)
            .args(&flags)
            .args(["-o", &out, &src])
            .status()
            .unwrap_or_else(|e| panic!("hipcc not found at {}: {}", hipcc, e));

        if !status.success() {
            panic!(
                "hipcc compilation failed for kernels/{}.hip\n\
                 Hint: check ROCM_PATH (current: {}) and kernel syntax.",
                name, rocm
            );
        }

        println!("cargo:warning=Compiled kernels/{}.hip → {}", name, out);
    }
}
