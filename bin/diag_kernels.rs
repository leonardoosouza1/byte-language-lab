//! diag_kernels — diagnóstico de carga de .hsaco (identifica qual módulo falha).
//!
//! Carrega cada kernel individualmente via HsaContext::load_module para
//! determinar qual .hsaco produz HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS.
//!
//! ```bash
//! cargo run --release --features gpu-aql --bin diag_kernels
//! ```
#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

// Acede ao HsaContext directamente (sem HsaDispatcher) para carga granular.
use hip_ffi::hsa_dispatch::{HsaDispatcher, HsaError};

// .hsaco compilados em build-time (embedded no binário).
static FORWARD_HSACO:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_forward.hsaco"));
static BACKWARD_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_backward.hsaco"));
static BPTT_HSACO:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_backward_bptt.hsaco"));
static GRAD_ACC_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_grad_acc.hsaco"));
static NORM_HSACO:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_norm.hsaco"));
static SGD_HSACO:      &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_sgd.hsaco"));

fn try_load_all() -> Result<(), HsaError> {
    let modules = [
        ("smg_forward",       FORWARD_HSACO,  FORWARD_HSACO.len()),
        ("smg_backward",      BACKWARD_HSACO, BACKWARD_HSACO.len()),
        ("smg_backward_bptt", BPTT_HSACO,     BPTT_HSACO.len()),
        ("smg_grad_acc",      GRAD_ACC_HSACO, GRAD_ACC_HSACO.len()),
        ("smg_norm",          NORM_HSACO,     NORM_HSACO.len()),
        ("smg_sgd",           SGD_HSACO,      SGD_HSACO.len()),
    ];

    // Cria dispatcher com apenas UM módulo de cada vez para isolar o erro.
    for (name, bytes, sz) in &modules {
        eprint!("  Carregando {}.hsaco ({} bytes)... ", name, sz);
        match HsaDispatcher::new(&[bytes], 512, 4) {
            Ok(_disp) => eprintln!("OK ✓"),
            Err(e)    => {
                eprintln!("FALHOU ✗");
                eprintln!("    Erro: {:?}", e);
                return Err(e);
            }
        }
    }

    eprintln!("\n  Carregando TODOS em simultâneo...");
    let all: Vec<&[u8]> = modules.iter().map(|(_, b, _)| *b).collect();
    match HsaDispatcher::new(&all, 512, 128) {
        Ok(_) => eprintln!("  Todos OK ✓"),
        Err(e) => {
            eprintln!("  FALHOU na carga combinada: {:?}", e);
            return Err(e);
        }
    }

    Ok(())
}

fn main() {
    println!("=== diag_kernels: diagnóstico de carga HSA ===\n");

    match try_load_all() {
        Ok(()) => println!("\nTodos os módulos carregados com sucesso."),
        Err(e)  => {
            eprintln!("\nDiagnóstico concluído: erro em {:?}", e);
            std::process::exit(1);
        }
    }
}
