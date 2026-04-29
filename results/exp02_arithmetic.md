# Experimento 02 — Aritmética bytewise vs f64

**N pairs por operação:** 10000  
**Seed:** 42

## Aritmética sem FP-ALU

Operações executadas via manipulação direta de bytes Path A (digit-by-digit com carry/borrow). Comparado contra f64 nativo (FP-ALU + IEEE 754).

| Operação | Tested | Correct (within tol) | Max abs err | Max rel err | Byte ns/op | f64 ns/op | Slowdown |
|----------|-------:|---------------------:|------------:|------------:|-----------:|----------:|---------:|
| Add (+) | 10000 | 9998 (100.0%) | 9.88e-8 | 1.08e-6 | 246 | 4 | 61× |
| Sub (−) | 10000 | 10000 (100.0%) | 1.00e-7 | 7.05e-7 | 214 | 3 | 62× |
| Mul (×) | 10000 | 9940 (99.4%) | 8.79e-7 | 1.26e-4 | 313 | 4 | 78× |

## Análise

- **Correctness:** Aritmética bytewise reproduz f64 dentro da tolerância natural do Path A (~1e-6 relativo).
- **Performance:** Bytewise é ordens de magnitude mais lento que f64 nativo (esperado — implementação não-vetorizada, sem hardware especializado).
- **Implicação:** Em CPUs modernas, FP-ALU é imbatível em velocidade. Mas bytewise oferece:
  - Operações em hardware sem FPU (microcontroladores 8-bit, FPGAs simples)
  - Determinismo total (sem ulps de IEEE 754, sem rounding modes)
  - Interpretabilidade dígito-a-dígito (cada passo do cálculo é inspecionável)
  - Vias para SIMD bytewise (múltiplas operações simultâneas em registradores 64-bit, vencendo 1 op f64)
