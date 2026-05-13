# DESIGN — Formato de Byte Estruturado

## Princípio fundamental

```
1 byte = 8 bits = [valor: 4 bits][metadata: 4 bits]
```

Onde:
- **valor** é a unidade semântica primária (dígito, símbolo, classe)
- **metadata** é informação contextual (posição, sinal, flags, tipo)

Diferentes formatos codificam metadata de formas diferentes. Comparamos 3 paths.

---

## Path A — Posicional puro (BCD anotado)

Mais simples e direto. Cada byte = um dígito + sua posição relativa ao ponto decimal.

```
bits 7-4: digit
  0..=9    → dígito decimal
  10 (0xA) → ponto decimal "."
  11 (0xB) → sinal negativo "-"
  12 (0xC) → marcador exponencial "e"
  13 (0xD) → fim de número (delimitador)
  14 (0xE) → reservado
  15 (0xF) → padding/null

bits 3-0: position
  0..=14 → posição do dígito (0=menos significativo, cresce p/ esquerda)
  15     → posição genérica (não relevante, ex: para "." e delimitadores)

Exemplo: 47.83
  byte 0: [0100][0001]  = digit=4, position=1
  byte 1: [0111][0000]  = digit=7, position=0
  byte 2: [1010][1111]  = digit=., position=any
  byte 3: [1000][1110]  = digit=8, position=-1 (encoded as 14)
  byte 4: [0011][1101]  = digit=3, position=-2 (encoded as 13)
  byte 5: [1101][1111]  = end-of-number delimiter
```

**Vantagens:** simplicidade, posição explícita, fácil de mutar.
**Desvantagens:** range limitado (-14 .. +14 posições = ~10^14 a 10^-14), padding necessário para números curtos.

---

## Path B — Símbolos ricos com flags

Mais expressivo. Aproveita os 4 bits de metadata como flags categóricos.

```
bits 7-4: digit
  0..=9    → dígito decimal
  10 (0xA) → ponto "."
  11 (0xB) → "e" (notação científica)
  12 (0xC) → "/" (fração)
  13 (0xD) → "π" (constante simbólica)
  14 (0xE) → "∞" (infinito)
  15 (0xF) → delimitador/end

bits 3-0: flags (cada bit independente)
  bit 3: signed (1=negativo)
  bit 2: exponent_part (1=está no expoente)
  bit 1: fractional_part (1=está após o ponto)
  bit 0: high_significance (1=primeiro dígito significativo)

Exemplo: -0.483e-2
  byte 0: [1011][1000]  = '-' (digit=B, signed flag)
  byte 1: [0000][0001]  = 0 (high_sig)
  byte 2: [1010][0010]  = '.' (fractional)
  byte 3: [0100][0011]  = 4 (fractional + high_sig)
  byte 4: [1000][0010]  = 8 (fractional)
  byte 5: [0011][0010]  = 3 (fractional)
  byte 6: [1011][0100]  = 'e' (exponent_part)
  byte 7: [1011][1100]  = '-' (signed + exponent)
  byte 8: [0010][0101]  = 2 (exponent + high_sig)
  byte 9: [1111][0000]  = end
```

**Vantagens:** expressividade rica, flags compostáveis, lida com notação científica e símbolos especiais nativamente.
**Desvantagens:** menos compacto, decoder mais complexo, posição precisa ser inferida da ordem.

---

## Path C — Domain-tagged (header + payload)

Cada bloco semântico começa com um byte HEADER que declara seu tipo. Permite mistura de domínios em um único stream.

```
HEADER byte:
  bits 7-4: domain
    0 → number (decimal)
    1 → number (hex)
    2 → text (ASCII subset)
    3 → command (action class)
    4 → state (FSM byte semantics: família+nível+escala+spike)
    5 → boolean
    6 → percent (0-100, 1 byte)
    7..15 → reservado

  bits 3-0: length (em bytes do payload, 0=variável até FF)

PAYLOAD bytes seguem com semântica do domínio.

Exemplo: comando "throttle 47%, então estado IDLE"
  HEADER: [0011][0010]  = command, 2 bytes payload
  PAYLOAD: [0001][0000] = action_id=1 (throttle)
  PAYLOAD: [0110][0000] = 96 (out of 256, ≈ 37%)... ou usa domain=percent
  
  Mais limpo:
  HEADER: [0110][0001]  = percent, 1 byte payload
  PAYLOAD: [0010][1111] = 47

  HEADER: [0100][0001]  = state, 1 byte payload
  PAYLOAD: [0010][0000] = família=RESSONADOR, nível=0 (IDLE)
```

**Vantagens:** unifica multi-domínio (número + texto + estado em um stream), header explícito facilita parsing, compatível com FSM bytes do `fsm-cognitive-architecture`.
**Desvantagens:** overhead de 1 byte por bloco, tamanho variável quebra alinhamento SIMD.

---

## Critérios de comparação

Cada path será medido em:

| Critério | Como mede |
|----------|-----------|
| **Compactness** | bytes médios para representar 1M números aleatórios em [-1e6, 1e6] |
| **Round-trip fidelity** | (decode ∘ encode)(x) ≈ x para 1M valores |
| **Encode throughput** | ns/encode em CPU single-thread |
| **Decode throughput** | ns/decode em CPU single-thread |
| **Mutation locality** | flip 1 byte aleatório → magnitude da mudança no valor decodificado |
| **Arithmetic ergonomics** | linhas de código para implementar `add` |
| **Multi-domain capacity** | capaz de carregar número + texto + estado num stream sem perda? (sim/não) |

## Ground truth

Toda operação é validada contra `f64` (IEEE 754 double precision):
- Round-trip fidelity: `|decode(encode(x)) - x| < ε`
- Aritmética: `|byte_op(a,b) - f64_op(a,b)| < ε`

ε é definido por path (mais bits de mantissa equivalente = ε menor).

## Hipótese a falsificar

> H0 (null): formato de byte estruturado é estritamente pior que f64 em todos os critérios mensuráveis.
> H1: existe ao menos um path/critério onde byte estruturado é competitivo OU significativamente melhor (interpretabilidade, mutation locality, multi-domain).

Se H0 não for rejeitado em nenhum experimento, a hipótese cai.
Se H1 for confirmada em ≥1 experimento, há base para próxima fase.
