# Experimento 05 — Substrato bytewise unificado (FSM + numérico)

Demonstra que `byte-language-lab` e `fsm-cognitive-architecture` compartilham
o MESMO formalismo de byte estruturado: 8 bits com semântica posicional, onde
o domain header desambigua a interpretação (estado FSM vs número vs comando).

## Equivalência formal

| Domínio | Layout do byte | Interpretação |
|---------|----------------|---------------|
| FSM (cognitive) | `[família:3][nível:3][escala:1][spike:1]` | Estado de neurônio FSM |
| Path A (numérico) | `[digit:4][position:4]` | Dígito decimal posicionado |
| Path B (BCD) | `[digit:4][confidence:4]` | Dígito + grau de certeza |
| Path C State (this exp) | `[família:3][nível:3][escala:1][spike:1]` | **Idêntico ao FSM** |

**Conclusão estrutural:** `Block::State(byte)` no Path C aceita exatamente o mesmo
byte que `Microcoluna.neurons[i]` no fsm-cognitive-architecture. Não há tradução,
não há perda — é o MESMO byte com a MESMA interpretação.

## Frames sintéticos

Gerados 1000 frames mistos (estado FSM + sensores numéricos + comando) totalizando:

| Métrica | Valor |
|---------|------:|
| Total bytes | 90238 |
| Bytes/frame médio | 90.2 |
| Blocos/frame médio | 25.0 |
| Round-trip OK | 1000 (100.0%) |
| State bytes (FSM) | 16000 |
| Number blocks | 3000 |
| Average throttle (sanity) | 51.5% |

## Distribuição de famílias FSM observadas

| Família | Contagem |
|---------|---------:|
| DIFERENCIADOR | 2000 |
| AMPLIFICADOR | 2000 |
| MEMORIZADOR | 2000 |
| ALEATORIZADOR | 2000 |
| COMPARADOR | 2000 |
| INIBIDOR | 2000 |
| RESSONADOR | 2000 |
| INTEGRADOR | 2000 |

## Exemplo de frame (hex dump)

Frame #0 (89 bytes):
```
  26 66 72 61 6D 65 23 02 07 DF 25 70 65 72 63 3A 
  41 1D 41 35 41 40 41 60 41 80 41 A0 41 C0 41 E0 
  24 64 65 63 3A 41 00 41 20 41 40 41 79 41 95 41 
  BD 41 C0 41 E0 25 73 65 6E 73 3A 0B 88 77 AF 46 
  25 94 33 12 21 10 DF 0C BF 38 27 AF 26 15 24 63 
  82 31 30 DF 61 46 32 07 F3 
```

Decodificação por bloco:

```
  [ 0] Text("frame#")
  [ 1] Number(0.0)
  [ 2] Text("perc:")
  [ 3] State(0x1D)  → INTEGRADOR[7] rfx 🔥
  [ 4] State(0x35)  → DIFERENCIADOR[5] rfx 🔥
  [ 5] State(0x40)  → RESSONADOR[0] rfx ·
  [ 6] State(0x60)  → INIBIDOR[0] rfx ·
  [ 7] State(0x80)  → AMPLIFICADOR[0] rfx ·
  [ 8] State(0xA0)  → COMPARADOR[0] rfx ·
  [ 9] State(0xC0)  → MEMORIZADOR[0] rfx ·
  [10] State(0xE0)  → ALEATORIZADOR[0] rfx ·
  [11] Text("dec:")
  [12] State(0x00)  → INTEGRADOR[0] rfx ·
  [13] State(0x20)  → DIFERENCIADOR[0] rfx ·
  [14] State(0x40)  → RESSONADOR[0] rfx ·
  [15] State(0x79)  → INIBIDOR[6] rfx 🔥
  [16] State(0x95)  → AMPLIFICADOR[5] rfx 🔥
  [17] State(0xBD)  → COMPARADOR[7] rfx 🔥
  [18] State(0xC0)  → MEMORIZADOR[0] rfx ·
  [19] State(0xE0)  → ALEATORIZADOR[0] rfx ·
  [20] Text("sens:")
  [21] Number(87.42931205157348)
  [22] Number(-32.212683345323114)
  [23] Percent(70)
  [24] Command { action: 7, magnitude: 243 }
```

## Implicação

Esta validação empírica fecha o loop teórico:

1. **Universe** roda hoje com `cortex F32` + `SpecialistBrain F32` (lição empírica champion 22734m).
2. **fsm-cognitive-architecture** propõe `microcoluna 8 bytes = 8 neurônios FSM` como substrato pós-LLM.
3. **byte-language-lab** valida que bytes estruturados com semântica posicional são VIÁVEIS, MAIS APRENDÍVEIS que strings, e UNIFICÁVEIS multi-domain.
4. Este experimento mostra que itens 2 e 3 falam o **mesmo dialeto byte** — `Microcoluna.neurons[i]` é literalmente um byte que cabe num `Block::State` do Path C, sem conversão.

**O substrato cognitivo unificado existe e é mensurável.** Próximo: usá-lo num runtime real (Universe FSM specialist ou destilação Qwen).
