# Experimento 04 — Multi-domain byte stream composition

Demonstração: número, texto, comando, estado FSM, boolean e percent
coexistem em UM único stream de bytes (Path C, domain-tagged).

## Sentenças testadas

| Caso | Blocos | Bytes total | Bytes/bloco | Domínios | Header overhead | JSON equiv | BP/JSON |
|------|------:|------------:|-----------:|:---------|---------------:|-----------:|--------:|
| telemetry | 8 | 34 | 4.2 | Text, Number, Percent, State, Boolean | 24% | 89 | 0.38× |
| command | 7 | 33 | 4.7 | Text, Command, Number | 21% | 84 | 0.39× |
| game_state | 8 | 37 | 4.6 | Text, Number, Percent, Boolean | 22% | 93 | 0.40× |
| **Total** | — | **104** | — | — | — | **266** | **0.39×** |

## Exemplo: hex dump da sentença `telemetry`

Bytes (34 total):
```
  26 66 72 61 6D 65 3A 06 19 28 77 AF 56 DF 05 BF 
  37 AF 26 DF 61 3C 61 00 05 07 AF 16 55 DF 41 9D 
  51 00 
```

Decodificação por bloco:
```
  [0] Text("frame:")
  [1] Number(127.5)
  [2] Number(-3.2)
  [3] Percent(60)
  [4] Percent(0)
  [5] Number(0.15)
  [6] State(157)
  [7] Boolean(false)
```

## Análise

- **Compactness vs JSON:** Path C usa ~39% do espaço de JSON equivalente.
- **Round-trip 100%:** todos os blocos preservaram identidade após encode/decode.
- **Multi-domain nativo:** mesma stream contém número (Path A no payload), texto UTF-8, comando estruturado, estado FSM byte, boolean e percent — sem necessidade de tokenizer separado por tipo.
- **Implicação:** "linguagem de bytes" pode unificar input/output em sistemas que mistura semântica (ex: assistant que recebe texto + telemetria + comandos).
