# byte-language-lab

> Hipótese: **Inteligência fala bytes estruturados, não tokens nem floats.**
> Streams de bytes com positional coding são uma alternativa viável ao paradigma LLM-Transformer-FloatALU.

## O que esse lab investiga

LLMs tokenizam números como strings opacas e fazem aritmética por emergência estatística.
FP-ALU multiplica mantissas e alinha expoentes em silício especializado.
Ambos abstraem demais.

Este projeto explora um terceiro caminho: cada byte é um átomo semântico autocontido com 4 bits de **valor** + 4 bits de **metadata posicional**. Um número como `0.4847` vira sequência de bytes `[0,.,4,8,4,7]`, cada um interpretável sem ALU.

## Experimentos (executáveis isolados)

```bash
cargo run --release --bin exp01_format       # round-trip 1M números
cargo run --release --bin exp02_arithmetic   # +,-,*,/ bytewise vs f64 ground truth
cargo run --release --bin exp03_learn        # tiny learner: byte-positional vs char-string
cargo run --release --bin exp04_compose      # multi-domain stream (números + comandos + estados)
cargo run --release --bin run_all            # pipeline completo, gera RESULTS.md
```

Todos os experimentos:
- Reprodutíveis (seeds explícitos)
- Sem dependências externas (Rust puro stdlib)
- Geram artefato markdown próprio em `results/`
- Falham loud se invariantes quebram (`assert!` no path de validação)

## Estrutura

```
src/
  format/      3 variantes de byte format (path A/B/C)
  arithmetic/  +, -, *, / sobre byte streams
  learn/       tiny learner FSM-style + baseline char-string
  compose/     decoder multi-domínio
bin/           experimentos executáveis
tests/         integração end-to-end
results/       artefatos gerados pelos experimentos
```

## Filosofia

- Sem `unsafe` exceto em transmutes documentados
- Zero alocações em hot paths (verificado por testes específicos)
- Const generics e const fns onde possível (LUTs em compile time)
- Property tests em invariantes (round-trip, comutatividade, idempotência)
- Cada decisão tem comentário explicando POR QUÊ

Ver [`DESIGN.md`](DESIGN.md) para o formalismo dos formatos.
Ver [`PROGRESS.md`](PROGRESS.md) para status ao vivo enquanto roda.
Ver [`RESULTS.md`](RESULTS.md) para conclusões finais (gerado pelo `run_all`).
