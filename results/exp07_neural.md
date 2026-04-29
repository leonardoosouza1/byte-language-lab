# Experimento 07 — Tiny MLP gradient-trained

**Hipótese decisiva:** se byte-positional reduz perplexity n-gram (exp03/06),
um MODELO TREINÁVEL deveria também aprender mais rápido sob essa estrutura.
Aqui treinamos um MLP de 1 camada oculta (sem deps externas) por encoding

**Setup:**
- Window K = 6 tokens previous
- Hidden H = 32 units
- Learning rate = 0.01
- Steps = 8000 (SGD, 1 sample per step)
- Eval test loss every 250 steps
- Train/test split: 80%/20%

## Resultados finais

| Encoding | Vocab | Train pairs | Test pairs | Final train loss | Final test loss | Test accuracy | Steps→loss<2.0 |
|----------|------:|------------:|-----------:|-----------------:|----------------:|--------------:|---------------:|
| ASCII | 15 | 14428 | 3606 | 1.777 | 1.789 | 39.6% | 2000 |
| Qwen BPE | 14 | 10449 | 2613 | 2.145 | 2.113 | 24.6% | never |
| Byte-positional (Path B) | 16 | 26428 | 6606 | 1.111 | 1.139 | 63.0% | 750 |
| Path C (domain-tagged) | 49 | 41156 | 10230 | 1.404 | 1.405 | 55.4% | 1750 |

## Loss curves (test)

Loss em test set (subset 500 amostras) a cada 250 steps:

| Step | ASCII | Qwen BPE | Path B | Path C |
|-----:|------:|---------:|-------:|-------:|
| 0 | 2.715 | 2.661 | 2.744 | 3.907 |
| 250 | 2.494 | 2.557 | 2.348 | 3.552 |
| 500 | 2.361 | 2.487 | 2.061 | 3.085 |
| 750 | 2.265 | 2.431 | 1.861 | 2.648 |
| 1000 | 2.199 | 2.404 | 1.727 | 2.333 |
| 1250 | 2.132 | 2.371 | 1.621 | 2.179 |
| 1500 | 2.075 | 2.331 | 1.557 | 2.055 |
| 1750 | 2.066 | 2.322 | 1.492 | 1.957 |
| 2000 | 1.997 | 2.317 | 1.414 | 1.883 |
| 2250 | 1.996 | 2.262 | 1.369 | 1.805 |
| 2500 | 1.952 | 2.225 | 1.337 | 1.759 |
| 2750 | 1.934 | 2.223 | 1.300 | 1.711 |
| 3000 | 1.917 | 2.188 | 1.287 | 1.686 |
| 3250 | 1.905 | 2.217 | 1.262 | 1.656 |
| 3500 | 1.892 | 2.175 | 1.243 | 1.622 |
| 3750 | 1.865 | 2.155 | 1.228 | 1.604 |
| 4000 | 1.860 | 2.132 | 1.225 | 1.598 |
| 4250 | 1.849 | 2.136 | 1.208 | 1.565 |
| 4500 | 1.857 | 2.162 | 1.201 | 1.543 |
| 4750 | 1.850 | 2.144 | 1.219 | 1.534 |
| 5000 | 1.819 | 2.139 | 1.173 | 1.529 |
| 5250 | 1.852 | 2.154 | 1.169 | 1.505 |
| 5500 | 1.848 | 2.142 | 1.170 | 1.502 |
| 5750 | 1.832 | 2.129 | 1.166 | 1.492 |
| 6000 | 1.821 | 2.152 | 1.168 | 1.508 |
| 6250 | 1.868 | 2.130 | 1.154 | 1.499 |
| 6500 | 1.816 | 2.147 | 1.149 | 1.511 |
| 6750 | 1.813 | 2.127 | 1.139 | 1.439 |
| 7000 | 1.822 | 2.161 | 1.152 | 1.435 |
| 7250 | 1.836 | 2.113 | 1.162 | 1.448 |
| 7500 | 1.833 | 2.115 | 1.148 | 1.405 |
| 7750 | 1.799 | 2.150 | 1.145 | 1.427 |
| 7999 | 1.789 | 2.132 | 1.147 | 1.409 |

## Análise

**Vencedor (final test loss):** Byte-positional (Path B) (1.139)

**Vencedor (test accuracy):** Byte-positional (Path B) (63.0%)

**Steps to loss < 2.0** (proxy de velocidade de convergência):
- ASCII → step 2000
- Qwen BPE → never reached
- Byte-positional (Path B) → step 750
- Path C (domain-tagged) → step 1750

## Implicação

Este é o teste empírico mais direto da hipótese. Se Path B/C convergiram MAIS RÁPIDO
ou alcançaram MENOR loss, isso é evidência que a estrutura posicional dos bytes
ajuda um modelo aprendido a capturar relações aritméticas — exatamente o tipo de
tarefa onde LLMs falham (multi-digit math).

Caveats:
- MLP de 1 camada é muito simples; resultados podem mudar com transformers.
- Vocab effective varia (ASCII=15, Qwen≈14, Path B≈16, Path C≈49). Maior vocab pode dificultar.
- Steps fixos não compensam diferença de stream length (Path C tem mais tokens, mais oportunidades de update).
- Para confirmação rigorosa, seria preciso transformer + mais dados + ablations.
