# Experimento 15 — Text learnability: capital de país

**Pergunta canônica:** _"qual é a capital da frança?"_  
**Resposta esperada:** `paris`

**Tese:** byte-positional foi validado em ARITMÉTICA (exp02-14).
Aqui medimos se um MLP minúsculo (1 hidden, ~250k params) consegue
memorizar associações país→capital, e se Path C tagged ajuda vs ASCII raw.

**Setup:**
- 20 pares país→capital (português, ASCII normalizado)
- 40 repetições por par (total 800 sequências de treino)
- MLP: window=16, hidden=256, lr=0.01, epochs=100
- Test: prompt = país (formatado conforme encoding); gerar greedy até newline/EOB.

## Pares treinados

| País | Capital |
|------|---------|
| franca | paris |
| italia | roma |
| alemanha | berlim |
| espanha | madrid |
| portugal | lisboa |
| brasil | brasilia |
| japao | toquio |
| china | pequim |
| russia | moscou |
| inglaterra | londres |
| egito | cairo |
| grecia | atenas |
| turquia | ancara |
| canada | ottawa |
| argentina | buenosaires |
| chile | santiago |
| peru | lima |
| uruguai | montevideu |
| paraguai | assuncao |
| bolivia | lapaz |

## Resultados

### ASCII raw — `frança:paris\n`

- Vocab: 26 unique bytes
- Bytes/sequence (média): 23.4
- Training (window→target) pairs: 5960

**Training loss curve:**

| Epoch | Loss |
|------:|-----:|
| 1 | 1.0425 |
| 10 | 0.0030 |
| 20 | 0.0012 |
| 30 | 0.0008 |
| 40 | 0.0005 |
| 50 | 0.0004 |
| 60 | 0.0003 |
| 70 | 0.0003 |
| 80 | 0.0002 |
| 90 | 0.0002 |
| 100 | 0.0002 |

- Final loss: **0.0002**
- Train accuracy (next-byte prediction): **100.0%**

**Test prompts (autoregressive generation, greedy):**

| País | Esperado | Gerado | Match |
|------|----------|--------|:-----:|
| franca | paris | `paris` | ✓ |
| italia | roma | `roma` | ✓ |
| alemanha | berlim | `berlim` | ✓ |
| espanha | madrid | `madrid` | ✓ |
| portugal | lisboa | `lisboa` | ✓ |
| brasil | brasilia | `brasilia` | ✓ |
| japao | toquio | `toquio` | ✓ |
| china | pequim | `pequim` | ✓ |
| russia | moscou | `moscou` | ✓ |
| inglaterra | londres | `londres` | ✓ |
| egito | cairo | `cairo` | ✓ |
| grecia | atenas | `atenas` | ✓ |
| turquia | ancara | `ancara` | ✓ |
| canada | ottawa | `ottawa` | ✓ |
| argentina | buenosaires | `buenosaires` | ✓ |
| chile | santiago | `santiago` | ✓ |
| peru | lima | `lima` | ✓ |
| uruguai | montevideu | `montevideu` | ✓ |
| paraguai | assuncao | `assuncao` | ✓ |
| bolivia | lapaz | `lapaz` | ✓ |

- **Test recall: 20/20 = 100%**

### Path C tagged — `[T:N frança][T:M paris]\xFE`

- Vocab: 32 unique bytes
- Bytes/sequence (média): 24.4
- Training (window→target) pairs: 6760

**Training loss curve:**

| Epoch | Loss |
|------:|-----:|
| 1 | 1.1131 |
| 10 | 0.0029 |
| 20 | 0.0012 |
| 30 | 0.0007 |
| 40 | 0.0005 |
| 50 | 0.0004 |
| 60 | 0.0003 |
| 70 | 0.0003 |
| 80 | 0.0002 |
| 90 | 0.0002 |
| 100 | 0.0002 |

- Final loss: **0.0002**
- Train accuracy (next-byte prediction): **100.0%**

**Test prompts (autoregressive generation, greedy):**

| País | Esperado | Gerado | Match |
|------|----------|--------|:-----:|
| franca | paris | `paris` | ✓ |
| italia | roma | `roma` | ✓ |
| alemanha | berlim | `berlim` | ✓ |
| espanha | madrid | `madrid` | ✓ |
| portugal | lisboa | `lisboa` | ✓ |
| brasil | brasilia | `brasilia` | ✓ |
| japao | toquio | `toquio` | ✓ |
| china | pequim | `pequim` | ✓ |
| russia | moscou | `moscou` | ✓ |
| inglaterra | londres | `londres` | ✓ |
| egito | cairo | `cairo` | ✓ |
| grecia | atenas | `atenas` | ✓ |
| turquia | ancara | `ancara` | ✓ |
| canada | ottawa | `ottawa` | ✓ |
| argentina | buenosaires | `buenosaires` | ✓ |
| chile | santiago | `santiago` | ✓ |
| peru | lima | `lima` | ✓ |
| uruguai | montevideu | `montevideu` | ✓ |
| paraguai | assuncao | `assuncao` | ✓ |
| bolivia | lapaz | `lapaz` | ✓ |

- **Test recall: 20/20 = 100%**

## Comparação

| Encoding | Final loss | Train acc | Recall (test) |
|----------|-----------:|----------:|--------------:|
| ASCII raw | 0.0002 | 100.0% | 20/20 (100%) |
| Path C tagged | 0.0002 | 100.0% | 20/20 (100%) |

## Demonstração canônica

Pergunta: **qual é a capital da frança?**

- Prompt enviado ao MLP: `franca:`
- Resposta gerada (greedy autoregressivo): `paris`
- Esperado: `paris`
- **✓ CORRETO — o MLP responde `paris`**

## Análise honesta

**O que foi provado:** Um MLP minúsculo (1 camada oculta, ~250k params,
0 dependências) memoriza 20 associações país→capital com 100% de recall
em apenas 100 épocas. Para a pergunta canônica _"qual é a capital da
frança?"_ o modelo responde **`paris`** corretamente, byte por byte,
via geração autoregressiva greedy.

**O que NÃO foi provado:**
1. **Memorização ≠ inteligência.** Tabela hash com 20 entradas e
   repetição também resolve isto — sem precisar de neural net. O MLP
   é apenas um lookup table com soft boundaries.
2. **Generalização semântica não testada.** O modelo NÃO sabe que
   "capital" significa algo. Ele só decorou padrões de bytes.
3. **Vantagem de byte-positional para texto: NULA neste teste.**
   ASCII e Path C convergem com loss idêntica e recall idêntico.
   Path C tem vocab+6 bytes (headers), 800 training pairs a mais —
   pequeno overhead, sem ganho mensurável.

**Implicação para a tese ampla:**

- Em ARITMÉTICA (exp02-14): byte-positional **vence** com larga margem
  (94% acc bytewise vs 25% Qwen 1.5B em multiplicação 5d×5d).
- Em TEXTO/Q&A: estruturação por header **não ajuda**. A diferença é
  cosmética (vocab size, pair count) sem efeito em learnability.
- Conclusão refinada: **"inteligência fala bytes estruturados"**
  vale para domínios COM estrutura matemática (números, posições).
  Para texto, a estrutura útil é semântica (embeddings, attention).
  Para Q&A simples (memorização), qualquer encoding razoável serve.

