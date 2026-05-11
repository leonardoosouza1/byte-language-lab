# Experimento 16 — SMG-tiny distillation from Qwen 2.5 1.5B

**Tese:** o modelo aprende o **bit**, não a semântica. Supervisão = bytes do Qwen.

Ver `docs/smg-design.md` §6 para critérios de sucesso/falha.

## Setup

- Teacher: **Qwen 2.5 1.5B** (greedy, T=0, top_k=1) via iara-engine HTTP.
- Student: SMG-tiny baseline (MLP 1-hidden, vocab=41, window=48, hidden=512).
- Encoding: Path C tagged. `[T:N prompt][T:M response]\xFE`
- Treino: 100 prompts × 24 reps = 2400 streams, 71376 pares window→target, 30 épocas.
- SGD paralelo: 12 threads, batch_size=64, lr=0.064.
- Tempo: 345.5s.

## Loss curve

| Epoch | Loss |
|------:|-----:|
| 1 | 2.0896 |
| 5 | 0.0753 |
| 10 | 0.0174 |
| 15 | 0.0091 |
| 20 | 0.0060 |
| 25 | 0.0044 |
| 30 | 0.0035 |

- Final loss: **0.0035**
- Train next-byte accuracy: **100.0%**

## Byte-agreement com teacher

| Prompt | Qwen (teacher) | SMG-tiny (student) | edit/len | exact |
|--------|----------------|--------------------|---------:|:-----:|
| qual e a capital da franca | `a capital da franca e paris` | `a capital da franca e paris` | 0/27 | ✓ |
| qual e a capital da italia | `a capital da italia e roma` | `a capital da italia e roma` | 0/26 | ✓ |
| qual e a capital da alemanha | `a capital da alemanha e berlim` | `a capital da alemanha e berlim` | 0/30 | ✓ |
| qual e a capital da espanha | `a capital da espanha e madrid` | `a capital da espanha e madrid` | 0/29 | ✓ |
| qual e a capital de portugal | `a capital de portugal e lisboa` | `a capital de portugal e lisboa` | 0/30 | ✓ |
| qual e a capital do brasil | `a capital do brasil e brasilia` | `a capital do brasil e brasilia` | 0/30 | ✓ |
| qual e a capital do japao | `a capital do japao e toquio` | `a capital do japao e toquio` | 0/27 | ✓ |
| qual e a capital da china | `a capital da china e pequim em` | `a capital da china e pequim em` | 0/30 | ✓ |
| qual e a capital da russia | `a capital da russia e moscou` | `a capital da russia e moscou` | 0/28 | ✓ |
| qual e a capital da inglaterra | `a capital da inglaterra e` | `a capital da inglaterra e` | 0/25 | ✓ |
| qual e a capital do egito | `a capital do egito e caire` | `a capital do egito e caire` | 0/26 | ✓ |
| qual e a capital da grecia | `a capital da grecia e atenas` | `a capital da grecia e atenas` | 0/28 | ✓ |
| qual e a capital da turquia | `a capital da turquia e ankara` | `a capital da turquia e ankara` | 0/29 | ✓ |
| qual e a capital do canada | `a capital do canada e ottawa` | `a capital do canada e ottawa` | 0/28 | ✓ |
| qual e a capital da argentina | `a capital da argentina e buenos` | `a capital da argentina e buenos` | 0/31 | ✓ |
| qual e a capital do chile | `a capital do chile e santiago` | `a capital do chile e santiago` | 0/29 | ✓ |
| qual e a capital do peru | `a capital do peru e lima` | `a capital do peru e lima` | 0/24 | ✓ |
| qual e a capital do uruguai | `a capital do uruguai e montevide` | `a capital do uruguai e montevide` | 0/32 | ✓ |
| qual e a capital do paraguai | `a capital do paraguai e asuncao` | `a capital do paraguai e asuncao` | 0/31 | ✓ |
| qual e a capital da bolivia | `a capital da bolivia e la paz` | `a capital da bolivia e la paz` | 0/29 | ✓ |
| quanto e 2 mais 2 | `2 mais 2 e 4` | `2 mais 2 e 4` | 0/12 | ✓ |
| quanto e 7 vezes 8 | `7 vezes 8 e 56` | `7 vezes 8 e 56` | 0/14 | ✓ |
| quanto e 100 menos 37 | `100 menos 37 e igual a 63` | `100 menos 37 e igual a 63` | 0/25 | ✓ |
| quanto e 144 dividido por 12 | `144 dividido por 12 e igual a 12` | `144 dividido por 12 e igual a 12` | 0/32 | ✓ |
| quanto e 5 ao quadrado | `5 ao quadrado e 25` | `5 ao quadrado e 25` | 0/18 | ✓ |
| quanto e 10 ao cubo | `10 ao cubo e 10 x 10 x 10 que e` | `10 ao cubo e 10 x 10 x 10 que e` | 0/31 | ✓ |
| qual e a raiz quadrada de 81 | `a raiz quadrada de 81 e 9` | `a raiz quadrada de 81 e 9` | 0/25 | ✓ |
| quanto e 25 por cento de 200 | `25 por cento de 200 e igual a 50` | `25 por cento de 200 e igual a 50` | 0/32 | ✓ |
| quanto e 1 mais 1 | `1 mais 1 e 2` | `1 mais 1 e 2` | 0/12 | ✓ |
| quanto e 9 vezes 9 | `9 vezes 9 e 81` | `9 vezes 9 e 81` | 0/14 | ✓ |
| qual e o oposto de quente | `o oposto de quente e frio` | `o oposto de quente e frio` | 0/25 | ✓ |
| qual e o oposto de grande | `o oposto de grande e pequeno` | `o oposto de grande e pequeno` | 0/28 | ✓ |
| qual e o oposto de rapido | `o oposto de rapido e lento` | `o oposto de rapido e lento` | 0/26 | ✓ |
| qual e o oposto de claro | `o oposto de claro e escurto` | `o oposto de claro e escurto` | 0/27 | ✓ |
| qual e o oposto de feliz | `o oposto de feliz e triste` | `o oposto de feliz e triste` | 0/26 | ✓ |
| qual e a cor do ceu | `a cor do ceu e uma questao de` | `a cor do ceu e uma questao de` | 0/29 | ✓ |
| qual e a cor do sangue | `a cor do sangue e geralmente` | `a cor do sangue e geralmente` | 0/28 | ✓ |
| qual e a cor da grama | `a cor da grama e geralmente` | `a cor da grama e geralmente` | 0/27 | ✓ |
| qual e a cor da neve | `a cor da neve e geralmente` | `a cor da neve e geralmente` | 0/26 | ✓ |
| qual e a cor do sol | `a cor do sol e geralmente` | `a cor do sol e geralmente` | 0/25 | ✓ |
| quantas patas tem um cachorro | `um cachorro tem 4 patas` | `um cachorro tem 4 patas` | 0/23 | ✓ |
| quantas patas tem uma aranha | `uma aranha tem 8 patas` | `uma aranha tem 8 patas` | 0/22 | ✓ |
| quantas patas tem uma formiga | `uma formiga tem 6 patas` | `uma formiga tem 6 patas` | 0/23 | ✓ |
| quantas asas tem uma borboleta | `a resposta depende do tipo` | `a resposta depende do tipo` | 0/26 | ✓ |
| quantos olhos tem um humano | `um humano tem 2 olhos` | `um humano tem 2 olhos` | 0/21 | ✓ |
| quantos dedos tem uma mao | `uma mao tem 5 dedos` | `uma mao tem 5 dedos` | 0/19 | ✓ |
| quantas estacoes tem o ano | `o ano tem 12 estacoes` | `o ano tem 12 estacoes` | 0/21 | ✓ |
| quantos meses tem o ano | `o ano tem 12 meses` | `o ano tem 12 meses` | 0/18 | ✓ |
| quantos dias tem uma semana | `um dia tem 24 horas uma hora` | `um dia tem 24 horas uma hora` | 0/28 | ✓ |
| quantas horas tem um dia | `um dia tem 24 horas` | `um dia tem 24 horas` | 0/19 | ✓ |
| qual e o maior planeta do sistema solar | `o maior planeta do sistema` | `o maior planeta do sistema` | 0/26 | ✓ |
| qual e o planeta mais proximo do sol | `o planeta mais proximo do sol e` | `o planeta mais proximo do sol e` | 0/31 | ✓ |
| qual e o nosso planeta | `o nosso planeta e a terra` | `o nosso planeta e a terra` | 0/25 | ✓ |
| qual e o satelite natural da terra | `o satelite natural da terra e a` | `o satelite natural da terra e a` | 0/31 | ✓ |
| quantos planetas tem o sistema solar | `o sistema solar tem 8 planetas` | `o sistema solar tem 8 planetas` | 0/30 | ✓ |
| qual e o oceano maior | `o oceano maior do mundo e o` | `o oceano maior do mundo e o` | 0/27 | ✓ |
| qual e o rio mais longo do mundo | `o rio mais longo do mundo e o` | `o rio mais longo do mundo e o` | 0/29 | ✓ |
| qual e a montanha mais alta do mundo | `a montanha mais alta do mundo e` | `a montanha mais alta do mundo e` | 0/31 | ✓ |
| qual e o deserto maior do mundo | `o deserto maior do mundo e o` | `o deserto maior do mundo e o` | 0/28 | ✓ |
| qual e o continente onde fica o brasil | `o brasil esta localizado no` | `o brasil esta localizado no` | 0/27 | ✓ |
| quem escreveu dom quixote | `dom quixote foi escrito por` | `dom quixote foi escrito por` | 0/27 | ✓ |
| quem pintou a mona lisa | `a mona lisa foi pintada por` | `a mona lisa foi pintada por` | 0/27 | ✓ |
| quem inventou o telefone | `o telefone foi inventado por` | `o telefone foi inventado por` | 0/28 | ✓ |
| quem descobriu a america | `a descoberta da america pelo` | `a descoberta da america pelo` | 0/28 | ✓ |
| quem foi albert einstein | `albert einstein foi um fisico` | `albert einstein foi um fisico` | 0/29 | ✓ |
| quem foi isaac newton | `isaac newton foi um dos principa` | `isaac newton foi um dos principa` | 0/32 | ✓ |
| quem escreveu hamlet | `hamlet e um romance historico` | `hamlet e um romance historico` | 0/29 | ✓ |
| quem foi cleopatra | `cleopatra e uma figura historica` | `cleopatra e uma figura historica` | 0/32 | ✓ |
| quem foi napoleao | `napoleao bonaparte 17691821 foi` | `napoleao bonaparte 17691821 foi` | 0/31 | ✓ |
| quem foi dom pedro primeiro | `dom pedro i tambem conhecido` | `dom pedro i tambem conhecido` | 0/28 | ✓ |
| em que ano o brasil foi descoberto | `o brasil foi descoberto pelo` | `o brasil foi descoberto pelo` | 0/28 | ✓ |
| em que ano caiu o muro de berlim | `o muro de berlim caiu em 9 de` | `o muro de berlim caiu em 9 de` | 0/29 | ✓ |
| em que ano o homem pisou na lua | `o homem pisou na lua em 1969` | `o homem pisou na lua em 1969` | 0/28 | ✓ |
| em que ano comecou a segunda guerra | `a segunda guerra mundial` | `a segunda guerra mundial` | 0/24 | ✓ |
| em que ano terminou a segunda guerra | `a segunda guerra mundial` | `a segunda guerra mundial` | 0/24 | ✓ |
| em que ano o brasil ganhou a primeira | `o brasil ganhou a primeira copa` | `o brasil ganhou a primeira copa` | 0/31 | ✓ |
| em que ano nasceu pele | `pele nasceu em 23 de outubro de` | `pele nasceu em 23 de outubro de` | 0/31 | ✓ |
| em que ano o titanic afundou | `o titanic afundou em 15 de` | `o titanic afundou em 15 de` | 0/26 | ✓ |
| em que ano caiu o imperio romano | `o imperio romano caiu em 476 d` | `o imperio romano caiu em 476 d` | 0/30 | ✓ |
| em que ano colombo chegou na america | `colombo chegou a america do sul` | `colombo chegou a america do sul` | 0/31 | ✓ |
| o gato e um animal | `sim o gato e um animal` | `sim o gato e um animal` | 0/22 | ✓ |
| a baleia e um peixe | `nao uma baleia nao e um peixe` | `nao uma baleia nao e um peixe` | 0/29 | ✓ |
| o tomate e uma fruta | `nao o tomate nao e uma fruta` | `nao o tomate nao e uma fruta` | 0/28 | ✓ |
| a lua e uma estrela | `nao a lua nao e uma estrela` | `nao a lua nao e uma estrela` | 0/27 | ✓ |
| o sol e uma estrela | `o sol e uma estrela muito` | `o sol e uma estrela muito` | 0/25 | ✓ |
| o pinguim voa | `o pinguim voa usando as suas` | `o pinguim voa usando as suas` | 0/28 | ✓ |
| o morcego voa | `o morcego voa usando as suas` | `o morcego voa usando as suas` | 0/28 | ✓ |
| a cobra tem patas | `sim uma cobra tem patas` | `sim uma cobra tem patas` | 0/23 | ✓ |
| o peixe respira no ar | `o peixe respira no ar atraves` | `o peixe respira no ar atraves` | 0/29 | ✓ |
| a planta precisa de luz | `sim a planta precisa de luz` | `sim a planta precisa de luz` | 0/27 | ✓ |
| o que e agua | `a agua e um composto quimico` | `a agua e um composto quimico` | 0/28 | ✓ |
| o que e fogo | `fogo e um processo fisico que` | `fogo e um processo fisico que` | 0/29 | ✓ |
| o que e vento | `o vento e um movimento do ar na` | `o vento e um movimento do ar na` | 0/31 | ✓ |
| o que e gelo | `gel e um estado de conservacao` | `gel e um estado de conservacao` | 0/30 | ✓ |
| o que e arcoiris | `arcoiris e um fenomeno natural` | `arcoiris e um fenomeno natural` | 0/30 | ✓ |
| o que e gravidade | `gravidade e a forca que nos` | `gravidade e a forca que nos` | 0/27 | ✓ |
| o que e eletricidade | `eletricidade e a corrente` | `eletricidade e a corrente` | 0/25 | ✓ |
| o que e o sol | `o sol e o centro do nosso` | `o sol e o centro do nosso` | 0/25 | ✓ |
| o que e o coracao | `o coracao e um orgao vital do` | `o coracao e um orgao vital do` | 0/29 | ✓ |
| o que e o cerebro | `o cerebro e o principal orgao` | `o cerebro e o principal orgao` | 0/29 | ✓ |

## Métricas finais

| Métrica | Valor |
|---------|------:|
| Records avaliados | 100 |
| Byte-exact match | 100/100 = 100.0% |
| Prefix-8 match  | 100/100 = 100.0% |
| Avg byte-agreement (1 - edit/len) | **100.0%** |

## Veredicto (vs `docs/smg-design.md` §6)

**✓ M1 ATINGIDO** — destilação byte-a-byte do Qwen é aprendível por modelo minúsculo.
Liberado para exp17 (substituir MLP por estrutura microcoluna FSM real).

