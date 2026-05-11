# Experimento 17 — SMG-grid vs MLP, generalização

Pergunta: substituindo MLP plano por grade 4×4 de microcolunas FSM (~32× menos parâmetros), a estrutura **generaliza melhor** para prompts não vistos?

Ver `docs/exp17-design.md` §1 para critério.

## Setup

- Teacher: Qwen 2.5 1.5B (mesmas 100 traces de exp16).
- Sanitize idêntico: prompt 40b max, response 32b max.
- Split: 80 train / 20 test, seed=4242.
- Encoding: Path C tagged `[T:N prompt][T:M response]\xFE`. Vocab=41.
- Window=48, batch=64, epochs=30, reps=24, threads=12.
- Pairs de treino: 57864.

### Modelos

| | params | LR | tempo |
|---|---:|---:|---:|
| MLP | 1029161 | 0.064 | 269.0s |
| SMG-grid | 91689 | 0.05 | 717.3s |

## Loss curves

| Epoch | MLP loss | Grid loss |
|------:|---------:|----------:|
| 1 | 2.1787 | 2.9304 |
| 5 | 0.0827 | 2.5126 |
| 10 | 0.0183 | 2.2991 |
| 15 | 0.0095 | 2.1878 |
| 20 | 0.0063 | 2.0944 |
| 25 | 0.0046 | 2.0123 |
| 30 | 0.0036 | 1.9468 |

## Métricas finais

| | byte-agreement train | exact train | byte-agreement TEST | exact TEST |
|---|---:|---:|---:|---:|
| MLP | 100.0% | 80/80 | **38.5%** | 0/20 |
| SMG-grid | 15.0% | 0/80 | **13.3%** | 0/20 |

## Test set (prompts não vistos)

| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |
|--------|-------------|-----|----------|---------:|----------:|
| `quantos dedos tem uma mao` | `uma mao tem 5 dedos` | `o daumtem 2o sora eo ce or` | `` | 17 | 19 |
| `o pinguim voa` | `o pinguim voa usando as suas` | `o sol e o canto e forosto` | `e e e e e e e e e e e e e e e e e e …` | 20 | 38 |
| `a lua e uma estrela` | `nao a lua nao e uma estrela` | `a o putal de poral a ue euradil` | `a e e e e e e e e e e e e e e e e e …` | 22 | 32 |
| `qual e a capital do japao` | `a capital do japao e toquio` | `a capital do egito e caire` | `a e e e e e e e e e e e e e e e e e …` | 10 | 34 |
| `a planta precisa de luz` | `sim a planta precisa de luz` | `a e a celet te eruan se` | `a e e e e e e e e e e e e e e e e e …` | 20 | 34 |
| `qual e o rio mais longo do mundo` | `o rio mais longo do mundo e o` | `a orisino auial ou elantr` | `a e e e e e e e e e e e e e e e e e …` | 22 | 35 |
| `quantas patas tem uma formiga` | `uma formiga tem 6 patas` | `a coresa sentem esumtem` | `a e e e e e e e e e e e e e e e e e …` | 15 | 32 |
| `quantas patas tem uma aranha` | `uma aranha tem 8 patas` | `a m cecaprta a canceatama e manatage` | `a e e e e e e e e e e e e e e e e e …` | 23 | 32 |
| `qual e o oposto de feliz` | `o oposto de feliz e triste` | `o oposto de clarl e escurto` | `a e e e e e e e e e e e e e e e e e …` | 11 | 32 |
| `qual e a capital da turquia` | `a capital da turquia e ankara` | `a capital da urugua a rm noma` | `a e e e e e e e e e e e e e e e e e …` | 11 | 36 |
| `quanto e 5 ao quadrado` | `5 ao quadrado e 25` | `a cor ao nera f do arem e a` | `a e e e e e e e e e e e e e e e e` | 17 | 28 |
| `o morcego voa` | `o morcego voa usando as suas` | `o sor nao o em o oitonoso bano ero` | `e e e e e e e e e e e e e e e e e e …` | 23 | 37 |
| `quantos meses tem o ano` | `o ano tem 12 meses` | `o o umum ememo sol eo` | `` | 14 | 18 |
| `quem foi dom pedro primeiro` | `dom pedro i tambem conhecido` | `o copito e o bera fonson` | `` | 20 | 28 |
| `quanto e 7 vezes 8` | `7 vezes 8 e 56` | `9 vezes 9 e 81` | `` | 4 | 14 |
| `quem escreveu dom quixote` | `dom quixote foi escrito por` | `o oeum ao certom nooris` | `a e e e e e e e e e e e e e e e e e …` | 20 | 35 |
| `quantos dias tem uma semana` | `um dia tem 24 horas uma hora` | `a coumeno de a emandaado e n` | `a e e e e e e e e e e e e e e e e e …` | 25 | 36 |
| `qual e a capital da italia` | `a capital da italia e roma` | `a capital da frania e parisi` | `a e e e e e e e e e e e e e e e e e …` | 8 | 34 |
| `qual e a capital da bolivia` | `a capital da bolivia e la paz` | `a capital da espinai e a duma` | `a e e e e e e e e e e e e e e e e e …` | 11 | 35 |
| `qual e o planeta mais proximo do sol` | `o planeta mais proximo do sol e` | `` | `a e e e e e e e e e e e e e e e e e …` | 31 | 37 |

## Train set (sanity check, prompts vistos)

<details><summary>80 prompts (clicar para expandir)</summary>

| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |
|--------|-------------|-----|----------|---------:|----------:|
| `em que ano caiu o muro de berlim` | `o muro de berlim caiu em 9 de` | `o muro de berlim caiu em 9 de` | `a e e e e e e e e e e e e e e e e e …` | 0 | 32 |
| `o que e arcoiris` | `arcoiris e um fenomeno natural` | `arcoiris e um fenomeno natural` | `` | 0 | 30 |
| `o que e vento` | `o vento e um movimento do ar na` | `o vento e um movimento do ar na` | `e e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e o oceano maior` | `o oceano maior do mundo e o` | `o oceano maior do mundo e o` | `` | 0 | 27 |
| `quanto e 2 mais 2` | `2 mais 2 e 4` | `2 mais 2 e 4` | `` | 0 | 12 |
| `quem foi albert einstein` | `albert einstein foi um fisico` | `albert einstein foi um fisico` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `quanto e 1 mais 1` | `1 mais 1 e 2` | `1 mais 1 e 2` | `` | 0 | 12 |
| `quem pintou a mona lisa` | `a mona lisa foi pintada por` | `a mona lisa foi pintada por` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e a cor do sangue` | `a cor do sangue e geralmente` | `a cor do sangue e geralmente` | `a e e e e e e e e e e e e e e e e e …` | 0 | 32 |
| `qual e a capital da russia` | `a capital da russia e moscou` | `a capital da russia e moscou` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `qual e a capital do uruguai` | `a capital do uruguai e montevide` | `a capital do uruguai e montevide` | `a e e e e e e e e e e e e e e e e e …` | 0 | 38 |
| `o tomate e uma fruta` | `nao o tomate nao e uma fruta` | `nao o tomate nao e uma fruta` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e o oposto de claro` | `o oposto de claro e escurto` | `o oposto de claro e escurto` | `` | 0 | 27 |
| `qual e a montanha mais alta do mundo` | `a montanha mais alta do mundo e` | `a montanha mais alta do mundo e` | `a e e e e e e e e e e e e e e e e e …` | 0 | 37 |
| `qual e a capital da alemanha` | `a capital da alemanha e berlim` | `a capital da alemanha e berlim` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `quanto e 25 por cento de 200` | `25 por cento de 200 e igual a 50` | `25 por cento de 200 e igual a 50` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `o gato e um animal` | `sim o gato e um animal` | `sim o gato e um animal` | `a e e e e e e e e e e e e e e e e e …` | 0 | 31 |
| `qual e o maior planeta do sistema so…` | `o maior planeta do sistema` | `o maior planeta do sistema` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `o peixe respira no ar` | `o peixe respira no ar atraves` | `o peixe respira no ar atraves` | `` | 0 | 29 |
| `em que ano colombo chegou na america` | `colombo chegou a america do sul` | `colombo chegou a america do sul` | `a e e e e e e e e e e e e e e e e e …` | 0 | 38 |
| `em que ano o brasil foi descoberto` | `o brasil foi descoberto pelo` | `o brasil foi descoberto pelo` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `em que ano o homem pisou na lua` | `o homem pisou na lua em 1969` | `o homem pisou na lua em 1969` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e a capital da china` | `a capital da china e pequim em` | `a capital da china e pequim em` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `quantas horas tem um dia` | `um dia tem 24 horas` | `um dia tem 24 horas` | `a e e e e e e e e e e e e e e e e` | 0 | 28 |
| `qual e a capital da grecia` | `a capital da grecia e atenas` | `a capital da grecia e atenas` | `a e e e e e e e e e e e e e e e e e …` | 0 | 34 |
| `qual e a capital da inglaterra` | `a capital da inglaterra e` | `a capital da inglaterra e` | `a e e e e e e e e e e e e e e e e e …` | 0 | 32 |
| `qual e o deserto maior do mundo` | `o deserto maior do mundo e o` | `o deserto maior do mundo e o` | `a e e e e e e e e e e e e e e e e e …` | 0 | 34 |
| `quem descobriu a america` | `a descoberta da america pelo` | `a descoberta da america pelo` | `a e e e e e e e e e e e e e e e e e …` | 0 | 34 |
| `quanto e 144 dividido por 12` | `144 dividido por 12 e igual a 12` | `144 dividido por 12 e igual a 12` | `` | 0 | 32 |
| `o que e gelo` | `gel e um estado de conservacao` | `gel e um estado de conservacao` | `e e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e o satelite natural da terra` | `o satelite natural da terra e a` | `o satelite natural da terra e a` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `a cobra tem patas` | `sim uma cobra tem patas` | `sim uma cobra tem patas` | `a e e e e e e e e e e e e e e e e e …` | 0 | 32 |
| `qual e a capital de portugal` | `a capital de portugal e lisboa` | `a capital de portugal e lisboa` | `a e e e e e e e e e e e e e e e e e …` | 0 | 37 |
| `qual e a capital do chile` | `a capital do chile e santiago` | `a capital do chile e santiago` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e a capital da argentina` | `a capital da argentina e buenos` | `a capital da argentina e buenos` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `quem inventou o telefone` | `o telefone foi inventado por` | `o telefone foi inventado por` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `em que ano terminou a segunda guerra` | `a segunda guerra mundial` | `a segunda guerra mundial` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `o sol e uma estrela` | `o sol e uma estrela muito` | `o sol e uma estrela muito` | `a e e e e e e e e e e e e e e e e e …` | 0 | 31 |
| `o que e o cerebro` | `o cerebro e o principal orgao` | `o cerebro e o principal orgao` | `` | 0 | 29 |
| `o que e fogo` | `fogo e um processo fisico que` | `fogo e um processo fisico que` | `e e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e a capital da espanha` | `a capital da espanha e madrid` | `a capital da espanha e madrid` | `a e e e e e e e e e e e e e e e e e …` | 0 | 35 |
| `qual e o oposto de rapido` | `o oposto de rapido e lento` | `o oposto de rapido e lento` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `quanto e 9 vezes 9` | `9 vezes 9 e 81` | `9 vezes 9 e 81` | `` | 0 | 14 |
| `qual e a raiz quadrada de 81` | `a raiz quadrada de 81 e 9` | `a raiz quadrada de 81 e 9` | `` | 0 | 25 |
| `em que ano caiu o imperio romano` | `o imperio romano caiu em 476 d` | `o imperio romano caiu em 476 d` | `` | 0 | 30 |
| `o que e gravidade` | `gravidade e a forca que nos` | `gravidade e a forca que nos` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `qual e a cor da neve` | `a cor da neve e geralmente` | `a cor da neve e geralmente` | `a e e e e e e e e e e e e e e e e e …` | 0 | 29 |
| `qual e a capital do egito` | `a capital do egito e caire` | `a capital do egito e caire` | `` | 0 | 26 |
| `qual e a capital do paraguai` | `a capital do paraguai e asuncao` | `a capital do paraguai e asuncao` | `a e e e e e e e e e e e e e e e e e …` | 0 | 38 |
| `qual e a cor do ceu` | `a cor do ceu e uma questao de` | `a cor do ceu e uma questao de` | `a e e e e e e e e e e e e e e e e e …` | 0 | 31 |
| `quantos planetas tem o sistema solar` | `o sistema solar tem 8 planetas` | `o sistema solar tem 8 planetas` | `a e e e e e e e e e e e e e e e e e …` | 0 | 37 |
| `quem foi isaac newton` | `isaac newton foi um dos principa` | `isaac newton foi um dos principa` | `a e e e e e e e e e e e e e e e e e …` | 0 | 41 |
| `qual e a capital do brasil` | `a capital do brasil e brasilia` | `a capital do brasil e brasilia` | `a e e e e e e e e e e e e e e e e e …` | 0 | 38 |
| `em que ano o titanic afundou` | `o titanic afundou em 15 de` | `o titanic afundou em 15 de` | `a e e e e e e e e e e e e e e e e e …` | 0 | 34 |
| `qual e o nosso planeta` | `o nosso planeta e a terra` | `o nosso planeta e a terra` | `a e e e e e e e e e e e e e e e e e …` | 0 | 31 |
| `quem foi napoleao` | `napoleao bonaparte 17691821 foi` | `napoleao bonaparte 17691821 foi` | `a e e e e e e e e e e e e e e e e e …` | 0 | 40 |
| `qual e a cor do sol` | `a cor do sol e geralmente` | `a cor do sol e geralmente` | `a e e e e e e e e e e e e e e e e e …` | 0 | 29 |
| `quantas patas tem um cachorro` | `um cachorro tem 4 patas` | `um cachorro tem 4 patas` | `` | 0 | 23 |
| `o que e o coracao` | `o coracao e um orgao vital do` | `o coracao e um orgao vital do` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `quantos olhos tem um humano` | `um humano tem 2 olhos` | `um humano tem 2 olhos` | `` | 0 | 21 |
| `em que ano nasceu pele` | `pele nasceu em 23 de outubro de` | `pele nasceu em 23 de outubro de` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `o que e eletricidade` | `eletricidade e a corrente` | `eletricidade e a corrente` | `a e e e e e e e e e e e e e e e e e …` | 0 | 30 |
| `qual e o oposto de grande` | `o oposto de grande e pequeno` | `o oposto de grande e pequeno` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `qual e a cor da grama` | `a cor da grama e geralmente` | `a cor da grama e geralmente` | `a e e e e e e e e e e e e e e e e e …` | 0 | 31 |
| `a baleia e um peixe` | `nao uma baleia nao e um peixe` | `nao uma baleia nao e um peixe` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `quanto e 100 menos 37` | `100 menos 37 e igual a 63` | `100 menos 37 e igual a 63` | `` | 0 | 25 |
| `quem escreveu hamlet` | `hamlet e um romance historico` | `hamlet e um romance historico` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `o que e o sol` | `o sol e o centro do nosso` | `o sol e o centro do nosso` | `do e e e e e e e e e e e e e e e e e…` | 0 | 31 |
| `quantas asas tem uma borboleta` | `a resposta depende do tipo` | `a resposta depende do tipo` | `a e e e e e e e e e e e e e e e e e …` | 0 | 32 |
| `quantas estacoes tem o ano` | `o ano tem 12 estacoes` | `o ano tem 12 estacoes` | `` | 0 | 21 |
| `em que ano comecou a segunda guerra` | `a segunda guerra mundial` | `a segunda guerra mundial` | `a e e e e e e e e e e e e e e e e e …` | 0 | 33 |
| `qual e a capital do canada` | `a capital do canada e ottawa` | `a capital do canada e ottawa` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `qual e o continente onde fica o bras…` | `o brasil esta localizado no` | `o brasil esta localizado no` | `a e e e e e e e e e e e e e e e e e …` | 0 | 36 |
| `qual e a capital do peru` | `a capital do peru e lima` | `a capital do peru e lima` | `a e e e e e e e e e e e e e e e e e …` | 0 | 31 |
| `o que e agua` | `a agua e um composto quimico` | `a agua e um composto quimico` | `e e e e e e e e e e e e e e e e e e …` | 0 | 37 |
| `quanto e 10 ao cubo` | `10 ao cubo e 10 x 10 x 10 que e` | `10 ao cubo e 10 x 10 x 10 que e` | `` | 0 | 31 |
| `quem foi cleopatra` | `cleopatra e uma figura historica` | `cleopatra e uma figura historica` | `a e e e e e e e e e e e e e e e e e …` | 0 | 41 |
| `qual e o oposto de quente` | `o oposto de quente e frio` | `o oposto de quente e frio` | `a e e e e e e e e e e e e e e e e e …` | 0 | 30 |
| `qual e a capital da franca` | `a capital da franca e paris` | `a capital da franca e paris` | `a e e e e e e e e e e e e e e e e e …` | 0 | 34 |
| `em que ano o brasil ganhou a primeir…` | `o brasil ganhou a primeira copa` | `o brasil ganhou a primeira copa` | `a e e e e e e e e e e e e e e e e e …` | 0 | 39 |
</details>

## Veredicto

**✗ Estrutura piora** — SMG-grid perde 25.3pp para MLP em test. Possíveis causas: BPTT instável, K muito pequeno, ou tese 'byte-language gera estrutura' precisa ser repensada.

