# Experimento 17 — SMG-grid vs MLP, generalização

Pergunta: substituindo MLP plano por grade 4×4 de microcolunas FSM (~32× menos parâmetros), a estrutura **generaliza melhor** para prompts não vistos?

Ver `docs/exp17-design.md` §1 para critério.

## Setup

- Teacher: Qwen 2.5 1.5B (mesmas 100 traces de exp16).
- Sanitize idêntico: prompt 40b max, response 32b max.
- Split: 80 train / 20 test, seed=4242.
- Encoding: Path C tagged `[T:N prompt][T:M response]\xFE`. Vocab=41.
- Window=48, batch=64, epochs=3, reps=4, threads=12.
- Pairs de treino: 9644.

### Modelos

| | params | LR | tempo |
|---|---:|---:|---:|
| MLP | 1029161 | 0.064 | 4.8s |
| SMG-grid | 195625 | 0.1 | 255.0s |

## Loss curves

| Epoch | MLP loss | Grid loss |
|------:|---------:|----------:|
| 1 | 2.9972 | 2.9201 |
| 3 | 2.2904 | 2.8058 |

## Métricas finais

| | byte-agreement train | exact train | byte-agreement TEST | exact TEST |
|---|---:|---:|---:|---:|
| MLP | 31.6% | 0/80 | **27.8%** | 0/20 |
| SMG-grid | 19.1% | 0/80 | **20.2%** | 0/20 |

## Test set (prompts não vistos)

| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |
|--------|-------------|-----|----------|---------:|----------:|
| `quantos dedos tem uma mao` | `uma mao tem 5 dedos` | `a oa e a ea e a a a ea a e aa ea` | `a eeeeee eee ee e` | 25 | 15 |
| `o pinguim voa` | `o pinguim voa usando as suas` | `e a e e e ca a a e e e e e e e e eae…` | `a  a eeeeeeeeeeeeeee` | 34 | 25 |
| `a lua e uma estrela` | `nao a lua nao e uma estrela` | `a ca a a ea a a a a a e ea a aa` | `a  a a aa aa a aa aaaaaaaaa` | 20 | 18 |
| `qual e a capital do japao` | `a capital do japao e toquio` | `a aa aaa a a a iaa a e a a aa aa aa …` | `a a aa aaaaaaaa aaaaaaaaaaaaaaaa` | 31 | 25 |
| `a planta precisa de luz` | `sim a planta precisa de luz` | `a e a ca a a e a ea a ae a a aa aa a…` | `a a aa aaaaa aaa aaaaaaaaa aaa` | 32 | 23 |
| `qual e o rio mais longo do mundo` | `o rio mais longo do mundo e o` | `a e e eo ae e e ee e e e e e` | `eeee eee e eeee eeeeeeeee eeee ee` | 22 | 29 |
| `quantas patas tem uma formiga` | `uma formiga tem 6 patas` | `a caria e e ear e e a e ee a ee` | `a a aaaa aaaaaaa aaaaaaaaaaaaa` | 24 | 24 |
| `quantas patas tem uma aranha` | `uma aranha tem 8 patas` | `a caaita  aa aai a e a aaiaa a a aai…` | `a aaa aaaaaaaaaaaaaaaaaaaaaaaa` | 27 | 23 |
| `qual e o oposto de feliz` | `o oposto de feliz e triste` | `a o costo de a ra e e e e ea` | `eeeeeeeeeeeee` | 15 | 22 |
| `qual e a capital da turquia` | `a capital da turquia e ankara` | `a aaiaa a a raaa a e a a aaaa a a aa…` | `a aa aa aaaaaaa aaaaaaaaaaaaaaaaaaaa` | 29 | 26 |
| `quanto e 5 ao quadrado` | `5 ao quadrado e 25` | `o ca a a a e aa a a a e ea aa eaa` | `a a a ae aa e e a` | 25 | 11 |
| `o morcego voa` | `o morcego voa usando as suas` | `o co a a e o e e e e e e e e ee e e …` | `eeeeeeeeeeeeeeeeee` | 34 | 27 |
| `quantos meses tem o ano` | `o ano tem 12 meses` | `o e o car e e e o eera e e ea` | `eeeeeeeeeeeee e` | 20 | 15 |
| `quem foi dom pedro primeiro` | `dom pedro i tambem conhecido` | `o oo e o e e ra ao e eo ee e ee` | `eeeeee e` | 23 | 24 |
| `quanto e 7 vezes 8` | `7 vezes 8 e 56` | `o cao o e e e e e e e e e e e` | `eeeeeeeee` | 22 | 11 |
| `quem escreveu dom quixote` | `dom quixote foi escrito por` | `a e ca ae e a e eera e a e e` | `eeeeeee eeee eeeeeeeeeeee` | 22 | 25 |
| `quantos dias tem uma semana` | `um dia tem 24 horas uma hora` | `a ca ie a da a a e e aa aa e a aa aa…` | `a a aa aaaa aaa aaaa aaaaaaaaaaa` | 31 | 25 |
| `qual e a capital da italia` | `a capital da italia e roma` | `a capata  da aaaa a e aaraa ea a aaa…` | `a aaaa aaaaaaaaaaaaaaaaaaaaaaaaaaa` | 21 | 26 |
| `qual e a capital da bolivia` | `a capital da bolivia e la paz` | `a aaiaa a a raaa a e a a a ea` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 18 | 28 |
| `qual e o planeta mais proximo do sol` | `o planeta mais proximo do sol e` | `a ea ea e e e e e ee ae ee e ee` | `a a e  aa aa  aa aa` | 24 | 23 |

## Train set (sanity check, prompts vistos)

<details><summary>80 prompts (clicar para expandir)</summary>

| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |
|--------|-------------|-----|----------|---------:|----------:|
| `em que ano caiu o muro de berlim` | `o muro de berlim caiu em 9 de` | `a cari a e e ear ea e e ea ee` | `ee ee e e eee eeeeee eeeeeeeeeee` | 18 | 25 |
| `o que e arcoiris` | `arcoiris e um fenomeno natural` | `a ca e a e a eae e e a eara a e e ea` | `a eeeeeeeeeeeeeee` | 26 | 25 |
| `o que e vento` | `o vento e um movimento do ar na` | `o  o a ca e a  a co e a e a ea ae a …` | `eeeeeeeeeeeeeeeeeeeee` | 30 | 28 |
| `qual e o oceano maior` | `o oceano maior do mundo e o` | `a o e o e ea e e o e ea ee e e e e e…` | `a a eeeeeeee ee ee eeeee` | 29 | 22 |
| `quanto e 2 mais 2` | `2 mais 2 e 4` | `a ca e a e a cara a  a a aa` | `a a a a a` | 21 | 7 |
| `quem foi albert einstein` | `albert einstein foi um fisico` | `a o ca e a eai e e e e a e e e ee a …` | `a eeeeeeeeeee eeeeeeeeeeee` | 34 | 24 |
| `quanto e 1 mais 1` | `1 mais 1 e 2` | `a ca ca e a  a ca ae a a ea` | `a  a ee` | 21 | 8 |
| `quem pintou a mona lisa` | `a mona lisa foi pintada por` | `a ca a a aa a a aa aa e aa aa aa aaa…` | `a a a aa aaaaaaaa aaaaaaaaaa` | 32 | 21 |
| `qual e a cor do sangue` | `a cor do sangue e geralmente` | `a cao a e ea a aa a aa a aa aaaa aaa…` | `a a a eeeeeeee eeeeee eee e` | 32 | 19 |
| `qual e a capital da russia` | `a capital da russia e moscou` | `a capita  da rana a e e eaaa` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 11 | 29 |
| `qual e a capital do uruguai` | `a capital do uruguai e montevide` | `a aaiaaa a e raraa e a aaa aa` | `a aa aa aaaaa aaaaaaaaaaaaaaaaaaaaaa…` | 21 | 31 |
| `o tomate e uma fruta` | `nao o tomate nao e uma fruta` | `a ca e a aa a a aa a a aa aa aaaa a …` | `a a a ee a  eeee e eeeee` | 32 | 20 |
| `qual e o oposto de claro` | `o oposto de claro e escurto` | `a o cosao e e e ra ae e  eao` | `eeeeeeeeeeee` | 16 | 24 |
| `qual e a montanha mais alta do mundo` | `a montanha mais alta do mundo e` | `a eariaa a a ea aa e e aa eaa ea eaa…` | `a a aa a aaaa aaaaaaaaaaaaaaaaaaaaaa…` | 33 | 33 |
| `qual e a capital da alemanha` | `a capital da alemanha e berlim` | `a caaiaa  a iaa aa aa e a aaaaa a aa…` | `aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa…` | 28 | 30 |
| `quanto e 25 por cento de 200` | `25 por cento de 200 e igual a 50` | `a cori a e a eara a e e ea a ae a` | `eeeeeeeeeeeeeeee` | 23 | 29 |
| `o gato e um animal` | `sim o gato e um animal` | `a ca a a a a a a a a a aa a a eaa a …` | `a a a aa aaa aa a aa` | 29 | 15 |
| `qual e o maior planeta do sistema so…` | `o maior planeta do sistema` | `o aaia e a eera a e e ea a a ea` | `aa  a aaaaaa aaaaa aaaaaaaaaaaaaaa` | 21 | 28 |
| `o peixe respira no ar` | `o peixe respira no ar atraves` | `o co a aa a a a a aaraa e ea a a aaa…` | `a a aa eeaaa aa aaaaaa a aaa` | 33 | 22 |
| `em que ano colombo chegou na america` | `colombo chegou a america do sul` | `o aorao e ea e a ae` | `ea aaa a aaa aaaaa aaaaaaaaaaaaaaa` | 22 | 29 |
| `em que ano o brasil foi descoberto` | `o brasil foi descoberto pelo` | `o e aera e a e e eae e e e e e ea` | `e ee ee e` | 24 | 22 |
| `em que ano o homem pisou na lua` | `o homem pisou na lua em 1969` | `a eo e aa ee e a e e e e e e` | `eee ee ee eee e eeeeeeeeeeeeee` | 20 | 25 |
| `qual e a capital da china` | `a capital da china e pequim em` | `a aa a aa aa a aaa a eaa a a aa` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 20 | 30 |
| `quantas horas tem um dia` | `um dia tem 24 horas` | `a  a a a e a e e aa e a ea a ae a` | `a a a aaa aaaa aa aaa` | 26 | 16 |
| `qual e a capital da grecia` | `a capital da grecia e atenas` | `a capital da araa a e aaaaaa` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 8 | 27 |
| `qual e a capital da inglaterra` | `a capital da inglaterra e` | `a aa aa a aaa e e aaraa eaa aa aaa a…` | `aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 30 | 27 |
| `qual e o deserto maior do mundo` | `o deserto maior do mundo e o` | `a e ee a e a e e e e ea ee e` | `eeee eee e e e` | 21 | 21 |
| `quem descobriu a america` | `a descoberta da america pelo` | `a capa a  aa ea a a aaraaa a aaaaaa …` | `a a aaa aaaaaaaaaa aaaaaaaa` | 32 | 22 |
| `quanto e 144 dividido por 12` | `144 dividido por 12 e igual a 12` | `a cari a  do ear e e a eaa ea ee e a` | `eeeeee` | 24 | 31 |
| `o que e gelo` | `gel e um estado de conservacao` | `o a e e e e e o e e te e e e e e e e…` | `eeeeeeeeeeeeeeeeeeee` | 33 | 25 |
| `qual e o satelite natural da terra` | `o satelite natural da terra e a` | `e e ra ae e e e ae ae ee e ee` | `a a aaa  aaaaaaaaaaaaaaaaaaaaaaaaaaa…` | 23 | 31 |
| `a cobra tem patas` | `sim uma cobra tem patas` | `a e ca a aa a a a a a eara a aa aaaa…` | `a a aa aaaaaa aaaa  a` | 29 | 18 |
| `qual e a capital de portugal` | `a capital de portugal e lisboa` | `a caiia e da ao aa a a aa aa` | `a a a aaaa aa aaaaaaaaaaaaaaaaaaaaaa` | 17 | 28 |
| `qual e a capital do chile` | `a capital do chile e santiago` | `a aa aaa a a a iaa e e a aaaa aa aa …` | `a a a aa aaaa aaaaaaaaaaaaaaaaaaa` | 31 | 26 |
| `qual e a capital da argentina` | `a capital da argentina e buenos` | `a aaia a a aaaaa e a eaa aa aaaa a a…` | `aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa…` | 34 | 32 |
| `quem inventou o telefone` | `o telefone foi inventado por` | `o ee e e e e e e eere e ee e` | `eeeeeeeeeeeeeeeeeeeeeeeee` | 20 | 24 |
| `em que ano terminou a segunda guerra` | `a segunda guerra mundial` | `a eera a e e ea a ae a ee a eera` | `a a aaa a aaaaa aaaaaaaaaaaa` | 24 | 22 |
| `o sol e uma estrela` | `o sol e uma estrela muito` | `o co o a ea e a ea a ae a e a ea` | `a eeeeeeeeeee ee eeee` | 20 | 20 |
| `o que e o cerebro` | `o cerebro e o principal orgao` | `o cor e e e e e e e e e ee e` | `eeeeeeeeeeeeeeeeeeeeeee` | 20 | 26 |
| `o que e fogo` | `fogo e um processo fisico que` | `a c o e c e e e e e e e e ee ee ee e…` | `eeeeeeeeeeeeeeeeeee` | 35 | 26 |
| `qual e a capital da espanha` | `a capital da espanha e madrid` | `a aaia a a a iaa a e aa aa aa aaa aa…` | `a aaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 30 | 28 |
| `qual e o oposto de rapido` | `o oposto de rapido e lento` | `o oao to e a e rarae e e e` | `eeeeeeeeeeee` | 14 | 23 |
| `quanto e 9 vezes 9` | `9 vezes 9 e 81` | `a cao o e e e e e e e e e e e` | `eeeeeeeee` | 22 | 11 |
| `qual e a raiz quadrada de 81` | `a raiz quadrada de 81 e 9` | `a e aa aa ea ea e aa ea ea eaa e a e…` | `a aa a aaaa aa aaaaaaa aaaaaaaaa` | 27 | 24 |
| `em que ano caiu o imperio romano` | `o imperio romano caiu em 476 d` | `a ia e a e e eae e e ae e ea` | `e  ee ee eee eeeeee eeeeeee eeee` | 23 | 27 |
| `o que e gravidade` | `gravidade e a forca que nos` | `o ca a a a a ca a a a e e ea e a aaa…` | `a  eeeeeeeeeeeeee eeee` | 30 | 22 |
| `qual e a cor da neve` | `a cor da neve e geralmente` | `a ca a a o o ea a a a eae e a ea` | `a  a eeeeeeeeeee eeeeeee` | 21 | 17 |
| `qual e a capital do egito` | `a capital do egito e caire` | `a  aaita  da eriaa e e a e aa` | `a a a aa aaa  aaaaaa aaa aaaa` | 14 | 22 |
| `qual e a capital do paraguai` | `a capital do paraguai e asuncao` | `a caiiaa a a aa a e a e aaaa aa eaaa…` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa…` | 29 | 30 |
| `qual e a cor do ceu` | `a cor do ceu e uma questao de` | `a ca e a ca  e e e e e eae e` | `eeeeeeeeeeeeeeeeeeeeeeee` | 16 | 25 |
| `quantos planetas tem o sistema solar` | `o sistema solar tem 8 planetas` | `o e ra ee e e e eee` | `a aaa  aa aa a aaaaaaaaaaaaaaaaaaaaa…` | 20 | 31 |
| `quem foi isaac newton` | `isaac newton foi um dos principa` | `o a ca a a a e ra a a e e ea aea` | `a a a ea aaaa a aa a  aaa aaa` | 26 | 25 |
| `qual e a capital do brasil` | `a capital do brasil e brasilia` | `a a cata a a rara a e a eaaa` | `a a aa aaaaaaaa aaaaaaaaaaaaaaaaaaaa` | 19 | 28 |
| `em que ano o titanic afundou` | `o titanic afundou em 15 de` | `a cara a  a e a e e e e ea a ee` | `ee ee ee eeee e eeeeeeeee` | 22 | 23 |
| `qual e o nosso planeta` | `o nosso planeta e a terra` | `a ca eo e a e a e a e e ea` | `a  a eeeeeeeeeeee eeeeee` | 13 | 20 |
| `quem foi napoleao` | `napoleao bonaparte 17691821 foi` | `a ca e a  aa a e a e a aa ee a aa a …` | `a  a eeeeeeeeeeeeeeeeeeeee` | 31 | 27 |
| `qual e a cor do sol` | `a cor do sol e geralmente` | `a ca e a ca  e e a e a ea e ee` | `a eeeeeeeeeeeeeee` | 19 | 19 |
| `quantas patas tem um cachorro` | `um cachorro tem 4 patas` | `a caria a a e e e ea` | `a a a aaa a a aa aa aaaaaaa` | 16 | 21 |
| `o que e o coracao` | `o coracao e um orgao vital do` | `o cor e e e e e e e e e e ee` | `eeeeeeeeeeeeeeeeeeeeeee` | 18 | 28 |
| `quantos olhos tem um humano` | `um humano tem 2 olhos` | `o c o e e ea ee e a e e` | `eeee eeeee e ee e ee` | 19 | 19 |
| `em que ano nasceu pele` | `pele nasceu em 23 de outubro de` | `a ca a a e a ea ae e aa ea aa eaa ea…` | `eeeeeeeeeeeeeeeeeeeeeeeeeeee` | 36 | 25 |
| `o que e eletricidade` | `eletricidade e a corrente` | `e e ca a a e e e earae e e eaiae ea` | `eeeeeeeeeeeeeeeee eee` | 24 | 19 |
| `qual e o oposto de grande` | `o oposto de grande e pequeno` | `o o o to ee e e de e ee` | `eeeeeeeeeeee` | 12 | 23 |
| `qual e a cor da grama` | `a cor da grama e geralmente` | `a a a aa a a a a a aaaaa a aaaaa aa …` | `aa a aaaaaaaaaaaaaaaaaaaaaaaa` | 32 | 23 |
| `a baleia e um peixe` | `nao uma baleia nao e um peixe` | `a ca a a ea a a a aa a a aa aa eaa a…` | `a  a a eeeeee eeeeeee eeeee` | 32 | 19 |
| `quanto e 100 menos 37` | `100 menos 37 e igual a 63` | `a e o ea e e e e ea ae e e ee` | `eeeeeeeeeeeeeeeeeeeee` | 20 | 23 |
| `quem escreveu hamlet` | `hamlet e um romance historico` | `e ca e a ea e a ea e ae a eea` | `eeeeeeeeeeeee eeeeeeeee` | 24 | 25 |
| `o que e o sol` | `o sol e o centro do nosso` | `e  e a ca e a o o oa e a a e e ea e …` | `eeeeeeeeeeeeeeee` | 30 | 23 |
| `quantas asas tem uma borboleta` | `a resposta depende do tipo` | `a carioa e a rara a e e ra ae ea e a…` | `a aa aaaa a a aa a aaaaaaaaaaaa` | 29 | 27 |
| `quantas estacoes tem o ano` | `o ano tem 12 estacoes` | `a a o ta aoa e e ra a a aa e a ea a` | `a a aa a a e aa a  ea` | 25 | 15 |
| `em que ano comecou a segunda guerra` | `a segunda guerra mundial` | `a eari a eaa e a eae ea a` | `ee e ea e e eeee eeee eeeee` | 18 | 23 |
| `qual e a capital do canada` | `a capital do canada e ottawa` | `a capita  da eana a e a eaae` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 9 | 25 |
| `qual e o continente onde fica o bras…` | `o brasil esta localizado no` | `e e eria e a ea e ae ae ea e ee` | `ee e` | 22 | 25 |
| `qual e a capital do peru` | `a capital do peru e lima` | `a a a caiaa e a iaa e e aa aaa aa aa…` | `a a a a aaa aa aaaaaa aaaa` | 27 | 18 |
| `o que e agua` | `a agua e um composto quimico` | `o e a e a e e a e a ea e a e e e e e…` | `a eeeeeeeeeeeeeeee` | 33 | 25 |
| `quanto e 10 ao cubo` | `10 ao cubo e 10 x 10 x 10 que e` | `o co ao a a e o eo a e e e e e eee e…` | `eeeeeeeeeeeeeeeeeeeeeeeee` | 30 | 28 |
| `quem foi cleopatra` | `cleopatra e uma figura historica` | `o e ca a e a ea a a ra e a ea aaa aa…` | `a a a aee eeeeeeeeee ee eeee` | 33 | 27 |
| `qual e o oposto de quente` | `o oposto de quente e frio` | `o o oeto e e e rae e e e e` | `eeeeeeeeeeeee` | 13 | 21 |
| `qual e a capital da franca` | `a capital da franca e paris` | `a capata  da eara a e a eaaa` | `a aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 12 | 26 |
| `em que ano o brasil ganhou a primeir…` | `o brasil ganhou a primeira copa` | `o eeno e ea e a aa e ee e ee` | `aa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa…` | 22 | 32 |
</details>

## Veredicto

**✗ Estrutura piora** — SMG-grid perde 7.6pp para MLP em test. Possíveis causas: BPTT instável, K muito pequeno, ou tese 'byte-language gera estrutura' precisa ser repensada.

