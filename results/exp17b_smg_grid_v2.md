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
| MLP | 1029161 | 0.064 | 267.6s |
| SMG-grid | 187433 | 0.1 | 2674.9s |

## Loss curves

| Epoch | MLP loss | Grid loss |
|------:|---------:|----------:|
| 1 | 2.1787 | 2.7255 |
| 5 | 0.0827 | 1.9871 |
| 10 | 0.0183 | 1.5755 |
| 15 | 0.0095 | 1.3168 |
| 20 | 0.0063 | 1.0341 |
| 25 | 0.0046 | 0.6738 |
| 30 | 0.0036 | 0.3275 |

## Métricas finais

| | byte-agreement train | exact train | byte-agreement TEST | exact TEST |
|---|---:|---:|---:|---:|
| MLP | 100.0% | 80/80 | **38.5%** | 0/20 |
| SMG-grid | 49.5% | 0/80 | **33.9%** | 0/20 |

## Test set (prompts não vistos)

| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |
|--------|-------------|-----|----------|---------:|----------:|
| `quantos dedos tem uma mao` | `uma mao tem 5 dedos` | `o daumtem 2o sora eo ce or` | `ua do mane e mao soceal` | 17 | 16 |
| `o pinguim voa` | `o pinguim voa usando as suas` | `o sol e o canto e forosto` | `ana gau a ma uma fossica fo om` | 20 | 23 |
| `a lua e uma estrela` | `nao a lua nao e uma estrela` | `a o putal de poral a ue euradil` | `a ala e ale ale mer a der` | 22 | 19 |
| `qual e a capital do japao` | `a capital do japao e toquio` | `a capital do egito e caire` | `a acapail a pao para e oma` | 10 | 15 |
| `a planta precisa de luz` | `sim a planta precisa de luz` | `a e a celet te eruan se` | `a pala era le barse lberasbola itela…` | 20 | 32 |
| `qual e o rio mais longo do mundo` | `o rio mais longo do mundo e o` | `a orisino auial ou elantr` | `arisa lam fois o umne gale` | 22 | 20 |
| `quantas patas tem uma formiga` | `uma formiga tem 6 patas` | `a coresa sentem esumtem` | `a maprarsa stam 4a paes` | 15 | 14 |
| `quantas patas tem uma aranha` | `uma aranha tem 8 patas` | `a m cecaprta a canceatama e manatage` | `a mapara sa e mapa ema mde` | 23 | 16 |
| `qual e o oposto de feliz` | `o oposto de feliz e triste` | `o oposto de clarl e escurto` | `o oposote el e fone o leme` | 11 | 14 |
| `qual e a capital da turquia` | `a capital da turquia e ankara` | `a capital da urugua a rm noma` | `a capital a deu ruma entendo` | 11 | 13 |
| `quanto e 5 ao quadrado` | `5 ao quadrado e 25` | `a cor ao nera f do arem e a` | `a opara do a dara 1 1da e que` | 17 | 18 |
| `o morcego voa` | `o morcego voa usando as suas` | `o sor nao o em o oitonoso bano ero` | `fo a gara garga cegarasa cassa` | 23 | 22 |
| `quantos meses tem o ano` | `o ano tem 12 meses` | `o o umum ememo sol eo` | `uma ne eso meno tara a celo e` | 14 | 21 |
| `quem foi dom pedro primeiro` | `dom pedro i tambem conhecido` | `o copito e o bera fonson` | `a opiso e guerma er aile e os` | 20 | 23 |
| `quanto e 7 vezes 8` | `7 vezes 8 e 56` | `9 vezes 9 e 81` | `a 2 el e e 2 es` | 4 | 10 |
| `quem escreveu dom quixote` | `dom quixote foi escrito por` | `o oeum ao certom nooris` | `a arcora ecare visitr do id` | 20 | 22 |
| `quantos dias tem uma semana` | `um dia tem 24 horas uma hora` | `a coumeno de a emandaado e n` | `u aim assneo mo som` | 25 | 19 |
| `qual e a capital da italia` | `a capital da italia e roma` | `a capital da frania e parisi` | `a capatial a da aine a emubrai` | 8 | 14 |
| `qual e a capital da bolivia` | `a capital da bolivia e la paz` | `a capital da espinai e a duma` | `a apatal ica braia e berisiua m` | 11 | 17 |
| `qual e o planeta mais proximo do sol` | `o planeta mais proximo do sol e` | `` | `a ira om orpeita de do oneguet` | 31 | 25 |

## Train set (sanity check, prompts vistos)

<details><summary>80 prompts (clicar para expandir)</summary>

| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |
|--------|-------------|-----|----------|---------:|----------:|
| `em que ano caiu o muro de berlim` | `o muro de berlim caiu em 9 de` | `o muro de berlim caiu em 9 de` | `a orsu e demo isli um e` | 0 | 19 |
| `o que e arcoiris` | `arcoiris e um fenomeno natural` | `arcoiris e um fenomeno natural` | `a carira e um enasina e natel` | 0 | 15 |
| `o que e vento` | `o vento e um movimento do ar na` | `o vento e um movimento do ar na` | `a vonve e tam vemivima da ra na` | 0 | 16 |
| `qual e o oceano maior` | `o oceano maior do mundo e o` | `o oceano maior do mundo e o` | `ao aconao e ara uma do ino ute` | 0 | 19 |
| `quanto e 2 mais 2` | `2 mais 2 e 4` | `2 mais 2 e 4` | `a 2a ma e 2 e2` | 0 | 7 |
| `quem foi albert einstein` | `albert einstein foi um fisico` | `albert einstein foi um fisico` | `a alberite neis efo ium fio` | 0 | 14 |
| `quanto e 1 mais 1` | `1 mais 1 e 2` | `1 mais 1 e 2` | `a 1a ma e 2 es` | 0 | 8 |
| `quem pintou a mona lisa` | `a mona lisa foi pintada por` | `a mona lisa foi pintada por` | `a ana ol apisa 1 18 11 8az` | 0 | 19 |
| `qual e a cor do sangue` | `a cor do sangue e geralmente` | `a cor do sangue e geralmente` | `a a cora nsa gue egare nete` | 0 | 13 |
| `qual e a capital da russia` | `a capital da russia e moscou` | `a capital da russia e moscou` | `a acail a parr use mensosi` | 0 | 17 |
| `qual e a capital do uruguai` | `a capital do uruguai e montevide` | `a capital do uruguai e montevide` | `a acapital dura ugue em eumnovde` | 0 | 14 |
| `o tomate e uma fruta` | `nao o tomate nao e uma fruta` | `nao o tomate nao e uma fruta` | `nao ma otena a e ar uma 1me` | 0 | 16 |
| `qual e o oposto de claro` | `o oposto de claro e escurto` | `o oposto de claro e escurto` | `o oposo ce el em orcensque e` | 0 | 14 |
| `qual e a montanha mais alta do mundo` | `a montanha mais alta do mundo e` | `a montanha mais alta do mundo e` | `a manta nham isa ma lume mnem` | 0 | 16 |
| `qual e a capital da alemanha` | `a capital da alemanha e berlim` | `a capital da alemanha e berlim` | `a apiala a danale ea berar` | 0 | 14 |
| `quanto e 25 por cento de 200` | `25 por cento de 200 e igual a 50` | `25 por cento de 200 e igual a 50` | `2 pero cento de 200 e igua a e` | 0 | 6 |
| `o gato e um animal` | `sim o gato e um animal` | `sim o gato e um animal` | `o gaima a e a nal e mas` | 0 | 17 |
| `qual e o maior planeta do sistema so…` | `o maior planeta do sistema` | `o maior planeta do sistema` | `o maior ponte al mesisto mo` | 0 | 11 |
| `o peixe respira no ar` | `o peixe respira no ar atraves` | `o peixe respira no ar atraves` | `aipeix era prana e a uar am12` | 0 | 18 |
| `em que ano colombo chegou na america` | `colombo chegou a america do sul` | `colombo chegou a america do sul` | `ocolbo ege geua ema emprad` | 0 | 19 |
| `em que ano o brasil foi descoberto` | `o brasil foi descoberto pelo` | `o brasil foi descoberto pelo` | `o brasil foi esce berto quem` | 0 | 7 |
| `em que ano o homem pisou na lua` | `o homem pisou na lua em 1969` | `o homem pisou na lua em 1969` | `o hame sasi um una ol em e9` | 0 | 14 |
| `qual e a capital da china` | `a capital da china e pequim em` | `a capital da china e pequim em` | `a acaita cal de na e enqe em mvis` | 0 | 17 |
| `quantas horas tem um dia` | `um dia tem 24 horas` | `um dia tem 24 horas` | `um ainta e m2 horas` | 0 | 7 |
| `qual e a capital da grecia` | `a capital da grecia e atenas` | `a capital da grecia e atenas` | `a acapita da grea gea e enas` | 0 | 8 |
| `qual e a capital da inglaterra` | `a capital da inglaterra e` | `a capital da inglaterra e` | `a capita da nagurai e ersaca` | 0 | 12 |
| `qual e o deserto maior do mundo` | `o deserto maior do mundo e o` | `o deserto maior do mundo e o` | `o esterima do e fure m o` | 0 | 15 |
| `quem descobriu a america` | `a descoberta da america pelo` | `a descoberta da america pelo` | `a caresum aicaima darra e` | 0 | 19 |
| `quanto e 144 dividido por 12` | `144 dividido por 12 e igual a 12` | `144 dividido por 12 e igual a 12` | `14 44ividid opro e2 e igua ua a2 a` | 0 | 13 |
| `o que e gelo` | `gel e um estado de conservacao` | `gel e um estado de conservacao` | `a fo geu a cer cesiscisua cque e` | 0 | 27 |
| `qual e o satelite natural da terra` | `o satelite natural da terra e a` | `o satelite natural da terra e a` | `a isa teu ntarita da re ura e me` | 0 | 17 |
| `a cobra tem patas` | `sim uma cobra tem patas` | `sim uma cobra tem patas` | `isa ma cau a peta te as` | 0 | 13 |
| `qual e a capital de portugal` | `a capital de portugal e lisboa` | `a capital de portugal e lisboa` | `a apiata l egrate gaure mibando` | 0 | 18 |
| `qual e a capital do chile` | `a capital do chile e santiago` | `a capital do chile e santiago` | `a acapital do e ncea e etema` | 0 | 13 |
| `qual e a capital da argentina` | `a capital da argentina e buenos` | `a capital da argentina e buenos` | `a apital a gerane ne bere b` | 0 | 14 |
| `quem inventou o telefone` | `o telefone foi inventado por` | `o telefone foi inventado por` | `a olenete fori vindeo 2 do` | 0 | 16 |
| `em que ano terminou a segunda guerra` | `a segunda guerra mundial` | `a segunda guerra mundial` | `a esgunes fugraum deisl` | 0 | 12 |
| `o sol e uma estrela` | `o sol e uma estrela muito` | `o sol e uma estrela muito` | `o naso ua e emra esalu e` | 0 | 16 |
| `o que e o cerebro` | `o cerebro e o principal orgao` | `o cerebro e o principal orgao` | `o cerale e bror e pinal e galare` | 0 | 18 |
| `o que e fogo` | `fogo e um processo fisico que` | `fogo e um processo fisico que` | `agal e a prace scopissica qua` | 0 | 14 |
| `qual e a capital da espanha` | `a capital da espanha e madrid` | `a capital da espanha e madrid` | `a apiala pa e sana pae mbrrai` | 0 | 13 |
| `qual e o oposto de rapido` | `o oposto de rapido e lento` | `o oposto de rapido e lento` | `o oposo te er pida or enes` | 0 | 10 |
| `quanto e 9 vezes 9` | `9 vezes 9 e 81` | `9 vezes 9 e 81` | `a veze e 2 e 81` | 0 | 4 |
| `qual e a raiz quadrada de 81` | `a raiz quadrada de 81 e 9` | `a raiz quadrada de 81 e 9` | `a ariza quadrad e1 8 e e` | 0 | 8 |
| `em que ano caiu o imperio romano` | `o imperio romano caiu em 476 d` | `o imperio romano caiu em 476 d` | `o ime prirao mnao imu e fomo` | 0 | 18 |
| `o que e gravidade` | `gravidade e a forca que nos` | `gravidade e a forca que nos` | `agraviada e a fera qua naos` | 0 | 7 |
| `qual e a cor da neve` | `a cor da neve e geralmente` | `a cor da neve e geralmente` | `a cara da ne a eger ane em` | 0 | 11 |
| `qual e a capital do egito` | `a capital do egito e caire` | `a capital do egito e caire` | `a acapial ta degiso e e emuai` | 0 | 13 |
| `qual e a capital do paraguai` | `a capital do paraguai e asuncao` | `a capital do paraguai e asuncao` | `a apaita pal asirga emairundo` | 0 | 17 |
| `qual e a cor do ceu` | `a cor do ceu e uma questao de` | `a cor do ceu e uma questao de` | `a acra co de ue gua erma cele` | 0 | 17 |
| `quantos planetas tem o sistema solar` | `o sistema solar tem 8 planetas` | `o sistema solar tem 8 planetas` | `iso sisate omar tos um emnos` | 0 | 17 |
| `quem foi isaac newton` | `isaac newton foi um dos principa` | `isaac newton foi um dos principa` | `a aisa nano ena fo em 11 fois` | 0 | 23 |
| `qual e a capital do brasil` | `a capital do brasil e brasilia` | `a capital do brasil e brasilia` | `a a pail sta brles isaisl ie` | 0 | 19 |
| `em que ano o titanic afundou` | `o titanic afundou em 15 de` | `o titanic afundou em 15 de` | `a itainta  fundou em 1ber` | 0 | 10 |
| `qual e o nosso planeta` | `o nosso planeta e a terra` | `o nosso planeta e a terra` | `o onsa opnale e a e terra` | 0 | 10 |
| `quem foi napoleao` | `napoleao bonaparte 17691821 foi` | `napoleao bonaparte 17691821 foi` | `anapolao apana e71666611121` | 0 | 18 |
| `qual e a cor do sol` | `a cor do sol e geralmente` | `a cor do sol e geralmente` | `a caro  do le gera em escenla` | 0 | 14 |
| `quantas patas tem um cachorro` | `um cachorro tem 4 patas` | `um cachorro tem 4 patas` | `a cacorchora pte sata d ge8gura ete` | 0 | 24 |
| `o que e o coracao` | `o coracao e um orgao vital do` | `o coracao e um orgao vital do` | `o carao cara e ugara vaita do` | 0 | 12 |
| `quantos olhos tem um humano` | `um humano tem 2 olhos` | `um humano tem 2 olhos` | `um haim uma em2 o hos` | 0 | 9 |
| `em que ano nasceu pele` | `pele nasceu em 23 de outubro de` | `pele nasceu em 23 de outubro de` | `aplea ena e uma 2 1debre` | 0 | 19 |
| `o que e eletricidade` | `eletricidade e a corrente` | `eletricidade e a corrente` | `eletricaide e ara cere nteso` | 0 | 10 |
| `qual e o oposto de grande` | `o oposto de grande e pequeno` | `o oposto de grande e pequeno` | `o oposto de gerande e equo` | 0 | 4 |
| `qual e a cor da grama` | `a cor da grama e geralmente` | `a cor da grama e geralmente` | `a car da gragara e e manten` | 0 | 10 |
| `a baleia e um peixe` | `nao uma baleia nao e um peixe` | `nao uma baleia nao e um peixe` | `ana al acea eana e e ma pe` | 0 | 17 |
| `quanto e 100 menos 37` | `100 menos 37 e igual a 63` | `100 menos 37 e igual a 63` | `100 ano es en7 igail a 63` | 0 | 9 |
| `quem escreveu hamlet` | `hamlet e um romance historico` | `hamlet e um romance historico` | `alha ete ra ceuma esciisto uto` | 0 | 18 |
| `o que e o sol` | `o sol e o centro do nosso` | `o sol e o centro do nosso` | `foo no o per onco penosisa gura e ge…` | 0 | 26 |
| `quantas asas tem uma borboleta` | `a resposta depende do tipo` | `a resposta depende do tipo` | `a raspesa tem2 de8 pisti da e` | 0 | 16 |
| `quantas estacoes tem o ano` | `o ano tem 12 estacoes` | `o ano tem 12 estacoes` | `a ona e 1m 2estascoe` | 0 | 9 |
| `em que ano comecou a segunda guerra` | `a segunda guerra mundial` | `a segunda guerra mundial` | `a esgesu agura eum deris` | 0 | 15 |
| `qual e a capital do canada` | `a capital do canada e ottawa` | `a capital do canada e ottawa` | `a acapia ca da asa egaro ma enda` | 0 | 17 |
| `qual e o continente onde fica o bras…` | `o brasil esta localizado no` | `o brasil esta localizado no` | `o brasis e tarl caizzzes` | 0 | 11 |
| `qual e a capital do peru` | `a capital do peru e lima` | `a capital do peru e lima` | `a acapial a pe ura e guaem eim` | 0 | 15 |
| `o que e agua` | `a agua e um composto quimico` | `a agua e um composto quimico` | `a gala e um acapo scaisa qua a` | 0 | 17 |
| `quanto e 10 ao cubo` | `10 ao cubo e 10 x 10 x 10 que e` | `10 ao cubo e 10 x 10 x 10 que e` | `ara o cao u ber a isca ou e que` | 0 | 21 |
| `quem foi cleopatra` | `cleopatra e uma figura historica` | `cleopatra e uma figura historica` | `alacao ea para ua 1 1a1 asio` | 0 | 24 |
| `qual e o oposto de quente` | `o oposto de quente e frio` | `o oposto de quente e frio` | `o oposto e que ne em feri` | 0 | 6 |
| `qual e a capital da franca` | `a capital da franca e paris` | `a capital da franca e paris` | `a paicapal a fras e pana es` | 0 | 13 |
| `em que ano o brasil ganhou a primeir…` | `o brasil ganhou a primeira copa` | `o brasil ganhou a primeira copa` | `a brasisl a uagua era epral` | 0 | 17 |
</details>

## Veredicto

**~ Estrutura é neutra** — diferença em test = -4.6pp, dentro do ruído. Estrutura por si só não generaliza melhor; valor virá da hierarquia + bus (exp18-19) ou é necessário rever a tese.

