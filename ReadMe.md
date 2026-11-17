# 🏍️ RotaViva — Previsão de risco de sinistros com motos em São Paulo

**Global Solution 2026.2 · FIAP · CitizenTech · Inteligência Artificial & Machine Learning**

O RotaViva usa dados públicos para prever **quando e onde o risco de sinistros com motos é maior** em São Paulo, para alertar o motociclista e ajudar os órgãos públicos a se prepararem antes.

---

## 👥 Participantes

| Nome | RM |
|---|---|
| Guilherme Flores Pereira de Almeida | RM554948 |
| Luiz Fernando de Aragão Souza | RM555561 |
| Bruno Otavio Silva De Oliveira | RM556196 |
| Marcello de Freitas Moreira | RM557531 |
| Leonardo Gonçalves Novaes | RM554807 |

---

## 1. Problema e dados reais

### O problema
São Paulo registrou **379 mortes de motociclistas entre janeiro e setembro de 2026**, o maior número para o período desde o início da série do Infosiga, em 2015. Os motociclistas foram cerca de **48% dos mortos no trânsito** da capital no mesmo período.

Hoje, nem o motociclista nem os órgãos de atendimento sabem **com antecedência** quando e onde o risco será maior.

### Pergunta de Machine Learning
> **Dado a zona da cidade, o dia, a faixa de 3 horas e a chuva, quantos acidentes com moto sao esperados?**

A previsão vira um **nível de risco** (🟢 Baixo, 🟡 Moderado, 🟠 Alto ou 🔴 Crítico) que:
- **alerta o motociclista** em um app de navegação (ex.: "reduza a velocidade e evite o corredor");
- **orienta PM, CET, SAMU e Bombeiros** sobre os horários e zonas em que vale reforçar patrulhas e equipes.

**ODS relacionados:** 3 (meta 3.6, reduzir mortes no trânsito), 11 (meta 11.2, transporte seguro) e 13 (adaptação a eventos climáticos).

### Dados reais e públicos

| Base | Fonte | Link | Período usado |
|---|---|---|---|
| Sinistros de trânsito | Infosiga SP (Detran-SP) | https://infosiga.detran.sp.gov.br/ | 01/01/2019 a 30/09/2026 |
| Chuva e temperatura horárias | INMET, estação A701 (São Paulo – Mirante) | https://portal.inmet.gov.br/dadoshistoricos | 01/01/2019 a 31/10/2026 |


### Preparação dos dados
1. Dos **343.241** registros da capital, mantivemos só os **sinistros confirmados** (fatais e não fatais). Os registros `NOTIFICACAO` são avisos ainda não confirmados.
2. Usamos dados a partir de **2019**, porque antes disso o Infosiga só registrava sinistros fatais.
3. Ficaram os sinistros com **pelo menos uma moto** envolvida (134.338).
4. Removemos registros sem hora ou com coordenadas fora da cidade, chegando a **129.069 sinistros** (perda de 4%).
5. Dividimos a cidade em **5 zonas** a partir do Marco Zero (Praça da Sé): o raio de 5 km forma o Centro, e a direção define Norte, Sul, Leste ou Oeste.
6. Montamos a base de treino com **uma linha por zona × dia × faixa de 3 horas** (113.200 linhas), incluindo os períodos **sem nenhum sinistro** (40,9%), para o modelo aprender como é um período tranquilo.

**Variáveis de entrada:** todas são conhecidas antes do sinistro.

| Grupo | Variáveis |
|---|---|
| Local e horário | zona (one-hot), faixa de 3 horas |
| Calendário | dia da semana, mês, feriado (nacional, estadual e municipal) |
| Clima (INMET) | chuva na faixa, chuva na faixa anterior, temperatura média |
| Histórico | `historico_30d`: média de sinistros da mesma zona e faixa entre 60 e 31 dias antes |

A defasagem do histórico é proposital. Como o Infosiga é atualizado mensalmente, a variável só usa dados que já estariam publicados no momento da previsão.

**Divisão treino/teste (temporal):** treino em **2019–2024** (87.680 linhas) e teste em **jan/2025–set/2026** (25.520 linhas). Não usamos divisão aleatória porque ela colocaria dias do futuro no treino.

---

## 2. Escolha justificada do modelo

Escolhemos o **XGBoost** com objetivo de **regressão Poisson**:

- **Dados em tabela** com variáveis numéricas e categóricas, que é o cenário em que o XGBoost costuma ter melhor desempenho.
- **Interações não lineares.** A análise exploratória mostrou que **chuva fraca no pico da manhã aumenta os sinistros** (às 7h, a média sobe de 3,0 para 3,4 por hora), enquanto **chuva moderada ou forte reduz**, provavelmente porque menos motos circulam. Árvores de decisão capturam esse tipo de combinação.
- **Valores faltantes** (horas sem medição do INMET) são tratados nativamente, sem precisar inventar valores.
- **O alvo é uma contagem** (0, 1, 2, 3...) com muitos zeros, que é o caso para o qual o objetivo Poisson foi feito.

**Baseline de comparação:** a média histórica de sinistros por zona + faixa + fim de semana. O modelo só agrega valor se superar essa regra simples.

### Treino com diferentes hiperparâmetros
Usamos `GridSearchCV` com **12 combinações de 3 hiperparâmetros**:

| Hiperparâmetro | Valores testados | O que controla |
|---|---|---|
| `max_depth` | 3, 5, 7 | profundidade (complexidade) das árvores |
| `learning_rate` | 0.05, 0.1 | tamanho do passo de aprendizado |
| `n_estimators` | 200, 500 | quantidade de árvores |

A validação cruzada foi feita **só no treino**, com `TimeSeriesSplit` (3 divisões cronológicas, sempre validando num período posterior ao de treino). A métrica foi o MAE.

| max_depth | learning_rate | n_estimators | MAE treino | MAE validação | diferença |
|---|---|---|---|---|---|
| **5** | **0.10** | **200** | **0,776** | **0,839** | **0,063** |
| 5 | 0.05 | 500 | 0,771 | 0,840 | 0,069 |
| 3 | 0.10 | 500 | 0,788 | 0,840 | 0,052 |
| 7 | 0.10 | 200 | 0,735 | 0,843 | 0,108 |
| 7 | 0.05 | 500 | 0,723 | 0,844 | 0,121 |
| 7 | 0.10 | 500 | 0,675 | 0,851 | **0,176** |
| 3 | 0.05 | 200 | 0,813 | 0,854 | 0,040 |

*(As 12 combinações completas estão no notebook)*



**Discussão sobre overfitting:**
- **Overfitting:** com árvores profundas e muitas iterações, o modelo teve o **menor erro de treino (0,675)** e o **pior erro de validação (0,851)**. Ele decorou o treino.
- **Underfitting:** com `max_depth=3`, `learning_rate=0.05` e 200 árvores, o modelo ficou simples demais e aprendeu pouco.
- **Escolhida:** `max_depth=5`, `learning_rate=0.1` e `n_estimators=200`, com o **menor erro de validação** e uma diferença moderada entre treino e validação.

---

## 3. Resultados em treino e teste

Fizemos **dois experimentos**:
- **Experimento 1:** calendário com `ano` e `fim_de_semana`, chuva, temperatura e zona.
- **Experimento 2:** troca `ano` e `fim_de_semana` pelo **histórico recente** (`historico_30d`).

| Métrica | Baseline (teste) | Exp. 1 – treino | Exp. 1 – teste | **Exp. 2 – treino** | **Exp. 2 – teste** |
|---|---|---|---|---|---|
| MAE | 0,822 | 0,814 | 0,827 | **0,817** | **0,805** |
| RMSE | 1,085 | 1,089 | 1,082 | **1,093** | **1,069** |
| R² | 0,252 | 0,334 | 0,255 | **0,330** | **0,273** |


**Leitura dos resultados:**
- **Experimento 1:** empatou com o baseline e **superestimou o teste em ~15%**. Os sinistros não fatais confirmados caíram cerca de 12% em 2025, e o modelo só conhecia o patamar até 2024. A variável `ano` não resolve isso, porque árvores não extrapolam para anos que nunca viram.
- **Experimento 2 (modelo final):** com o histórico recente, o modelo acompanhou a mudança de patamar e **superou o baseline em todas as métricas**. O MAE de teste (0,805) ficou próximo do de treino (0,817), então **não há sinal de overfitting**. A queda do R² de 0,330 para 0,273 é esperada num período futuro e desconhecido.
- **Limitação:** o modelo segue a tendência, mas **suaviza as quedas bruscas** em férias de fim de ano e feriados prolongados.


### Níveis de risco validados no teste
Os cortes foram definidos pelos quantis de 40%, 70% e 90% das previsões de treino:

| Nível previsto | Períodos | Média **real** de sinistros | Média prevista |
|---|---|---|---|
| Baixo | 10.333 | 0,43 | 0,45 |
| Moderado | 7.817 | 1,12 | 1,17 |
| Alto | 5.200 | 1,78 | 1,88 |
| Crítico | 2.170 | 2,29 | 2,51 |

➡️ Os períodos classificados como **Crítico tiveram 5,3 vezes mais sinistros reais** que os classificados como Baixo, em dados que o modelo nunca viu.

---

## 4. Conclusão


**🏍️ Para o motociclista**
- Os maiores riscos estão nos horários de deslocamento ao trabalho: **06–09h e 18–21h**, principalmente nas **zonas Sul e Leste**.
- **Chuva fraca no pico da manhã aumenta o risco.** Nesses cenários, o RotaViva mostra: *"🔴 Risco Crítico: reduza a velocidade e evite o corredor"*.

**🌙 A madrugada engana**

De madrugada acontecem poucos sinistros, mas **entre 2h e 4h, 12–13% dos sinistros com moto têm morte**, contra 1,5% durante o dia, quase **8 vezes mais**. O alerta de madrugada deve falar de **gravidade**, não de quantidade.

**🚓 Para os órgãos públicos (PM, CET, SAMU, Bombeiros)**

O ranking zona × faixa com nível Crítico indica **onde e quando concentrar patrulhas e deixar equipes de prontidão**. Exemplo: **sexta-feira, 18–21h, zona Sul**.

**📌 Em resumo**

O modelo não prevê o acidente individual (R² de 0,27). O que ele faz bem é **classificar o nível de risco de cada período**, e é isso que permite priorizar alertas e recursos públicos onde eles salvam mais vidas.

### Limitações e próximos passos
- O Infosiga registra só sinistros oficialmente notificados.
- A chuva vem de **uma única estação** (Mirante, zona norte). Mais estações por zona devem melhorar o efeito climático.
- As zonas são uma aproximação geométrica das zonas oficiais e têm áreas diferentes.
- No uso real, a chuva medida seria substituída pela **previsão do tempo**.

---

## ▶️ Como executar

```bash
pip install -r ml/requirements.txt
```
1. Baixe as bases
2. Abra [`ml/main.ipynb`](ml/main.ipynb) e execute todas as células.



**Fontes:** Infosiga SP / Detran-SP; INMET; [BNews – Mortes de motociclistas batem recorde nos primeiros nove meses em SP (20/10/2026)](https://www.bnewssaopaulo.com.br/amp/noticias/policia/mortes-de-motociclistas-batem-recorde-nos-primeiros-nove-meses-em-sp.html); [Correio da Manhã – Mortes no trânsito crescem em SP; motos somam 48% (out/2026)](https://www.correiodamanha.com.br/nacional/sao-paulo/2026/10/amp/328793-mortes-no-transito-crescem-em-sp-motos-somam-48-dos-obitos.html).
