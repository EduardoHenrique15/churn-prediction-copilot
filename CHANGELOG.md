# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versões seguem [SemVer](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Publicado
- API no Render (`churn-radar-api.onrender.com`) e interface no Streamlit Community Cloud (`churn-radar-br.streamlit.app`), com os links no README.

### Corrigido
- **Rate limit:** o IP vem do último endereço do `X-Forwarded-For` (o que o proxy anexa); trocar o cabeçalho não dribla mais o limite.
- **Base do RAG:** a `chroma_db/` não é mais versionada nem copiada na imagem — o assistente a monta na primeira busca. A reindexação calcula os embeddings antes de tocar no disco e troca a base de uma vez: uma falha (cota, rede) não apaga a base antiga.
- **Interface:** rótulos e subtítulos saíam com 15 px (o estilo do Streamlit vencia o do tema) e as seções ficavam coladas; tabela da Estratégia cortada; títulos gigantes nas respostas do assistente; formulário da página Cliente com a parte de Cobrança inalcançável enquanto fixo; `/inicio` abria "Page not found".
- **Cliente da API:** 429 e 500 (API acordada recusando o pedido) não aparecem mais como "API acordando"; carregar o modelo local não trava o chip de status.
- `fmt_pp` mostrava "+0,0 pp" para diferenças que arredondam a zero; a sigmoide da explicação estourava para log-odds muito negativos.

### Mudado
- **Carteira:** as hipóteses de custo ajustadas na Estratégia valem também ali (corte, fila e valor esperado), numa faixa de status única.
- **Assistente:** a ferramenta de previsão lista os valores permitidos de cada campo (enum no schema do Gemini); a última rodada não executa ferramenta cujo resultado seria descartado; o contador de perguntas atualiza na hora e a pergunta bloqueada pelo limite vira aviso; o dia da cota é contado no fuso do Pacífico, como o do Gemini.
- **Textos derivados do treino:** nomes dos modelos num lugar só (`src/labels.py`) e números como "28 pesos", "7.032 clientes" e o churn do contrato bienal saem de `metrics.json`/`evaluation.json`.
- `redundancy_stats` (a evidência para tirar mensalidade e total gasto do modelo) é medida só no treino.
- `record_demo` salva a cada resposta e retoma de onde parou se a cota acabar.
- Revisão de estética: radar no topo da Visão geral, tabelas que viram cartões no celular, faixa de risco colorida na fila de contato, capturas e GIF do README refeitos.

### Adicionado
- CI: build e healthcheck da imagem da interface, permissões mínimas (`contents: read`) e logs do container quando o teste da API falha.
- Testes: de 263 para 287 (erros 500/503 da API, 429 no log, cliente com API ocupada × dormindo, reindexação do RAG, hipóteses compartilhadas, schema da ferramenta do agente).

## [2.0.0] — 2026-09-26

### Corrigido
- **Train/serve skew no encoding:** com um cliente só, o `get_dummies` descartava todas as categorias e todo cliente virava o de referência (contrato mensal, DSL, transferência). Mudar contrato, internet ou pagamento não mudava a previsão, e o resultado de um cliente dependia dos outros do lote. Agora o encoding usa categorias fixas e testes de regressão travam o comportamento.
- Combinações impossíveis (sem internet, mas com suporte técnico; total gasto incompatível com tempo × mensalidade) são recusadas com 422 e mensagem em português.
- Lote: uma linha inválida derrubava o arquivo inteiro; agora a validação é linha a linha, com relatório e download das linhas com problema. CSV do Kaggle com total gasto vazio (clientes novos) é aceito.
- Reenviar um CSV já pontuado não duplica mais as colunas de resultado.
- Visitantes não veem mais traceback (`showErrorDetails = "type"`) nem instruções de desenvolvimento.
- Texto em R$ no Markdown não vira mais fórmula (cifrão escapado).

### Mudado
- **Modelo:** sem reponderação de classes (probabilidades calibradas: ECE de 0,117 para 0,031); escolha pela regra de 1 erro-padrão com validação cruzada 5 × 3 e erro-padrão corrigido; regressão logística no lugar do Random Forest.
- **Variáveis:** mensalidade e total gasto saem do modelo (colineares com serviços e tempo de casa); a API continua recebendo e validando os campos.
- **Decisão:** corte de contato escolhido por valor esperado nas previsões fora da amostra do treino (0,32); faixas de risco derivadas da mesma conta.
- **Avaliação:** intervalos de confiança por bootstrap, calibração, métricas por subgrupo e razões de chances em unidades de negócio.
- **Agente:** uma chamada ao modelo por rodada (antes, a resposta final era pedida de novo), streaming real, mensagens de erro amigáveis, limites por sessão e por dia e eval de roteamento com metade do custo de cota.
- **RAG:** documentos reescritos com números conferidos; os documentos do modelo e do guia de leitura são gerados pelo treino; trechos cortados por seção, com contexto.
- **Dependências:** `requirements-api.txt` (API), `requirements.txt` (interface e agente), `requirements-dev.txt`.
- **Docker:** CMD em forma exec (desligamento limpo), imagem da API sem a interface, `.dockerignore` com padrões recursivos.

### Adicionado
- **Interface nova** (6 páginas, navegação no topo): visão geral, análise de cliente com explicação e cenários, carteira com fila de contato e monitor de drift (PSI), simulador de estratégia, model card e assistente. Gráficos em HTML/SVG legíveis no celular; fontes servidas pelo próprio app.
- **Plano B local:** enquanto a API do Render hiberna, a interface calcula a mesma previsão localmente e acorda a API em segundo plano.
- API: limite de requisições por IP, log estruturado, tempo de resposta no cabeçalho, `/health` com versão, modelo e data do treino.
- Respostas gravadas do assistente (`python -m src.record_demo`), exemplos de carteira gerados do conjunto de teste (`python -m src.make_examples`).
- Testes: de 63 para 263 (interface com `AppTest`, agente com modelo falso, cliente, validação, treino, empacotamento); cobertura no CI; build da imagem da API no CI com medição de memória; `pre-commit`.
- Documentação: registros de decisão (`docs/decisoes/`), guia de deploy, README em inglês, notebook de EDA refeito e executado.

### Removido
- `data/processed_churn.csv` (não era usado pelo treino).
- Heurística de "fatores de risco" escrita à mão (substituída pela explicação exata do modelo).

## [1.0.0] — 2026-09-24

Primeira versão publicada: comparação de 4 modelos, API FastAPI, agente com RAG e interface Streamlit com dashboard, simulador de ROI, model card e chat.
