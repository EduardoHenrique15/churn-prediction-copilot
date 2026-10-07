# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versões seguem [SemVer](https://semver.org/lang/pt-BR/).

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
