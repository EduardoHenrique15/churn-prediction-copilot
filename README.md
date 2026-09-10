# 🔮 Churn Prediction Copilot

Sistema de previsão de cancelamento de clientes (churn) para uma empresa de telecomunicações, com modelo de Machine Learning servido via API REST, rastreamento completo de experimentos, e um agente de IA conversacional com RAG e function calling.

## 📌 Problema de negócio

Empresas de assinatura (telecom, SaaS, streaming) enfrentam um desafio constante: identificar **quais clientes estão prestes a cancelar** antes que isso aconteça, para que times de retenção possam agir preventivamente (ofertas, contato proativo, ajuste de plano).

Este projeto treina um modelo de classificação que prevê a probabilidade de um cliente cancelar o serviço, disponibiliza essa previsão via API, e oferece um **agente de IA** que explica os resultados em linguagem natural, combinando a previsão do modelo com uma base de conhecimento sobre churn.

## 🗂️ Dataset

**Telco Customer Churn** ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) — 7.043 clientes de uma empresa de telecomunicações, com informações sobre:
- Dados demográficos (gênero, se é idoso, parceiro, dependentes)
- Serviços contratados (telefone, internet, segurança online, streaming, etc.)
- Dados de conta (tempo de contrato, tipo de contrato, forma de pagamento, cobrança)
- Variável alvo: `Churn` (cliente cancelou ou não)

## 🔍 Principais insights da análise exploratória

- **Dataset desbalanceado**: 73,4% dos clientes não cancelam vs. 26,6% que cancelam — por isso o modelo usa `class_weight="balanced"` e a avaliação prioriza recall/F1 em vez de acurácia pura.
- **Tempo de contrato (`tenure`) é o preditor mais forte**: clientes que cancelam têm mediana de ~10 meses de casa, contra ~38 meses dos que permanecem — ou seja, o risco de churn é maior nos primeiros meses do contrato.

## 🏗️ Arquitetura

```
Dados brutos (CSV)
        ↓
Pré-processamento (utils.py) → limpeza + one-hot encoding
        ↓
Treino (train.py) → RandomForestClassifier + tracking no MLflow
        ↓
Modelo serializado (.pkl)
        ↓
API REST (api.py, FastAPI) → endpoint /predict
        ↓
Containerização (Docker) → aplicação portável e reprodutível
        ↓
Agente de IA (agent.py, LangChain + Gemini)
        ├── Function calling → chama a API /predict
        └── RAG (ChromaDB) → busca contexto na base de conhecimento
```

## 📊 Resultados do modelo

| Métrica | Valor |
|---|---|
| Acurácia | 0.75 |
| Precision (classe "Churn") | 0.52 |
| Recall (classe "Churn") | 0.77 |
| F1-score | 0.62 |

**Por que priorizar recall em vez de acurácia:** o custo de não identificar um cliente que vai cancelar (falso negativo) é maior para o negócio do que contatar preventivamente um cliente que não ia cancelar (falso positivo). Por isso o modelo foi ajustado para maximizar a detecção de clientes em risco, mesmo aceitando mais falsos positivos.

## 🤖 Agente de IA (RAG + Function Calling)

Além da API de previsão, o projeto conta com um agente conversacional que combina duas capacidades:

1. **Function calling**: quando o usuário fornece os dados de um cliente específico, o agente chama a API `/predict` automaticamente e traduz o resultado em linguagem natural, explicando os fatores de risco.
2. **RAG (Retrieval-Augmented Generation)**: para perguntas conceituais sobre churn (o que é, fatores de risco, recomendações de retenção), o agente busca contexto numa base de conhecimento vetorizada (ChromaDB) antes de responder.

### Stack do agente
- **LLM:** Gemini (via `langchain-google-genai`)
- **Orquestração:** LangChain
- **Banco vetorial:** ChromaDB
- **Embeddings:** `gemini-embedding-001`

### Exemplo de interação

```
Você: Um cliente com tenure de 3 meses, contrato mensal, sem serviços de
segurança, cobrança de 95/mês. Qual o risco de cancelamento?

Agente: Com base nas informações fornecidas, o cliente apresenta um alto
risco de cancelamento (55,89%).

Fatores que contribuem para este risco:
1. Tempo de contrato curto (tenure de 3 meses)
2. Contrato mensal, sem fidelidade
3. Ausência de serviços de proteção (segurança online, suporte técnico)
4. Forma de pagamento associada a maior índice de churn

Recomendações de retenção:
- Oferecer desconto para migração para contrato anual
- Incluir suporte técnico gratuito por período de teste
```

## 🛠️ Stack técnica

- **Linguagem:** Python 3.11
- **Análise e modelagem:** Pandas, Scikit-learn
- **Rastreamento de experimentos:** MLflow
- **API:** FastAPI + Uvicorn
- **Containerização:** Docker
- **Agente de IA:** LangChain + Gemini API
- **Banco vetorial (RAG):** ChromaDB

## 🚀 Como rodar o projeto

### Pré-requisitos
- Python 3.11+
- Docker Desktop (opcional, para rodar via container)
- Chave de API do Gemini ([Google AI Studio](https://aistudio.google.com))

### Configuração inicial

```bash
# Clone o repositório
git clone https://github.com/EduardoHenrique15/churn-prediction-copilot.git
cd churn-prediction-copilot

# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instale as dependências
pip install -r requirements.txt

# Crie um arquivo .env na raiz com sua chave do Gemini
# GEMINI_API_KEY=sua_chave_aqui
```

### 1. Treinar o modelo

```bash
cd src
python train.py
```

### 2. Rodar a API

```bash
uvicorn api:app --reload
```

Acesse a documentação interativa em `http://127.0.0.1:8000/docs`

### 3. Rodando via Docker (alternativa aos passos 1-2)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### 4. Construir a base de conhecimento do agente (RAG)

```bash
python src/build_knowledge_base.py
```

### 5. Conversar com o agente

Com a API rodando (passo 2 ou 3) em um terminal, abra outro terminal:

```bash
cd src
python chat_test.py
```

### Visualizando os experimentos no MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse `http://localhost:5000`

## 📡 Exemplo de uso da API

**Requisição** (`POST /predict`):
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 450.75
}
```

**Resposta:**
```json
{
  "churn_prediction": true,
  "churn_probability": 0.5952,
  "risk_level": "Alto"
}
```

## 📁 Estrutura do projeto

```
churn-prediction-copilot/
├── data/
│   ├── raw/                     # dataset original
│   ├── knowledge_base/          # documentos usados no RAG
│   └── processed_churn.csv      # dataset pré-processado
├── models/
│   ├── churn_model.pkl          # modelo treinado
│   └── feature_columns.pkl      # colunas esperadas pelo modelo
├── notebooks/
│   └── 01_eda.ipynb             # análise exploratória
├── src/
│   ├── train.py                 # script de treino
│   ├── api.py                   # API FastAPI
│   ├── utils.py                 # funções de pré-processamento compartilhadas
│   ├── build_knowledge_base.py  # indexação dos documentos no ChromaDB
│   ├── tools.py                 # ferramentas do agente (predict + RAG)
│   ├── agent.py                 # lógica do agente conversacional
│   └── chat_test.py             # script de teste do agente via terminal
├── Dockerfile
├── requirements.txt              # dependências de produção
├── requirements-dev.txt          # dependências de desenvolvimento (Jupyter, etc.)
└── README.md
```

## 🔮 Próximos passos (roadmap)

- [x] Integrar um agente de IA generativa (LLM) para explicar as previsões em linguagem natural
- [x] Implementar RAG para perguntas conceituais sobre churn
- [ ] Unir tudo numa interface visual (Streamlit)
- [ ] Adicionar explicabilidade do modelo com SHAP (por que cada previsão foi feita)
- [ ] Adicionar testes automatizados (pytest)
- [ ] Deploy em nuvem (Render/Railway)

## 👤 Autor

Eduardo Henrique — Estudante de Ciência de Dados e IA na CESAR School
[LinkedIn](https://www.linkedin.com/in/eduardo-henrique15/)