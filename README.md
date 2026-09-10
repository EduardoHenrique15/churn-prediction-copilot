# 🔮 Churn Prediction Copilot

Sistema de previsão de cancelamento de clientes (churn) para uma empresa de telecomunicações, com modelo de Machine Learning servido via API REST e rastreamento completo de experimentos.

## 📌 Problema de negócio

Empresas de assinatura (telecom, SaaS, streaming) enfrentam um desafio constante: identificar **quais clientes estão prestes a cancelar** antes que isso aconteça, para que times de retenção possam agir preventivamente (ofertas, contato proativo, ajuste de plano).

Este projeto treina um modelo de classificação que prevê a probabilidade de um cliente cancelar o serviço, com base em seu perfil de uso e contrato, e disponibiliza essa previsão via API pronta para integração com outros sistemas.

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
```

## 📊 Resultados do modelo

| Métrica | Valor |
|---|---|
| Acurácia | 0.75 |
| Precision (classe "Churn") | 0.52 |
| Recall (classe "Churn") | 0.77 |
| F1-score | 0.62 |

**Por que priorizar recall em vez de acurácia:** o custo de não identificar um cliente que vai cancelar (falso negativo) é maior para o negócio do que contatar preventivamente um cliente que não ia cancelar (falso positivo). Por isso o modelo foi ajustado para maximizar a detecção de clientes em risco, mesmo aceitando mais falsos positivos.

## 🛠️ Stack técnica

- **Linguagem:** Python 3.11
- **Análise e modelagem:** Pandas, Scikit-learn
- **Rastreamento de experimentos:** MLflow (tracking de parâmetros, métricas e versionamento de modelo)
- **API:** FastAPI + Uvicorn
- **Containerização:** Docker

## 🚀 Como rodar o projeto

### Pré-requisitos
- Python 3.11+
- Docker Desktop (opcional, para rodar via container)

### Rodando localmente

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

# Treine o modelo
cd src
python train.py

# Suba a API
uvicorn api:app --reload
```

Acesse a documentação interativa em `http://127.0.0.1:8000/docs`

### Rodando via Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Acesse `http://127.0.0.1:8000/docs`

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
│   └── processed_churn.csv      # dataset pré-processado
├── models/
│   ├── churn_model.pkl          # modelo treinado
│   └── feature_columns.pkl      # colunas esperadas pelo modelo
├── notebooks/
│   └── 01_eda.ipynb             # análise exploratória
├── src/
│   ├── train.py                 # script de treino
│   ├── api.py                   # API FastAPI
│   └── utils.py                 # funções de pré-processamento compartilhadas
├── Dockerfile
├── requirements.txt              # dependências de produção
├── requirements-dev.txt          # dependências de desenvolvimento (Jupyter, etc.)
└── README.md
```

## 🔮 Próximos passos (roadmap)

- [ ] Adicionar explicabilidade do modelo com SHAP (por que cada previsão foi feita)
- [ ] Integrar um agente de IA generativa (LLM) para explicar as previsões em linguagem natural
- [ ] Adicionar testes automatizados (pytest)
- [ ] Deploy em nuvem (Render/Railway)

## 👤 Autor

Eduardo Henrique — Estudante de Ciência de Dados e IA na CESAR School
[LinkedIn](https://www.linkedin.com/in/eduardo-henrique15/)