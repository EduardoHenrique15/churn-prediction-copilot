# Deploy — passo a passo

A API vai para o **Render** (Docker, plano gratuito) e a interface para o **Streamlit Community Cloud**. Siga na ordem: a interface precisa da URL da API.

## 1. Antes de publicar (na sua máquina)

```bash
pip install -r requirements-dev.txt

# 1. Treina com o código novo: gera models/* e reescreve os documentos 04 e 05
#    da base de conhecimento com os números do modelo.
python -m src.train

# 2. Confere tudo (offline, sem gastar cota do Gemini).
ruff check . && ruff format --check .
pytest

# 3. Reindexa a base do assistente (usa algumas requisições da API de
#    embeddings do Gemini).
python -m src.build_knowledge_base

# 4. Grava as respostas de demonstração do assistente (~6 requisições da
#    cota diária do Gemini; o script pede confirmação antes).
python -m src.record_demo
```

Rode a API e a interface juntas para uma última olhada:

```bash
uvicorn src.api:app --reload      # terminal 1
streamlit run app.py              # terminal 2 → http://localhost:8501
```

Depois, versione os artefatos novos (modelo, documentos 04 e 05 e `data/demo/agent_demo.json`):

```bash
git add -A
git status   # confira: nada de .env, .streamlit/secrets.toml nem chroma_db/
git commit -m "Retreino e respostas de demonstração"
git push
```

A pasta `chroma_db/` **não** vai para o repositório: é gerada a partir dos `.md` de `data/knowledge_base/` (que estão versionados). Na nuvem e no Docker, o assistente monta a base sozinho na primeira busca — uma requisição de embeddings, e só quando há `GEMINI_API_KEY` (sem a chave, o assistente nem é usado ao vivo).

## 2. API no Render

1. Em [dashboard.render.com](https://dashboard.render.com), crie um **Blueprint** apontando para o repositório — o `render.yaml` já descreve o serviço (Docker, plano free, healthcheck em `/health`).
   - Alternativa: **New → Web Service → Docker**, com `Dockerfile` na raiz e *Health Check Path* `/health`.
2. Espere o primeiro build terminar e abra `https://<seu-serviço>.onrender.com/health`. A resposta mostra a versão da API, o modelo e o corte em uso.
3. Abra também `/docs`: é a documentação interativa (Swagger) da API — um bom link para o README e para o post.

Variáveis opcionais (aba *Environment*): `RATE_LIMIT_PER_MINUTE` (padrão 120 requisições por minuto por IP) e `CORS_ALLOW_ORIGINS`.

> O plano gratuito desliga o serviço após 15 minutos sem tráfego, e ele leva ~1 minuto para voltar. A interface não depende disso: enquanto a API acorda, ela calcula a mesma previsão localmente (ver `docs/decisoes/004-plano-b-local.md`).

## 3. Interface no Streamlit Community Cloud

1. Em [share.streamlit.io](https://share.streamlit.io), **Create app** → repositório, branch `main`, arquivo principal `app.py`.
2. Em **Advanced settings**, escolha **Python 3.11** e cole os secrets:

```toml
GEMINI_API_KEY = "sua-chave-do-google-ai-studio"
CHURN_API_URL = "https://<seu-serviço>.onrender.com"

AUTHOR_NAME = "Eduardo Henrique"
GITHUB_URL = "https://github.com/EduardoHenrique15/churn-prediction-copilot"
LINKEDIN_URL = "https://www.linkedin.com/in/eduardo-henrique15/"

# Protegem a cota gratuita do Gemini (~20 requisições/dia)
AGENT_MAX_QUESTIONS_PER_SESSION = "3"
AGENT_MAX_QUESTIONS_PER_DAY = "6"
```

3. Publique. O Streamlit Cloud instala `requirements.txt` (sem FastAPI) e lê `.streamlit/config.toml` (tema e fontes).

## 4. Conferência depois do deploy

- [ ] O chip no topo mostra **API online** (se mostrar "acordando", espere 1 minuto e recarregue).
- [ ] **Cliente:** os quatro exemplos funcionam; "Cliente real" mostra o desfecho verdadeiro.
- [ ] **Carteira:** os dois exemplos carregam; "clientes novos" acusa mudança forte.
- [ ] **Estratégia:** mexer nos controles muda o corte recomendado e o gráfico.
- [ ] **Modelo:** gráficos e tabelas aparecem com os números do treino.
- [ ] **Assistente:** as respostas gravadas aparecem; uma pergunta ao vivo responde e mostra a ferramenta usada.
- [ ] No celular: a navegação abre pelo botão do topo e o resultado aparece antes do formulário.
- [ ] Atualize os links de demonstração no topo do `README.md`.

## 5. Opcional: manter a API acordada

Um monitor gratuito (UptimeRobot, cron-job.org) chamando `https://<seu-serviço>.onrender.com/health` a cada 10 minutos evita a hibernação. As 750 horas gratuitas por mês do Render cobrem um serviço ligado o mês inteiro. Não é necessário: o plano B local já cobre o tempo de acordar.
