# 005 — Dependências separadas por serviço

**Status:** aceita · **Data:** 26/09/2026

## Contexto

Um único `requirements.txt` instalava tudo em todas as imagens: a API levava Streamlit, LangChain, ChromaDB e SHAP (~1,35 GB instalados) sem usar nenhum deles, num servidor com 512 MB de RAM.

## Decisão

| Arquivo | Quem usa | Conteúdo |
|---|---|---|
| `requirements-api.txt` | Dockerfile da API (Render) | FastAPI, Uvicorn, Pydantic, pandas, scikit-learn, joblib |
| `requirements.txt` | Streamlit Community Cloud e `Dockerfile.streamlit` | interface, modelo (plano B local) e assistente |
| `requirements-dev.txt` | desenvolvimento e CI | os dois acima + treino, notebook e testes |

`tests/test_packaging.py` lê os imports do código e falha se a API passar a importar algo fora da lista dela (ou a interface, fora da dela), e se um retreino escolher um modelo de árvore sem o SHAP na imagem da API.

## Consequências

- A API sobe com cerca de 170 MB de RAM (medido com só `requirements-api.txt` instalado); o CI mede de novo a cada push, dentro do container.
- Versões fixadas iguais nos arquivos que se sobrepõem (um teste confere).
