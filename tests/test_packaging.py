"""Guardas de empacotamento: cada imagem instala só o que usa.

A API roda no plano gratuito do Render (512 MB de RAM). Estes testes leem os
imports do código (sem executá-lo) e falham se:
- a API passar a importar algo que não está em requirements-api.txt (por
  exemplo, Streamlit ou LangChain puxados sem querer por um import);
- a interface ou o agente importarem algo fora de requirements.txt;
- um retreino escolher um modelo de árvore e o SHAP não estiver na API.
"""

import ast
import re
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent

# nome do import -> nome do pacote no requirements
PACKAGE_OF = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "langchain_core": "langchain-core",
    "langchain_google_genai": "langchain-google-genai",
    "langchain_chroma": "langchain-chroma",
    "langchain_text_splitters": "langchain-text-splitters",
}
# dependências transitivas garantidas pelos pacotes listados
TRANSITIVE = {"numpy", "starlette"}


def _requirements(name: str) -> set[str]:
    packages = set()
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            packages |= _requirements(line[3:].strip())
            continue
        packages.add(re.split(r"[=<>!~\[ ]", line)[0].lower())
    return packages


def _imports(path: Path, top_level_only: bool) -> tuple[set[str], set[str]]:
    """(módulos de terceiros, módulos src.*) importados pelo arquivo."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    third, local = set(), set()
    for node in nodes:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root == "src":
                local.add(name)
            elif root not in sys.stdlib_module_names and root != "__future__":
                third.add(root)
    return third, local


def _closure(entry: Path, top_level_only: bool) -> set[str]:
    """Pacotes de terceiros alcançáveis a partir de um arquivo, seguindo src.*"""
    seen, pending, third = set(), [entry], set()
    while pending:
        path = pending.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        found, local = _imports(path, top_level_only)
        third |= found
        for module in local:
            candidate = ROOT / (module.replace(".", "/") + ".py")
            package = ROOT / module.replace(".", "/") / "__init__.py"
            pending += [candidate, package]
    return third


def _as_packages(modules: set[str]) -> set[str]:
    return {PACKAGE_OF.get(m, m).lower() for m in modules} - TRANSITIVE


def test_api_so_importa_o_que_esta_em_requirements_api():
    # Imports dentro de funções ficam de fora de propósito: o SHAP é importado
    # sob demanda e só para modelos de árvore (ver o teste abaixo).
    needed = _as_packages(_closure(ROOT / "src" / "api.py", top_level_only=True))
    missing = needed - _requirements("requirements-api.txt")
    assert not missing, f"A API importa pacotes fora de requirements-api.txt: {missing}"


def test_api_nao_carrega_interface_nem_agente():
    needed = _closure(ROOT / "src" / "api.py", top_level_only=True)
    assert not needed & {"streamlit", "langchain_core", "langchain_google_genai", "chromadb"}


def test_modelo_de_arvore_exige_shap_na_imagem_da_api():
    from src.explain import is_linear

    model = joblib.load(ROOT / "models" / "churn_model.pkl")
    if is_linear(model):
        pytest.skip("modelo linear: explicação exata, sem SHAP")
    assert "shap" in _requirements("requirements-api.txt")


@pytest.mark.parametrize(
    "entry",
    ["app.py", "src/agent.py", "src/tools.py", "src/record_demo.py", "src/build_knowledge_base.py"],
)
def test_interface_e_agente_so_importam_o_que_esta_em_requirements(entry):
    needed = _as_packages(_closure(ROOT / entry, top_level_only=False))
    ui_pages = set()
    for page in (ROOT / "src" / "ui").rglob("*.py"):
        ui_pages |= _as_packages(_imports(page, top_level_only=False)[0])
    missing = (needed | ui_pages) - _requirements("requirements.txt")
    # SHAP só é usado se o modelo for de árvore (e aí a explicação vem da API).
    missing.discard("shap")
    assert not missing, f"Pacotes fora de requirements.txt: {missing}"


def test_versoes_fixadas_iguais_nos_dois_requirements():
    def pins(name: str) -> dict[str, str]:
        out = {}
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                package, version = line.split("==")
                out[package.lower()] = version
        return out

    api, ui = pins("requirements-api.txt"), pins("requirements.txt")
    for package in api.keys() & ui.keys():
        assert api[package] == ui[package], f"{package}: {api[package]} x {ui[package]}"
