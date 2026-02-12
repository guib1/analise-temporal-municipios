import json
import os
import shutil
import sys
from pathlib import Path

def find_repo_root():
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return current

ROOT = find_repo_root()
print(f"Raiz do repositório detectada: {ROOT}")

# 1. Limpar Cache
cache_dir = ROOT / "data/cache"
if cache_dir.exists():
    print(f"Limpando cache em: {cache_dir}")
    try:
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Cache limpo com sucesso.")
    except Exception as e:
        print(f"⚠️ Erro ao limpar cache: {e}")
else:
    print("ℹ️ Diretório de cache não existe (nada a limpar).")

# 2. Corrigir Notebook
nb_path = ROOT / "src/baixar_shapefiles.ipynb"
if not nb_path.exists():
    print(f"❌ Notebook não encontrado: {nb_path}")
    sys.exit(1)

print(f"Corrigindo notebook: {nb_path}")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Célula de setup robusta
setup_code = [
    "import sys\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Identificar raiz do projeto automaticamente\n",
    "def get_project_root():\n",
    "    current = Path.cwd()\n",
    "    while current.name != 'analise-temporal-municipios' and current.parent != current:\n",
    "        current = current.parent\n",
    "    return current\n",
    "\n",
    "ROOT = get_project_root()\n",
    "print(f\"Project Root: {ROOT}\")\n",
    "\n",
    "# Adicionar raiz ao sys.path para permitir imports de 'src'\n",
    "if str(ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(ROOT))\n",
    "\n",
    "# Mudar diretório de trabalho para a raiz para facilitar caminhos relativos (data/...)\n",
    "os.chdir(ROOT)\n",
    "print(f\"Working Directory defined to: {os.getcwd()}\")\n"
]

# Verificar se já existe célula de setup
first_cell = nb["cells"][0]
is_setup = False
if first_cell["cell_type"] == "code":
    source = "".join(first_cell["source"])
    if "sys.path" in source and "os.chdir" in source:
        is_setup = True

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code
}

if is_setup:
    print("Atualizando célula de setup existente...")
    nb["cells"][0] = new_cell
else:
    print("Inserindo nova célula de setup no topo...")
    nb["cells"].insert(0, new_cell)

# Remover células antigas que tentavam fazer setup manual e falhavam
cells_to_keep = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    # Remove old fix logic that might conflict
    if "def find_repo_root" in src and "os.chdir" in src and cell != nb["cells"][0]:
        print("Removendo célula de setup antiga/manual...")
        continue
    cells_to_keep.append(cell)

nb["cells"] = cells_to_keep

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Notebook corrigido com sucesso!")
