# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
source ~/.bashrc

# install modern Python and create venv
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# install deps (JAX GPU + project)
uv pip install -U pip
uv pip install -U "jax"
uv pip install -r requirements.txt