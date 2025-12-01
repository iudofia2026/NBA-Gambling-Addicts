# Player Peers Streamlit App

This small Streamlit app provides a quick, interactive comparison for a set of NBA players using the repository's `engineered_features.csv` dataset.

How to run (from repository root):

```bash
# create venv (optional)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# launch the app
streamlit run apps/player_peers/app.py
```

The app expects `data/processed/engineered_features.csv` to be present.

Alternative (recommended for fresh machines or when you see NumPy / pandas import errors):

```bash
# this script bootstraps a local venv and runs the app
python scripts/run_player_peers.py
```

This will create `./.venv`, install `requirements.txt` into it, and then launch the Streamlit app inside the venv.
