"""Usage: python -m scripts.build_index   (re-run whenever you add files to data/raw/)"""
from app.pipeline import build_index

store = build_index()
n_official = sum(c.official_text for c in store.chunks)
print(f"Indexed {len(store.chunks)} chunks ({n_official} from official raw files, {len(store.chunks) - n_official} seed summaries) "
      f"using backend '{store.embedder.name}'.")
