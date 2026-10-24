"""WikiText-103 test NLL only (p05.evaluate protocol, HF backend): python -m p05.diag_eval --ckpt D --suite diag_wiki --out D2 --data-root R"""
from . import evaluate as PE

PE.E.SUITES["diag_wiki"] = ["wikitext"]

if __name__ == "__main__":
    PE.main()
