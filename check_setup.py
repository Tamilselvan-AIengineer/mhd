"""
Pre-flight Setup Checker
Run this before starting the app to verify everything is in order.

    python check_setup.py
"""

import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS  = "  ✅"
FAIL  = "  ❌"
WARN  = "  ⚠️ "
DIVIDER = "─" * 54


def check(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    line = f"{icon}  {label}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    return ok


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ── 1. Python version ────────────────────────────────────────────────
section("Python")
py_ok = sys.version_info >= (3, 9)
check("Python >= 3.9", py_ok, f"found {sys.version.split()[0]}")

# ── 2. Required packages ─────────────────────────────────────────────
section("Required packages")

PACKAGES = [
    ("streamlit",            "streamlit"),
    ("sklearn",              "scikit-learn"),
    ("xgboost",              "xgboost"),
    ("pandas",               "pandas"),
    ("numpy",                "numpy"),
    ("nltk",                 "nltk"),
    ("textblob",             "textblob"),
    ("vaderSentiment",       "vaderSentiment"),
    ("plotly",               "plotly"),
    ("joblib",               "joblib"),
    ("scipy",                "scipy"),
]

all_pkgs_ok = True
for import_name, pip_name in PACKAGES:
    try:
        importlib.import_module(import_name)
        check(pip_name, True)
    except ImportError:
        check(pip_name, False, f"pip install {pip_name}")
        all_pkgs_ok = False

# ── 3. NLTK data ──────────────────────────────────────────────────────
section("NLTK resources")

import nltk
NLTK_RESOURCES = [
    ("punkt",                      "tokenizers/punkt"),
    ("stopwords",                  "corpora/stopwords"),
    ("wordnet",                    "corpora/wordnet"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]

for resource_name, resource_path in NLTK_RESOURCES:
    try:
        nltk.data.find(resource_path)
        check(resource_name, True)
    except LookupError:
        try:
            nltk.download(resource_name, quiet=True)
            check(resource_name, True, "downloaded now")
        except Exception:
            check(resource_name, False, f"nltk.download('{resource_name}')")

# ── 4. Project files ──────────────────────────────────────────────────
section("Project files")

REQUIRED_FILES = [
    "app.py",
    "train.py",
    "requirements.txt",
    "packages.txt",
    ".gitignore",
    ".streamlit/config.toml",
    "models/__init__.py",
    "models/ml_models.py",
    "utils/__init__.py",
    "utils/preprocessor.py",
    "utils/sentiment.py",
    "utils/visualizations.py",
    "utils/storage.py",
    "utils/data_loader.py",
]

all_files_ok = True
for f in REQUIRED_FILES:
    exists = os.path.exists(f)
    check(f, exists)
    if not exists:
        all_files_ok = False

# ── 5. Trained models ─────────────────────────────────────────────────
section("Trained models")

MODEL_FILES = [
    "models/tfidf_vectorizer.pkl",
    "models/logistic_regression.pkl",
    "models/svm_model.pkl",
    "models/xgboost_model.pkl",
]

models_present = all(os.path.exists(f) for f in MODEL_FILES)
for f in MODEL_FILES:
    check(f, os.path.exists(f))

if not models_present:
    print(f"\n{WARN}  Models not found. Run:  python train.py")

# ── 6. Data folder ────────────────────────────────────────────────────
section("Data folder")

check("data/ directory exists", os.path.isdir("data"))
csv_path = "data/suicide_detection.csv"
csv_exists = os.path.exists(csv_path)
if csv_exists:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        check("suicide_detection.csv", True, f"{len(df)} rows")
    except Exception as e:
        check("suicide_detection.csv", False, str(e))
else:
    print(f"{WARN}  suicide_detection.csv not found — demo dataset will be used")

# ── 7. Quick import test ──────────────────────────────────────────────
section("Module import test")

try:
    from utils.preprocessor import preprocess
    check("utils.preprocessor", True)
except Exception as e:
    check("utils.preprocessor", False, str(e))

try:
    from utils.sentiment import get_full_sentiment
    check("utils.sentiment", True)
except Exception as e:
    check("utils.sentiment", False, str(e))

try:
    from utils.storage import save_analysis
    check("utils.storage", True)
except Exception as e:
    check("utils.storage", False, str(e))

try:
    from utils.data_loader import validate_dataset
    check("utils.data_loader", True)
except Exception as e:
    check("utils.data_loader", False, str(e))

try:
    from models.ml_models import predict_single
    check("models.ml_models", True)
except Exception as e:
    check("models.ml_models", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
if all_pkgs_ok and all_files_ok:
    print("  ✅  All checks passed. You're ready to run:")
    print()
    if not models_present:
        print("      python train.py          (train models first)")
    print("      streamlit run app.py     (launch dashboard)")
    print("      uvicorn backend:app --reload --port 8000  (optional API)")
else:
    print("  ❌  Some checks failed. Fix the issues above, then re-run.")
print(DIVIDER)
