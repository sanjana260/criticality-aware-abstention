"""Central configuration: paths, constants, seeds, and the pre-registered t(c) table.

The t(c) table below is PRE-REGISTERED: it must not be changed after test data
has been looked at. See experiment plan, flaw fix #1.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- GLM judge client ---
GLM_API_KEY = os.getenv("GLM_API_KEY")
# Pinned 2026-11-23: key is a Coding Plan key; flash/flashx tiers not included in plan.
# glm-5.3 verified working (chat + JSON); glm-4.7 returns empty completions on this plan.
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4/")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-5.3")

# --- Base model (same checkpoint planned for Phase-2 RL) ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# --- Pre-registered threshold table (flaw fix #1) ---
# Judge emits levels 1-5; analysis bins: low = {1,2}, mid = {3}, high = {4,5}.
CRITICALITY_BIN_THRESHOLDS = {
    "low": 0.60,
    "mid": 0.85,
    "high": 0.97,
}
LEVEL_TO_BIN = {1: "low", 2: "low", 3: "mid", 4: "high", 5: "high"}

# --- Reproducibility ---
SEED = 20261123

# --- Data pool sizes (plan: standard scope) ---
N_GENERAL = 200          # ARC-Challenge + MMLU-Pro slice
N_MEDICAL = 300          # MedMCQA + MedHallu-derived MCQA
N_MATCHED_PAIRS = 40     # hand-built pairs -> 80 questions

# --- Split ---
CALIB_FRACTION = 0.30

# --- Judge gate criteria (plan: quick-look rigor) ---
SANITY_PAIRS_MIN_ACCURACY = 0.90   # fraction of hand-built pairs ordered correctly
RETEST_MIN_AGREEMENT = 0.80        # test-retest agreement on 30 questions

# --- Confidence extraction ---
LETTER_TOKENS = ["A", "B", "C", "D"]  # substantive options max 3 + IDK in v1 pools
