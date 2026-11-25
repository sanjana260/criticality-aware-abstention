"""Smoke test: find a working GLM endpoint + model id. Prints nothing sensitive."""
import sys

sys.path.insert(0, ".")

from critab import config
from critab.glm import chat

CANDIDATE_BASE_URLS = [
    "https://api.z.ai/api/paas/v4/",          # Z.ai (international)
    "https://open.bigmodel.cn/api/paas/v4/",  # Zhipu (China)
]
CANDIDATE_MODELS = ["glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4-plus"]


def main() -> None:
    for base in CANDIDATE_BASE_URLS:
        for model in CANDIDATE_MODELS:
            config.GLM_BASE_URL = base
            config.GLM_MODEL = model
            # reset cached client so it picks up new endpoint
            import critab.glm as g

            g._client = None
            try:
                out = chat("Reply with exactly one word.", "Say: ready", max_retries=1, timeout=20)
                print(f"OK  base={base} model={model} reply={out.strip()[:40]!r}")
                print(f"\nPin these in .env:\n  GLM_BASE_URL={base}\n  GLM_MODEL={model}")
                return
            except Exception as e:  # noqa: BLE001
                msg = str(e)[:120].replace("\n", " ")
                print(f"..  base={base} model={model} -> {type(e).__name__}: {msg}")
    print("No working endpoint/model combination found.")
    sys.exit(1)


if __name__ == "__main__":
    main()
