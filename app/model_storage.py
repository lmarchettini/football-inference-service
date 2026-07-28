import os
from pathlib import Path


def _resolve_model_artifacts_dir() -> Path:
    configured_path = os.getenv("MODEL_ARTIFACTS_DIR")

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    # Questo file si trova in:
    # BettingBrain/
    #   football-inference-service/
    #     app/
    #       config/
    #         model_storage.py
    #
    # parents[3] = BettingBrain
    betting_brain_root = Path(__file__).resolve().parents[2]

    return (betting_brain_root / "model-artifacts").resolve()


MODEL_ARTIFACTS_DIR = _resolve_model_artifacts_dir()
