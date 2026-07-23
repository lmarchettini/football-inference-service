import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from threading import Lock

from app.config import (
    EXPECTED_FEATURES,
    EXPECTED_FEATURE_VERSION,
)

from app.dto.prediction_response import (
    PredictionResponse,
)

from app.config import (
    EXPECTED_FEATURES,
    EXPECTED_FEATURE_VERSION,
    is_market_enabled,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MODEL_CACHE: dict[Path, Any] = {}
_MODEL_CACHE_LOCK = Lock()


class PredictionValidationError(ValueError):
    """Errore nei dati ricevuti dal client."""
    
class MarketDisabledError(
        RuntimeError
):
    """Market disabled."""


class ModelNotFoundError(FileNotFoundError):
    """Il file del modello non esiste."""


class ModelLoadError(RuntimeError):
    """Il modello non può essere caricato."""


class ModelCompatibilityError(RuntimeError):
    """Il modello non è compatibile con la richiesta."""


class PredictionExecutionError(RuntimeError):
    """Errore durante l'esecuzione dell'inferenza."""


def predict(request) -> PredictionResponse:

    logger.info(
        "Prediction request received: "
        "market=%s, feature_version=%s, "
        "features_count=%s, model_path=%s",
        request.market,
        request.feature_version,
        len(request.features)
        if request.features is not None
        else None,
        request.model_path,
    )
    
    if not is_market_enabled(
        request.market
):
        raise MarketDisabledError(
            f"Market '{request.market}' is disabled."
        )

    try:
        _validate_request(request)

        model_path = _resolve_model_path(
            request.model_path
        )

        logger.info(
            "Resolved model path: "
            "market=%s, requested_path=%s, "
            "resolved_path=%s, cwd=%s",
            request.market,
            request.model_path,
            model_path,
            Path.cwd(),
        )

        model = _load_model(
            model_path=model_path,
            market=request.market,
        )

        _validate_model(
            model=model,
            features_count=len(
                request.features
            ),
        )

        probability, negative_probability = (
            _predict_probabilities(
                model=model,
                features=request.features,
            )
        )

        response = PredictionResponse(
            market=request.market,
            probability=round(
                probability,
                4,
            ),
            negative_probability=round(
                negative_probability,
                4,
            ),
        )

        logger.info(
            "Prediction completed: "
            "market=%s, probability=%.4f, "
            "negative_probability=%.4f",
            request.market,
            probability,
            negative_probability,
        )

        return response

    except (
        PredictionValidationError,
        ModelNotFoundError,
        ModelLoadError,
        ModelCompatibilityError,
        PredictionExecutionError,
    ):
        # Queste eccezioni saranno convertite in risposte HTTP
        # appropriate dal controller FastAPI.
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected prediction error: "
            "market=%s, feature_version=%s, "
            "model_path=%s",
            getattr(
                request,
                "market",
                None,
            ),
            getattr(
                request,
                "feature_version",
                None,
            ),
            getattr(
                request,
                "model_path",
                None,
            ),
        )

        raise PredictionExecutionError(
            "Unexpected error during prediction"
        ) from exc


def _validate_request(request) -> None:

    if request.feature_version != (
        EXPECTED_FEATURE_VERSION
    ):
        message = (
            "Feature version mismatch: "
            f"expected={EXPECTED_FEATURE_VERSION}, "
            f"received={request.feature_version}"
        )

        logger.warning(
            message
        )

        raise PredictionValidationError(
            message
        )

    if request.features is None:
        raise PredictionValidationError(
            "Features are required"
        )

    received_features = len(
        request.features
    )

    if received_features != EXPECTED_FEATURES:
        message = (
            "Feature count mismatch: "
            f"expected={EXPECTED_FEATURES}, "
            f"received={received_features}"
        )

        logger.warning(
            "%s, market=%s",
            message,
            request.market,
        )

        raise PredictionValidationError(
            message
        )

    if (
        request.model_path is None
        or not request.model_path.strip()
    ):
        raise PredictionValidationError(
            "Model path is required"
        )

    if (
        request.market is None
        or not request.market.strip()
    ):
        raise PredictionValidationError(
            "Market is required"
        )

    try:
        features_array = np.asarray(
            request.features,
            dtype=float,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise PredictionValidationError(
            "All features must be numeric"
        ) from exc

    if not np.all(
        np.isfinite(features_array)
    ):
        raise PredictionValidationError(
            "Features cannot contain NaN "
            "or infinite values"
        )
        

def _resolve_model_path(
    raw_model_path: str,
) -> Path:

    requested_path = Path(
        raw_model_path
    ).expanduser()

    if requested_path.is_absolute():
        return requested_path.resolve()

    return (
        PROJECT_ROOT
        / requested_path
    ).resolve()


def _load_model(
    model_path: Path,
    market: str,
) -> Any:

    resolved_path = model_path.resolve()

    if not resolved_path.exists():
        logger.error(
            "Model file not found: "
            "market=%s, path=%s",
            market,
            resolved_path,
        )

        raise ModelNotFoundError(
            f"Model file not found: {resolved_path}"
        )

    if not resolved_path.is_file():
        logger.error(
            "Model path is not a file: "
            "market=%s, path=%s",
            market,
            resolved_path,
        )

        raise ModelNotFoundError(
            f"Model path is not a file: "
            f"{resolved_path}"
        )

    cached_model = _MODEL_CACHE.get(
        resolved_path
    )

    if cached_model is not None:
        logger.debug(
            "Using cached model: "
            "market=%s, path=%s",
            market,
            resolved_path,
        )

        return cached_model

    with _MODEL_CACHE_LOCK:

        # Un altro thread potrebbe aver caricato il modello
        # mentre aspettavamo il lock.
        cached_model = _MODEL_CACHE.get(
            resolved_path
        )

        if cached_model is not None:
            return cached_model

        try:
            logger.info(
                "Loading model from disk: "
                "market=%s, path=%s",
                market,
                resolved_path,
            )

            model = joblib.load(
                resolved_path
            )

            _MODEL_CACHE[
                resolved_path
            ] = model

            logger.info(
                "Model loaded and cached: "
                "market=%s, path=%s, type=%s",
                market,
                resolved_path,
                type(model).__name__,
            )

            return model

        except Exception as exc:
            logger.exception(
                "Failed to load model: "
                "market=%s, path=%s",
                market,
                resolved_path,
            )

            raise ModelLoadError(
                f"Failed to load model: "
                f"{resolved_path}"
            ) from exc


def _validate_model(
    model: Any,
    features_count: int,
) -> None:

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise ModelCompatibilityError(
            "Loaded model does not support "
            "predict_proba"
        )

    expected_by_model = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        expected_by_model is not None
        and expected_by_model
        != features_count
    ):
        message = (
            "Model feature count mismatch: "
            f"model_expected={expected_by_model}, "
            f"received={features_count}"
        )

        logger.error(
            message
        )

        raise ModelCompatibilityError(
            message
        )

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        raise ModelCompatibilityError(
            "Loaded model does not expose classes_"
        )

    if 0 not in classes or 1 not in classes:
        raise ModelCompatibilityError(
            "Binary model must contain "
            "classes 0 and 1; "
            f"received classes={classes.tolist()}"
        )


def _predict_probabilities(
    model: Any,
    features: list[float],
) -> tuple[float, float]:

    try:
        # DataFrame non necessario: il modello riceve un solo
        # vettore nello stesso ordine usato nel training.
        input_data = np.asarray(
            [features],
            dtype=float,
        )

        probabilities = model.predict_proba(
            input_data
        )

        if probabilities.shape[0] != 1:
            raise ModelCompatibilityError(
                "Unexpected predict_proba output: "
                f"shape={probabilities.shape}"
            )

        classes = list(
            model.classes_
        )

        positive_index = classes.index(
            1
        )

        negative_index = classes.index(
            0
        )

        probability = float(
            probabilities[
                0,
                positive_index,
            ]
        )

        negative_probability = float(
            probabilities[
                0,
                negative_index,
            ]
        )

        if not (
            0.0 <= probability <= 1.0
            and 0.0
            <= negative_probability
            <= 1.0
        ):
            raise ModelCompatibilityError(
                "Model returned invalid "
                "probability values"
            )

        return (
            probability,
            negative_probability,
        )

    except ModelCompatibilityError:
        raise

    except Exception as exc:
        logger.exception(
            "Model predict_proba failed"
        )

        raise PredictionExecutionError(
            "Model prediction failed"
        ) from exc