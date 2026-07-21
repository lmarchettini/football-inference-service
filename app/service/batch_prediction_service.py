import logging

from app.dto.batch_prediction_request import (
    BatchPredictionRequest,
)

from app.dto.batch_prediction_response import (
    BatchPredictionFailure,
    BatchPredictionResponse,
    BatchPredictionResult,
)

from app.service.prediction_service import (
    ModelCompatibilityError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionExecutionError,
    PredictionValidationError,
    predict,
)


logger = logging.getLogger(__name__)


def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:

    logger.info(
        "Batch prediction request received: "
        "predictions_count=%s",
        len(request.predictions),
    )

    predictions: list[
        BatchPredictionResult
    ] = []

    failures: list[
        BatchPredictionFailure
    ] = []

    for item in request.predictions:

        try:
            prediction = predict(
                item
            )

            predictions.append(
                BatchPredictionResult(
                    fixture_id=item.fixture_id,
                    market=prediction.market,
                    probability=(
                        prediction.probability
                    ),
                    negative_probability=(
                        prediction
                        .negative_probability
                    ),
                )
            )

        except PredictionValidationError as exc:
            failures.append(
                _build_failure(
                    fixture_id=item.fixture_id,
                    market=item.market,
                    code=(
                        "PREDICTION_VALIDATION_ERROR"
                    ),
                    exception=exc,
                )
            )

        except ModelNotFoundError as exc:
            failures.append(
                _build_failure(
                    fixture_id=item.fixture_id,
                    market=item.market,
                    code="MODEL_NOT_FOUND",
                    exception=exc,
                )
            )

        except ModelCompatibilityError as exc:
            failures.append(
                _build_failure(
                    fixture_id=item.fixture_id,
                    market=item.market,
                    code=(
                        "MODEL_COMPATIBILITY_ERROR"
                    ),
                    exception=exc,
                )
            )

        except ModelLoadError as exc:
            failures.append(
                _build_failure(
                    fixture_id=item.fixture_id,
                    market=item.market,
                    code="MODEL_LOAD_ERROR",
                    exception=exc,
                )
            )

        except PredictionExecutionError as exc:
            failures.append(
                _build_failure(
                    fixture_id=item.fixture_id,
                    market=item.market,
                    code=(
                        "PREDICTION_EXECUTION_ERROR"
                    ),
                    exception=exc,
                )
            )

        except Exception as exc:
            logger.exception(
                "Unexpected batch prediction error: "
                "fixture_id=%s, market=%s",
                item.fixture_id,
                item.market,
            )

            failures.append(
                _build_failure(
                    fixture_id=item.fixture_id,
                    market=item.market,
                    code=(
                        "UNEXPECTED_PREDICTION_ERROR"
                    ),
                    exception=exc,
                    public_message=(
                        "Unexpected internal "
                        "prediction error"
                    ),
                )
            )

    logger.info(
        "Batch prediction completed: "
        "requested=%s, successful=%s, "
        "failed=%s",
        len(request.predictions),
        len(predictions),
        len(failures),
    )

    return BatchPredictionResponse(
        predictions=predictions,
        failures=failures,
    )


def _build_failure(
    fixture_id: int,
    market: str,
    code: str,
    exception: Exception,
    public_message: str | None = None,
) -> BatchPredictionFailure:

    message = (
        public_message
        if public_message is not None
        else str(exception)
    )

    logger.warning(
        "Batch prediction item failed: "
        "fixture_id=%s, market=%s, "
        "code=%s, message=%s",
        fixture_id,
        market,
        code,
        message,
    )

    return BatchPredictionFailure(
        fixture_id=fixture_id,
        market=market,
        code=code,
        message=message,
    )