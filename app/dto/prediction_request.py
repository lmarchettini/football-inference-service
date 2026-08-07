from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class PredictionRequest(BaseModel):

    model_path: str = Field(min_length=1)

    market: str = Field(min_length=1)

    features: list[float | None]

    @field_validator("features")
    @classmethod
    def validate_features(
        cls,
        values: list[float | None],
    ) -> list[float | None]:

        if not values:
            raise ValueError("Features cannot be empty")

        return values
