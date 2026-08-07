import os

from dotenv import load_dotenv

load_dotenv()


def _parse_enabled_markets() -> set[str]:
    return {
        market.strip().lower()
        for market in os.getenv(
            "ENABLED_MARKETS",
            "*",
        ).split(",")
        if market.strip()
    }


def _parse_derived_inverse_markets() -> dict[str, str]:
    """
    Parses a configuration such as:

    DERIVED_INVERSE_MARKETS=
        double_chance_x2:home_win

    Meaning:

    probability(double_chance_x2)
        =
    probability(class 0 of home_win)
    """

    raw_value = os.getenv(
        "DERIVED_INVERSE_MARKETS",
        "",
    )

    derived_markets: dict[str, str] = {}

    for entry in raw_value.split(","):

        normalized_entry = entry.strip()

        if not normalized_entry:
            continue

        parts = normalized_entry.split(
            ":",
            maxsplit=1,
        )

        if len(parts) != 2:
            raise ValueError(
                "Invalid DERIVED_INVERSE_MARKETS entry: "
                f"'{normalized_entry}'. "
                "Expected format "
                "'derived_market:source_market'."
            )

        derived_market = parts[0].strip().lower()

        source_market = parts[1].strip().lower()

        if not derived_market:
            raise ValueError("Derived market cannot be empty")

        if not source_market:
            raise ValueError("Source market cannot be empty")

        if derived_market == source_market:
            raise ValueError(
                "Derived market and source market "
                "cannot be the same: "
                f"'{derived_market}'"
            )

        derived_markets[derived_market] = source_market

    return derived_markets


ENABLED_MARKETS = _parse_enabled_markets()

DERIVED_INVERSE_MARKETS = _parse_derived_inverse_markets()


def normalize_market(
    market: str | None,
) -> str | None:

    if market is None:
        return None

    normalized_market = market.strip().lower()

    return normalized_market if normalized_market else None


def is_market_enabled(
    market: str,
) -> bool:

    normalized_market = normalize_market(market)

    if normalized_market is None:
        return False

    return (
        "*" in ENABLED_MARKETS
        or normalized_market in ENABLED_MARKETS
        or normalized_market in DERIVED_INVERSE_MARKETS
    )


def get_inverse_source_market(
    market: str,
) -> str | None:

    normalized_market = normalize_market(market)

    if normalized_market is None:
        return None

    return DERIVED_INVERSE_MARKETS.get(normalized_market)


def is_inverse_derived_market(
    market: str,
) -> bool:

    return get_inverse_source_market(market) is not None
