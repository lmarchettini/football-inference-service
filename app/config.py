import os

ENABLED_MARKETS = {
    market.strip()
    for market in os.getenv(
        "ENABLED_MARKETS",
        "*",
    ).split(",")
    if market.strip()
}

def is_market_enabled(
        market: str,
) -> bool:

    return (
        "*" in ENABLED_MARKETS
        or market in ENABLED_MARKETS
    )

EXPECTED_FEATURE_VERSION = os.getenv(
    "EXPECTED_FEATURE_VERSION",
    "v8",
)

EXPECTED_FEATURES = 62
