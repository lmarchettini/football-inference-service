import os

ENABLED_MARKETS = {
    market.strip().lower()
    for market in os.getenv(
        "ENABLED_MARKETS",
        "*",
    ).split(",")
    if market.strip()
}


def is_market_enabled(
    market: str,
) -> bool:

    if market is None:
        return False

    normalized_market = market.strip().lower()

    return "*" in ENABLED_MARKETS or normalized_market in ENABLED_MARKETS
