"""Curated institutional managers that file Form 13F reports."""

WATCHLIST_MANAGERS = {
    # Small, concentrated public-equity manager tied to the VST accumulation signal.
    "thiel_macro": {"name": "Thiel Macro LLC", "cik": "1562087"},
    # David Tepper's concentrated Appaloosa portfolio is a useful smart-money signal.
    "appaloosa": {"name": "Appaloosa LP", "cik": "1656456"},
    # Berkshire is a well-known long-term equity holder and institutional benchmark.
    "berkshire_hathaway": {"name": "Berkshire Hathaway Inc", "cik": "1067983"},
    # Pershing Square is a prominent activist investor with concentrated positions.
    "pershing_square": {
        "name": "Pershing Square Capital Management, L.P.",
        "cik": "1336528",
    },
    # Citadel is a major hedge fund whose filings expose material position changes.
    "citadel_advisors": {"name": "Citadel Advisors LLC", "cik": "1423053"},
}
