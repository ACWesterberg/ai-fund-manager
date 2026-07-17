"""
Paper portfolios — user-created, one-shot portfolios pasted in from an LLM answer.

A paper portfolio is created on the web (/paper): set a starting value in SEK,
paste the stock picks an LLM produced (JSON or plain "TICKER weight%" lines),
and optionally keep the base prompt that produced them. The buys execute
immediately at real market prices, then the portfolio is tracked exactly like
the simulation funds: daily NAV vs benchmark, retrospective outcome evaluation
after ~4 weeks, and distilled learnings — all against its own isolated store.

Each portfolio lives in its own SQLite DB under data/paper/<slug>.db using the
standard Store schema, so the whole evaluator/learnings pipeline works on it
unchanged. Portfolio metadata (name, capital, benchmark, base prompt, currency
map) is kept in that DB's app_meta table; the registry is just the directory.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from fundmgr.config import DATA_DIR, FeeConfig
from fundmgr.state.models import NavPoint, RecommendationLog, Transaction
from fundmgr.state.store import Store

PAPER_DIR = DATA_DIR / "paper"
DEFAULT_BENCHMARK = "URTH"  # iShares MSCI World — same default as the global sims

_fees = FeeConfig()  # same schedule as every other fund: 0.10%, min 1, max 99 SEK

# Yahoo suffix → trading currency, used when yfinance metadata is unavailable.
_SUFFIX_CURRENCY = {
    "ST": "SEK", "OL": "NOK", "CO": "DKK", "HE": "EUR", "IC": "ISK",
    "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "BR": "EUR",
    "MC": "EUR", "MI": "EUR", "LS": "EUR", "VI": "EUR", "IR": "EUR",
    "L": "GBp", "SW": "CHF", "T": "JPY", "TO": "CAD", "V": "CAD",
    "AX": "AUD", "HK": "HKD", "TW": "TWD", "KS": "KRW", "SI": "SGD",
}


# ── Registry ──────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "portfolio"


def list_portfolios() -> list[dict]:
    """All paper portfolios (metadata only — no price fetches)."""
    if not PAPER_DIR.exists():
        return []
    out = []
    for db in sorted(PAPER_DIR.glob("*.db")):
        try:
            meta, _store = open_portfolio(db.stem)
            out.append(meta)
        except Exception:
            continue
    return out


def open_portfolio(slug: str) -> tuple[dict, Store]:
    """Load one portfolio's metadata + store. Raises KeyError if missing."""
    db = PAPER_DIR / f"{slug}.db"
    if not db.exists():
        raise KeyError(f"No paper portfolio '{slug}'")
    store = Store(db)
    meta = {
        "slug": slug,
        "name": store.get_meta("paper_name") or slug,
        "capital_sek": float(store.get_meta("paper_capital_sek") or 0),
        "benchmark": store.get_meta("paper_benchmark") or DEFAULT_BENCHMARK,
        "model_label": store.get_meta("paper_model_label") or "",
        "created_at": store.get_meta("paper_created_at") or "",
        "base_prompt": store.get_meta("paper_base_prompt") or "",
        "currency_map": json.loads(store.get_meta("paper_currency_map") or "{}"),
    }
    return meta, store


def delete_portfolio(slug: str) -> None:
    """Archive a portfolio's DB (rename, don't destroy — it's a history record)."""
    db = PAPER_DIR / f"{slug}.db"
    if not db.exists():
        raise KeyError(f"No paper portfolio '{slug}'")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    db.rename(db.with_suffix(f".db.deleted-{stamp}"))


# ── Parsing pasted picks ──────────────────────────────────────────────────────

_TICKER_KEYS = ("ticker", "symbol", "yahoo_ticker")
_WEIGHT_KEYS = ("weight_pct", "weight", "target_weight_pct", "allocation_pct",
                "allocation", "percent", "pct")
_THESIS_KEYS = ("thesis", "rationale", "reason", "why", "comment", "notes")
_LIST_KEYS = ("portfolio", "holdings", "positions", "stocks", "picks", "actions",
              "recommendations", "allocations")

_WEIGHT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:[.,]\d+)?)\s*%")

# Company name → Yahoo ticker, for prose answers ("Nvidia 8% — chokepoint: …").
# Names are matched after _clean_name() normalisation (lowercase, no punctuation).
# Nordic names default to their home listing; US names to the US listing.
NAME_ALIASES: dict[str, str] = {
    # US / global mega-caps
    "nvidia": "NVDA", "microsoft": "MSFT", "apple": "AAPL", "alphabet": "GOOGL",
    "google": "GOOGL", "amazon": "AMZN", "meta": "META", "meta platforms": "META",
    "tesla": "TSLA", "netflix": "NFLX", "broadcom": "AVGO", "micron": "MU",
    "micron technology": "MU", "tsmc": "TSM", "taiwan semiconductor": "TSM",
    "amd": "AMD", "intel": "INTC", "qualcomm": "QCOM", "arm": "ARM",
    "asml": "ASML", "asml holding": "ASML", "ge vernova": "GEV",
    "vertiv": "VRT", "eaton": "ETN", "schneider electric": "SU.PA",
    "berkshire hathaway": "BRK-B", "jpmorgan": "JPM", "visa": "V",
    "eli lilly": "LLY", "unitedhealth": "UNH", "palantir": "PLTR",
    "salesforce": "CRM", "oracle": "ORCL", "adobe": "ADBE", "shopify": "SHOP",
    "constellation energy": "CEG", "linde": "LIN", "caterpillar": "CAT",
    # Europe
    "novo nordisk": "NOVO-B.CO", "lvmh": "MC.PA", "sap": "SAP.DE",
    "siemens": "SIE.DE", "siemens energy": "ENR.DE", "shell": "SHEL.L",
    "astrazeneca": "AZN.L", "nestle": "NESN.SW", "novartis": "NOVN.SW",
    "roche": "ROG.SW", "equinor": "EQNR.OL", "kongsberg": "KOG.OL",
    "rheinmetall": "RHM.DE", "airbus": "AIR.PA", "safran": "SAF.PA",
    "bae systems": "BA.L", "rolls royce": "RR.L", "totalenergies": "TTE.PA",
    # Sweden / Nordics (Stockholm listings)
    "abb": "ABB.ST", "atlas copco": "ATCO-A.ST", "atlas copco a": "ATCO-A.ST",
    "atlas copco b": "ATCO-B.ST", "volvo": "VOLV-B.ST", "ericsson": "ERIC-B.ST",
    "investor": "INVE-B.ST", "hexagon": "HEXA-B.ST", "sandvik": "SAND.ST",
    "saab": "SAAB-B.ST", "evolution": "EVO.ST", "hm": "HM-B.ST",
    "h m": "HM-B.ST", "assa abloy": "ASSA-B.ST", "epiroc": "EPI-A.ST",
    "alfa laval": "ALFA.ST", "skf": "SKF-B.ST", "seb": "SEB-A.ST",
    "swedbank": "SWED-A.ST", "handelsbanken": "SHB-A.ST", "nordea": "NDA-SE.ST",
    "mycronic": "MYCR.ST", "munters": "MTRS.ST", "mildef": "MILDEF.ST",
    "lifco": "LIFCO-B.ST", "lagercrantz": "LAGR-B.ST", "indutrade": "INDT.ST",
    "addtech": "ADDT-B.ST", "beijer ref": "BEIJ-B.ST", "nibe": "NIBE-B.ST",
    "vitrolife": "VITR.ST", "sweco": "SWEC-B.ST", "afry": "AFRY.ST",
    "trelleborg": "TREL-B.ST", "securitas": "SECU-B.ST", "essity": "ESSITY-B.ST",
    # Common index sleeves → liquid ETF proxies with a live Yahoo feed
    "msci world": "URTH", "msci world etf": "URTH", "world index": "URTH",
    "global index fund": "URTH", "global cap weighted index fund": "URTH",
    "s p 500": "SPY", "sp500": "SPY", "nasdaq 100": "QQQ", "omxs30": "XACT-OMXS30.ST",
}

# Lines whose (cleaned) first word is one of these are portfolio commentary,
# not holdings — cluster headers, exposure math, bear-case paragraphs.
_SKIP_FIRST_WORDS = {
    "cluster", "cash", "total", "portfolio", "bear", "bull", "kill", "verdict",
    "benchmark", "note", "notes", "warning", "summary", "exposure", "exposures",
    "weight", "weights", "correlation", "the", "this", "here", "sum",
}

_SKIP_WORDS = {"THE", "AND", "FOR", "WITH", "CASH", "TOTAL", "SUM", "NOTE", "NOTES",
               "TICKER", "SYMBOL", "WEIGHT", "STOCK", "STOCKS", "NAME", "PORTFOLIO"}

_TICKERISH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-^=]{0,14}$")


def _clean_name(text: str) -> str:
    """Normalise a company name for alias lookup: lowercase, drop punctuation/markdown."""
    text = re.sub(r"[*_`#]+", " ", text)          # markdown emphasis
    text = re.sub(r"[^A-Za-z0-9&]+", " ", text)   # punctuation → spaces
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_holdings(text: str) -> list[dict]:
    """Parse pasted picks into [{ticker|None, name, weight_pct|None, thesis, confidence}].

    Accepts a JSON array/object (as an LLM would return), plain "AAPL 12%"
    lines, or prose bullets like "- Nvidia 8% — chokepoint: CUDA lock-in"
    (company names are mapped to Yahoo tickers via NAME_ALIASES; anything
    unrecognised comes back with ticker=None for resolve_holdings / manual
    fixing in the preview). Raises ValueError when nothing usable is found.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("No stocks pasted.")

    holdings = _parse_json_holdings(text)
    if holdings is None:
        holdings = _parse_line_holdings(text)
    if not holdings:
        raise ValueError(
            "Could not find any tickers in the pasted text. "
            "Paste JSON with ticker/weight fields, or one 'TICKER weight%' per line."
        )
    return _dedupe(holdings)


def _dedupe(holdings: list[dict]) -> list[dict]:
    """De-duplicate on ticker (or name while unresolved), keeping first occurrence."""
    seen: set[str] = set()
    unique = []
    for h in holdings:
        if h.get("ticker"):
            h["ticker"] = h["ticker"].upper()
        key = h.get("ticker") or _clean_name(h.get("name") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        h.setdefault("name", h.get("ticker") or "")
        unique.append(h)
    return unique


def _extract_json_block(text: str) -> str | None:
    """Pull the first JSON array/object out of surrounding prose or ``` fences."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _parse_json_holdings(text: str) -> list[dict] | None:
    block = _extract_json_block(text)
    if not block:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                items = data[key]
                break
        else:
            # a single {ticker: weight} map?
            if data and all(isinstance(v, (int, float)) for v in data.values()):
                items = [{"ticker": k, "weight": v} for k, v in data.items()]
    if not items:
        return None

    holdings = []
    for item in items:
        if isinstance(item, str):
            holdings.append({"ticker": item.strip(), "weight_pct": None,
                             "thesis": "", "confidence": None})
            continue
        if not isinstance(item, dict):
            continue
        ticker = next((str(item[k]).strip() for k in _TICKER_KEYS if item.get(k)), "")
        name = next((str(item[k]).strip() for k in ("name", "company") if item.get(k)), "")
        if not ticker and not name:
            continue
        weight = next((item[k] for k in _WEIGHT_KEYS
                       if isinstance(item.get(k), (int, float, str)) and item.get(k) != ""), None)
        if isinstance(weight, str):
            m = _WEIGHT_RE.search(weight) or re.search(r"\d+(?:[.,]\d+)?", weight)
            weight = float(m.group(m.lastindex or 0).replace(",", ".")) if m else None
        thesis = next((str(item[k]).strip() for k in _THESIS_KEYS if item.get(k)), "")
        confidence = item.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        holdings.append({
            "ticker": ticker or None,
            "name": name or ticker,
            "weight_pct": float(weight) if weight is not None else None,
            "thesis": thesis,
            "confidence": confidence,
        })
    return holdings or None


def _parse_line_holdings(text: str) -> list[dict]:
    holdings = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "Alphabet 6% and Amazon 6%. …" — give each weighted name its own
        # segment. Only split when ≥2 parts open with "Name N%" (weight in the
        # first few words), so "CUDA + NVLink" or a "Kill: … >20%" tail in a
        # thesis never chops the line apart.
        segments = [line]
        if line.count("%") > 1:
            parts = re.split(r"\s+and\s+", line)
            with_leading_weight = sum(
                1 for p in parts if (m := _WEIGHT_RE.search(p)) and m.start() < 30
            )
            if with_leading_weight > 1:
                segments = parts
        for seg in segments:
            h = _parse_segment(seg)
            if h:
                holdings.append(h)
    return holdings


_BULLET_RE = re.compile(r"^[\s\-*•\d.)]+")


def _parse_segment(seg: str) -> dict | None:
    seg = seg.strip()
    if not seg:
        return None

    wm = _WEIGHT_RE.search(seg)
    if not wm:
        # No weight: only accept a bare single-token ticker line ("AAPL")
        tok = _BULLET_RE.sub("", seg).strip().strip(".,;:")
        if (_TICKERISH_RE.match(tok) and any(c.isalpha() for c in tok)
                and (tok.isupper() or tok.islower()) and tok.upper() not in _SKIP_WORDS):
            return {"ticker": tok.upper(), "name": tok.upper(),
                    "weight_pct": None, "thesis": "", "confidence": None}
        return None

    weight = float(wm.group(1).replace(",", "."))
    before = _BULLET_RE.sub(" ", seg[: wm.start()])
    name = _clean_name(before)
    if not name:
        return None
    words = name.split()
    if words[0] in _SKIP_FIRST_WORDS:
        return None       # cluster header / exposure math / commentary
    if len(words) > 5:
        return None       # prose sentence that happens to contain a percentage

    thesis = seg[wm.end():].strip(" .,;:—–-|*")

    # Resolve to a ticker: alias map first, then a lone ticker-looking token.
    # Mixed-case words ("Bufab") are company names — leave unresolved for
    # resolve_holdings (Yahoo symbol search) or manual fixing in the preview.
    ticker = _lookup_alias(name)
    display_name = re.sub(r"[*_`]+", "", before).strip(" .,;:—–-|")
    if ticker is None:
        tokens = [t.strip(".,;:*()") for t in before.split()]
        tokens = [t for t in tokens if t]
        if len(tokens) == 1 and _TICKERISH_RE.match(tokens[0]) and (
            tokens[0].isupper() or tokens[0].islower()
        ):
            if tokens[0].upper() in _SKIP_WORDS:
                return None
            ticker = tokens[0].upper()

    return {
        "ticker": ticker,
        "name": display_name or (ticker or ""),
        "weight_pct": weight,
        "thesis": thesis,
        "confidence": None,
    }


def _lookup_alias(cleaned_name: str) -> str | None:
    """NAME_ALIASES lookup, retrying with trailing words dropped
    ("micron technology inc" → "micron technology" → "micron")."""
    words = cleaned_name.split()
    while words:
        hit = NAME_ALIASES.get(" ".join(words))
        if hit:
            return hit
        words = words[:-1]
    return None


def resolve_holdings(holdings: list[dict], search: bool = True) -> list[dict]:
    """Fill in tickers for holdings parsed from company names.

    Alias map first; optionally Yahoo Finance symbol search for the rest
    (network — skip with search=False). Unresolvable entries keep ticker=None;
    callers decide whether to drop them or surface them for manual fixing.
    """
    for h in holdings:
        if h.get("ticker"):
            continue
        alias = _lookup_alias(_clean_name(h.get("name") or ""))
        if alias:
            h["ticker"] = alias
            continue
        if search and h.get("name"):
            sym = _search_symbol(h["name"])
            if sym:
                h["ticker"] = sym
                h["resolved_via"] = "search"
    return _dedupe(holdings)


def _search_symbol(query: str) -> str | None:
    """Best-effort Yahoo Finance symbol lookup for a company/fund name."""
    try:
        import yfinance as yf
        res = yf.Search(query, max_results=5)
        for q in res.quotes or []:
            sym = q.get("symbol")
            if sym and q.get("quoteType", "").upper() in ("EQUITY", "ETF"):
                return str(sym).upper()
    except Exception:
        pass
    return None


def normalise_weights(holdings: list[dict]) -> list[dict]:
    """Fill in missing weights and rescale so the total never exceeds 100%.

    - Weights that look like fractions (sum ≤ 1.5) are treated as 0–1 and scaled ×100.
    - Missing weights share whatever % the explicit ones leave unclaimed.
    - A total over 100% is scaled down proportionally; under 100% stays in cash.
    """
    given = [h["weight_pct"] for h in holdings if h["weight_pct"] is not None]
    if given and sum(given) <= 1.5 and all(w <= 1 for w in given):
        for h in holdings:
            if h["weight_pct"] is not None:
                h["weight_pct"] *= 100
        given = [w * 100 for w in given]

    total_given = sum(given)
    missing = [h for h in holdings if h["weight_pct"] is None]
    if missing:
        remainder = max(0.0, 100.0 - total_given)
        each = remainder / len(missing) if remainder else 0.0
        if each == 0.0:
            # explicit weights already claim everything — equal-weight the whole set
            each = 100.0 / len(holdings)
            for h in holdings:
                h["weight_pct"] = each
        else:
            for h in missing:
                h["weight_pct"] = each

    total = sum(h["weight_pct"] for h in holdings)
    if total > 100.0:
        scale = 100.0 / total
        for h in holdings:
            h["weight_pct"] = h["weight_pct"] * scale
    for h in holdings:
        h["weight_pct"] = round(h["weight_pct"], 2)
    return holdings


# ── Prices & currency ─────────────────────────────────────────────────────────

def detect_currency(ticker: str) -> str:
    """Trading currency for a Yahoo ticker — metadata first, suffix map fallback."""
    try:
        import yfinance as yf
        cur = yf.Ticker(ticker).fast_info["currency"]
        if cur:
            return str(cur)
    except Exception:
        pass
    if "." in ticker:
        return _SUFFIX_CURRENCY.get(ticker.rsplit(".", 1)[1].upper(), "USD")
    return "USD"


def to_sek_price(native_price: float, currency: str, store: Store | None = None) -> float | None:
    """Native quote → SEK. Handles LSE pence (GBp). None when no FX rate."""
    from fundmgr.data.fx import rate_to_sek
    cur = currency or "USD"
    if cur in ("GBp", "GBX"):  # London quotes in pence
        rate = rate_to_sek("GBP", store)
        return native_price / 100.0 * rate if rate else None
    rate = rate_to_sek(cur.upper(), store)
    return native_price * rate if rate else None


def sek_prices_for(store: Store, tickers: list[str], currency_map: dict[str, str],
                   native: dict[str, float]) -> dict[str, float]:
    out = {}
    for t in tickers:
        p = native.get(t)
        if p is None:
            continue
        sek = to_sek_price(p, currency_map.get(t, "USD"), store)
        if sek is not None:
            out[t] = sek
    return out


def _cache_price_history(store: Store, tickers: list[str], lookback_days: int = 40) -> None:
    """Mirror recent daily closes into the store's price cache (native currency),
    so the evaluator can pin outcomes to decision_date + 28d closes."""
    from datetime import timedelta

    import yfinance as yf
    since = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=since, auto_adjust=True)
            if hist is None or hist.empty:
                continue
            rows = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(r["Open"]), "high": float(r["High"]),
                    "low": float(r["Low"]), "close": float(r["Close"]),
                    "volume": float(r.get("Volume") or 0),
                }
                for idx, r in hist.iterrows()
            ]
            if rows:
                store.save_prices(t, rows)
        except Exception:
            continue


# ── Creation ──────────────────────────────────────────────────────────────────

def create_portfolio(
    name: str,
    capital_sek: float,
    holdings_text: str,
    base_prompt: str = "",
    model_label: str = "",
    benchmark: str = DEFAULT_BENCHMARK,
    holdings_override: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """Create a paper portfolio and execute its initial buys at live prices.

    holdings_override (e.g. the user-edited preview table) bypasses text
    parsing; holdings_text is still stored as the pasted record either way.

    Returns (slug, log_lines). Raises ValueError on bad input (name taken,
    nothing parseable, no prices available).
    """
    name = name.strip()
    if not name:
        raise ValueError("Give the portfolio a name.")
    if capital_sek <= 0:
        raise ValueError("Starting value must be positive.")

    slug = slugify(name)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    db_path = PAPER_DIR / f"{slug}.db"
    if db_path.exists():
        raise ValueError(f"A paper portfolio named '{slug}' already exists.")

    if holdings_override:
        holdings = _dedupe([
            {
                "ticker": str(h.get("ticker") or "").strip().upper() or None,
                "name": str(h.get("name") or h.get("ticker") or "").strip(),
                "weight_pct": (float(h["weight_pct"])
                               if h.get("weight_pct") not in (None, "") else None),
                "thesis": str(h.get("thesis") or "").strip(),
                "confidence": _safe_confidence(h.get("confidence")),
            }
            for h in holdings_override
        ])
        if not holdings:
            raise ValueError("The edited holdings table is empty.")
    else:
        holdings = parse_holdings(holdings_text)

    holdings = resolve_holdings(holdings)
    log: list[str] = []
    unresolved = [h for h in holdings if not h.get("ticker")]
    holdings = [h for h in holdings if h.get("ticker")]
    for h in unresolved:
        log.append(f"⚠ could not resolve '{h['name']}' to a ticker — skipped "
                   "(its allocation stays in cash)")
    if not holdings:
        raise ValueError(
            "None of the pasted names could be resolved to a Yahoo Finance ticker. "
            "Use the preview to enter tickers manually."
        )

    holdings = normalise_weights(holdings)
    tickers = [h["ticker"] for h in holdings]

    # Live native-currency prices + per-ticker currency
    from fundmgr.data.quotes import live_prices
    native = {t: p for t, p in live_prices(tickers).items() if p}
    currency_map = {t: detect_currency(t) for t in tickers if t in native}

    priced = [h for h in holdings if h["ticker"] in native]
    unpriced = [h["ticker"] for h in holdings if h["ticker"] not in native]
    if not priced:
        raise ValueError(
            "Could not fetch a market price for any pasted ticker "
            f"({', '.join(tickers[:10])}). Are they valid Yahoo Finance symbols?"
        )
    for t in unpriced:
        log.append(f"⚠ {t}: no market price — skipped (allocation stays in cash)")

    store = Store(db_path)
    try:
        store.initialise(capital_sek)
        now = datetime.now(timezone.utc)
        store.set_meta("paper_name", name)
        store.set_meta("paper_capital_sek", str(capital_sek))
        store.set_meta("paper_benchmark", benchmark)
        store.set_meta("paper_model_label", model_label.strip())
        store.set_meta("paper_created_at", now.strftime("%Y-%m-%d %H:%M"))
        store.set_meta("paper_base_prompt", base_prompt.strip())
        store.set_meta("paper_currency_map", json.dumps(currency_map))
        store.set_meta("paper_pasted_text", holdings_text.strip())

        prices_sek = sek_prices_for(store, [h["ticker"] for h in priced], currency_map, native)

        # Execute the initial buys: gross = capital × weight, fee comes out of gross
        actions = []
        for h in priced:
            t = h["ticker"]
            price_sek = prices_sek.get(t)
            if not price_sek:
                log.append(f"⚠ {t}: no FX rate for {currency_map.get(t)} — skipped")
                continue
            gross = capital_sek * h["weight_pct"] / 100.0
            fee = _fees.calc(gross)
            shares = round((gross - fee) / price_sek, 4)
            if shares <= 0:
                log.append(f"⚠ {t}: allocation too small for one lot — skipped")
                continue
            store.apply_fill(Transaction(
                ticker=t, side="buy", shares=shares,
                price_sek=round(price_sek, 4), fee_sek=round(fee, 2),
                source="paper", currency=currency_map.get(t, "USD"), timestamp=now,
            ))
            actions.append({
                "ticker": t, "side": "buy",
                "target_weight_pct": h["weight_pct"],
                "confidence": h["confidence"],
                "thesis": h["thesis"],
                "sek_estimate": round(gross),
                "stop_loss_pct": None,
            })
            log.append(f"✓ Bought {shares:g} × {t} @ {price_sek:,.2f} SEK (fee {fee:.0f})")

        if not actions:
            raise ValueError("No positions could be opened — see the skip reasons above.")

        # Record the creation as this portfolio's one decision run, so the
        # standard evaluator/learnings pipeline picks it up like any fund run.
        run_id = f"paper-{slug}-{now.strftime('%Y%m%d%H%M%S')}"
        actions_json = json.dumps(actions)
        store.save_recommendation(RecommendationLog(
            run_id=run_id,
            timestamp=now,
            prompt_snapshot=json.dumps({
                "base_prompt": base_prompt.strip(),
                "pasted_holdings": holdings_text.strip(),
                "prices": {t: {"last_price": native[t]} for t in native},
                "currency_map": currency_map,
            }),
            llm_response=json.dumps({
                "market_summary": f"Paper portfolio '{name}' seeded from "
                                  f"{model_label.strip() or 'pasted'} picks at live market prices.",
                "notes": f"{len(actions)} positions · benchmark {benchmark}"
                         + "".join(f"\n{line}" for line in log if line.startswith("⚠")),
            }),
            guardrail_log="{}",
            actions_json=actions_json,
        ))
        store.seed_outcomes_for_run(run_id, actions_json, prices=native)

        # Benchmark + price-history caches so tracking/evaluation have data
        from fundmgr.data.benchmark import fetch_and_cache_benchmark
        fetch_and_cache_benchmark(store, symbol=benchmark)
        _cache_price_history(store, list(native.keys()))

        bench_rows = store.get_benchmark()
        store.upsert_nav(NavPoint(
            date=now.strftime("%Y-%m-%d"),
            portfolio_nav_sek=_cost_nav(store),
            benchmark_value=bench_rows[-1]["close"] if bench_rows else 0.0,
            cash_sek=store.get_cash(),
        ))
    except Exception:
        # Don't leave a half-created portfolio behind
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    log.append(f"✓ Portfolio '{name}' created — {len(actions)} positions, "
               f"{store.get_cash():,.0f} SEK cash")
    return slug, log


def _cost_nav(store: Store) -> float:
    return sum(p.shares * p.avg_cost_sek for p in store.get_positions()) + store.get_cash()


def _safe_confidence(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ── Daily tracking + learning ─────────────────────────────────────────────────

def track_portfolio(slug: str) -> list[str]:
    """Daily upkeep for one portfolio: refresh caches, snapshot NAV at live SEK
    prices, evaluate matured outcomes, distil learnings — same pipeline as the
    other funds, scoped to this portfolio's own store."""
    meta, store = open_portfolio(slug)
    log = [f"── {meta['name']} ({slug}) ──"]

    positions = store.get_positions()
    tickers = [p.ticker for p in positions]

    from fundmgr.data.benchmark import fetch_and_cache_benchmark
    fetch_and_cache_benchmark(store, symbol=meta["benchmark"])
    if tickers:
        _cache_price_history(store, tickers, lookback_days=10)

    # NAV at latest cached close (native → SEK); cost basis fallback per position
    native = {}
    for t in tickers:
        rows = store.get_prices(t)
        if rows:
            native[t] = rows[-1]["close"]
    prices_sek = sek_prices_for(store, tickers, meta["currency_map"], native)
    nav = sum(p.shares * prices_sek.get(p.ticker, p.avg_cost_sek) for p in positions) + store.get_cash()

    bench_rows = store.get_benchmark()
    store.upsert_nav(NavPoint(
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        portfolio_nav_sek=round(nav, 2),
        benchmark_value=bench_rows[-1]["close"] if bench_rows else 0.0,
        cash_sek=store.get_cash(),
    ))
    log.append(f"NAV {nav:,.0f} SEK ({len(positions)} positions)")

    # Learning pipeline — identical to the weekly fund runs
    from fundmgr.engine.evaluator import (
        evaluate_pending_outcomes,
        generate_learnings,
        generate_qualitative_learnings,
    )
    evaluated = evaluate_pending_outcomes(store)
    if evaluated:
        stat = generate_learnings(store)
        qual = generate_qualitative_learnings(store, evaluated)
        log.append(f"Evaluated {len(evaluated)} outcomes → "
                   f"{len(stat)} calibration + {len(qual)} qualitative learnings")
    return log


def track_all() -> list[str]:
    log: list[str] = []
    for meta in list_portfolios():
        try:
            log += track_portfolio(meta["slug"])
        except Exception as e:  # keep going — one broken portfolio shouldn't stop the rest
            log.append(f"⚠ {meta['slug']}: {e}")
    if not log:
        log.append("No paper portfolios yet.")
    return log
