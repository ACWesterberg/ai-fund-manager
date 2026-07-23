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
import os
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


def list_portfolios(kind: str | None = None) -> list[dict]:
    """Portfolios' metadata (no price fetches). `kind` filters by book type:
    'paper' (pasted sims) or 'live' (real monitored sleeves); None = all."""
    if not PAPER_DIR.exists():
        return []
    out = []
    for db in sorted(PAPER_DIR.glob("*.db")):
        try:
            meta, _store = open_portfolio(db.stem)
        except Exception:
            continue
        if kind is None or meta.get("kind") == kind:
            out.append(meta)
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
        # 'paper' = pasted sim (default, back-compat); 'live' = real monitored sleeve
        "kind": store.get_meta("paper_kind") or "paper",
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
_KILL_KEYS = ("kill_criterion", "kill_criteria", "kill", "sell_trigger", "sell_criterion")
_LIST_KEYS = ("portfolio", "holdings", "positions", "stocks", "picks", "actions",
              "recommendations", "allocations")

# Exchange labels (as LLMs write them) → Yahoo suffix, for JSON answers that
# give broker-style tickers like {"ticker": "ATCO A", "exchange": "OMX"}.
_EXCHANGE_SUFFIX = {
    "OMX": ".ST", "OMXS": ".ST", "STO": ".ST", "STOCKHOLM": ".ST",
    "OSL": ".OL", "OSE": ".OL", "OSLO": ".OL",
    "CPH": ".CO", "OMXC": ".CO", "COPENHAGEN": ".CO",
    "HEL": ".HE", "OMXH": ".HE", "HELSINKI": ".HE",
    "AMS": ".AS", "AEX": ".AS", "AMSTERDAM": ".AS",
    "XETRA": ".DE", "FRA": ".DE", "GER": ".DE", "ETR": ".DE",
    "LSE": ".L", "LON": ".L", "LONDON": ".L",
    "EPA": ".PA", "PAR": ".PA", "PARIS": ".PA",
    "MIL": ".MI", "BIT": ".MI", "MTA": ".MI",
    "BME": ".MC", "MCE": ".MC", "MADRID": ".MC",
    "SIX": ".SW", "SWX": ".SW", "EBS": ".SW",
    "TSE": ".T", "TYO": ".T", "TOKYO": ".T",
    "TSX": ".TO", "TOR": ".TO", "ASX": ".AX",
    "HKEX": ".HK", "HKG": ".HK",
}
_US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "OTC", "CBOE", "US"}

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
    "kongsberg gruppen": "KOG.OL", "be semiconductor": "BESI.AS",
    "be semiconductor industries": "BESI.AS", "besi": "BESI.AS",
    "sk hynix": "SKHY", "sk hynix inc": "SKHY", "vertiv holdings": "VRT",
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


def _join_kill(value) -> str:
    """Kill criteria arrive as a string or a JSON list (e.g. the Fable format's
    `"kill_criteria": ["A", "B"]`). Join lists into one readable clause so the
    news judge reads them cleanly instead of a Python list repr."""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


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
            holdings.append({"ticker": item.strip(), "name": item.strip(),
                             "weight_pct": None, "thesis": "", "confidence": None})
            continue
        if not isinstance(item, dict):
            continue
        ticker = next((str(item[k]).strip() for k in _TICKER_KEYS if item.get(k)), "")
        name = next((str(item[k]).strip() for k in ("name", "company") if item.get(k)), "")
        if not ticker and not name:
            continue

        # Cash rows carry no holding — their weight simply stays uninvested
        cluster = str(item.get("cluster") or "").strip()
        if (ticker.upper() == "CASH" or cluster.lower() == "cash"
                or _clean_name(name).startswith("cash")):
            continue

        exchange = str(item.get("exchange") or "").strip().upper()
        ticker = _normalise_json_ticker(ticker, exchange, name)

        weight = next((item[k] for k in _WEIGHT_KEYS
                       if isinstance(item.get(k), (int, float, str)) and item.get(k) != ""), None)
        if isinstance(weight, str):
            m = _WEIGHT_RE.search(weight) or re.search(r"\d+(?:[.,]\d+)?", weight)
            weight = float(m.group(m.lastindex or 0).replace(",", ".")) if m else None
        thesis = next((str(item[k]).strip() for k in _THESIS_KEYS if item.get(k)), "")
        kill_val = next((item[k] for k in _KILL_KEYS
                         if item.get(k) not in (None, "", [])), "")
        kill = _join_kill(kill_val)
        confidence = _safe_confidence(item.get("confidence"))
        holdings.append({
            "ticker": ticker,
            "name": name or ticker,
            "weight_pct": float(weight) if weight is not None else None,
            "thesis": thesis,
            "kill_criterion": kill,
            "cluster": cluster,
            "confidence": confidence,
        })
    return holdings or None


def _normalise_json_ticker(ticker: str, exchange: str, name: str) -> str | None:
    """Turn a broker-style JSON ticker into a Yahoo symbol.

    'ATCO A' + OMX → ATCO-A.ST; 'MTRS' + OMX → MTRS.ST (bare 'MTRS' is a
    different NYSE company on Yahoo); fund placeholders like INDEX_GLOBAL
    return None so resolve_holdings can map the *name* to an ETF proxy.
    """
    if not ticker:
        return None
    if "." in ticker:                     # already a Yahoo-style symbol
        return ticker
    if not _TICKERISH_RE.match(ticker.replace(" ", "-")) or exchange == "FUND":
        return None                       # INDEX_GLOBAL etc. — resolve by name
    if exchange in _EXCHANGE_SUFFIX:
        return ticker.replace(" ", "-") + _EXCHANGE_SUFFIX[exchange]
    if exchange in _US_EXCHANGES:
        return ticker
    # No usable exchange hint: prefer the name's alias when it points to a
    # suffixed home listing ('Munters'/'MTRS' → MTRS.ST), else keep as pasted.
    alias = _lookup_alias(_clean_name(name)) if name else None
    if alias and "." in alias:
        return alias
    return ticker.replace(" ", "-")


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
            return {"ticker": tok.upper(), "name": tok.upper(), "weight_pct": None,
                    "thesis": "", "kill_criterion": "", "confidence": None}
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
    # Prose theses often pre-register the falsification condition inline
    # ("… lock-in. Kill: hyperscaler custom silicon taking >20% …")
    kill = ""
    km = re.search(r"\bKill(?:\s+criteri\w+)?\s*[:—-]\s*(.+)$", thesis, re.IGNORECASE)
    if km:
        kill = km.group(1).strip()
        thesis = thesis[: km.start()].strip(" .,;:—–-|*")

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
        "kill_criterion": kill,
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


# ── Structured import (broker tickers → Yahoo) ────────────────────────────────

# Broker/Montrose-style tickers → Yahoo symbols, for `fund paper-import` of a
# structured answer that quotes broker tickers rather than Yahoo symbols. Bare
# US symbols (TSM, NVDA, GOOGL, GEV, VRT, CEG) are already valid and pass
# through unchanged; only the ones that mis-resolve on Yahoo are mapped.
MONTROSE_YAHOO = {
    "KOG": "KOG.OL",       # Kongsberg Gruppen (bare KOG = Kodiak Gas on Yahoo)
    "ENR": "ENR.DE",       # Siemens Energy (Xetra)
    "BESI": "BESI.AS",     # BE Semiconductor (Euronext Amsterdam)
    "ASML": "ASML.AS",     # ASML — Amsterdam listing (EUR) to match the sleeve
    # SK Hynix trades as SKHY on NasdaqGS (USD) — a valid Yahoo feed, so the
    # bare ticker passes through unchanged; no mapping needed.
}


def montrose_to_yahoo(ticker: str, currency: str | None = None,
                      name: str | None = None) -> str:
    """Map a broker/Montrose ticker to a Yahoo symbol. Explicit map first, then
    an already-suffixed symbol as-is, then the company name's home-listing
    alias, else the bare ticker (correct for US names)."""
    t = (ticker or "").strip().upper()
    if t in MONTROSE_YAHOO:
        return MONTROSE_YAHOO[t]
    if "." in t:
        return t
    alias = _lookup_alias(_clean_name(name or "")) if name else None
    if alias and "." in alias:
        return alias
    return t


def parse_structured_portfolio(data: dict) -> dict:
    """Turn a structured LLM answer (positions[] with broker tickers, cluster,
    kill_criteria, next_earnings, plus a top-level portfolio_kill_criterion and
    meta.deployable_capital_sek) into the pieces create_portfolio consumes.

    Broker tickers are mapped to Yahoo symbols; `excluded_holdings` are dropped
    entirely (never bought, never sized). Returns a dict with holdings_override,
    position_meta, capex_kill, capital_sek, name and excluded.
    """
    holdings: list[dict] = []
    position_meta: dict[str, dict] = {}
    for p in data.get("positions") or []:
        raw = str(p.get("ticker") or "").strip()
        if not raw:
            continue
        yt = montrose_to_yahoo(raw, p.get("currency"), p.get("name"))
        holdings.append({
            "ticker": yt,
            "name": p.get("name") or yt,
            "weight_pct": p.get("target_weight_pct"),
            "thesis": p.get("thesis") or "",
            "kill_criterion": _join_kill(p.get("kill_criteria")
                                         or p.get("kill_criterion") or ""),
            "cluster": p.get("cluster") or "",
            "confidence": None,
        })
        position_meta[yt] = {
            "watch": p.get("watch") or "",
            "thesis": p.get("thesis") or "",
            "bear_case": p.get("bear_case") or "",
            "next_earnings": p.get("next_earnings") or "",
        }

    capex = data.get("portfolio_kill_criterion") or {}
    capex_kill: dict = {}
    if (capex.get("trigger") or "").strip():
        capex_kill = {
            "trigger": capex.get("trigger", ""),
            "action": capex.get("action", ""),
            "note": capex.get("note", ""),
            "hyperscalers": capex.get("hyperscalers") or DEFAULT_HYPERSCALERS,
        }

    meta = data.get("meta") or {}
    excluded = [
        (e.get("ticker") or e.get("name") or "").strip()
        for e in (meta.get("excluded_holdings") or [])
    ]
    return {
        "holdings_override": holdings,
        "position_meta": position_meta,
        "capex_kill": capex_kill,
        "capital_sek": meta.get("deployable_capital_sek") or data.get("capital_sek"),
        "name": meta.get("name") or data.get("name") or "Imported portfolio",
        "excluded": [e for e in excluded if e],
    }


# ── Creation ──────────────────────────────────────────────────────────────────

def create_portfolio(
    name: str,
    capital_sek: float,
    holdings_text: str,
    base_prompt: str = "",
    model_label: str = "",
    benchmark: str = DEFAULT_BENCHMARK,
    holdings_override: list[dict] | None = None,
    position_meta: dict[str, dict] | None = None,
    capex_kill: dict | None = None,
    kind: str = "paper",
    execute_buys: bool = True,
) -> tuple[str, list[str]]:
    """Create a portfolio from a set of picks.

    holdings_override (e.g. the user-edited preview table) bypasses text
    parsing; holdings_text is still stored as the pasted record either way.

    position_meta ({ticker: {watch, thesis, bear_case, next_earnings}}) and
    capex_kill (a portfolio-level kill criterion + the hyperscaler ticker set)
    are optional extras the monitors quote — populated by `fund paper-import`
    from a structured answer; unused by the plain web-paste path.

    execute_buys=True (paper sims) opens every position at live prices now.
    execute_buys=False (real 'live' sleeves) imports the *plan* only — tickers,
    weights, kill criteria and notes are stored and monitored, but nothing is
    bought; positions appear as you record actual fills (`fund paper-fill` /
    Telegram screenshot). Cash stays whole.

    Returns (slug, log_lines). Raises ValueError on bad input (name taken,
    nothing parseable, or — when executing — no prices available).
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
                "kill_criterion": str(h.get("kill_criterion") or "").strip(),
                "cluster": str(h.get("cluster") or "").strip(),
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

    # Per-ticker trading currency (resolves without a live price via yfinance
    # metadata / suffix map) — needed for later fills and SEK drift math.
    currency_map = {t: detect_currency(t) for t in tickers}

    # Live native-currency prices: used to execute buys and seed the price
    # cache. In plan-only mode a missing price just means no opening fill.
    from fundmgr.data.quotes import live_prices
    native = {t: p for t, p in live_prices(tickers).items() if p}

    priced = [h for h in holdings if h["ticker"] in native]
    if execute_buys:
        if not priced:
            raise ValueError(
                "Could not fetch a market price for any pasted ticker "
                f"({', '.join(tickers[:10])}). Are they valid Yahoo Finance symbols?"
            )
        for t in [h["ticker"] for h in holdings if h["ticker"] not in native]:
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
        store.set_meta("paper_kind", kind if kind in ("paper", "live") else "paper")
        store.set_meta("paper_base_prompt", base_prompt.strip())
        store.set_meta("paper_currency_map", json.dumps(currency_map))
        store.set_meta("paper_pasted_text", holdings_text.strip())
        # The plan — kill criteria, target weights, per-position notes — covers
        # every resolved holding (bought or not yet), so the monitors and the
        # Watch panel work from the plan rather than only from executed fills.
        store.set_meta("paper_kill_criteria", json.dumps({
            h["ticker"]: (h.get("kill_criterion") or "").strip()
            for h in holdings if (h.get("kill_criterion") or "").strip()
        }))
        store.set_meta("paper_target_weights", json.dumps({
            h["ticker"]: round(float(h["weight_pct"]), 4) for h in holdings
        }))
        if position_meta:
            store.set_meta("paper_position_notes", json.dumps({
                t: v for t, v in position_meta.items() if t in set(tickers)
            }))
        if capex_kill:
            store.set_meta("paper_capex_kill", json.dumps(capex_kill))

        def _plan_thesis(h: dict) -> tuple[str, str]:
            kill = (h.get("kill_criterion") or "").strip()
            thesis = h["thesis"]
            if kill and "kill" not in thesis.lower():
                thesis = f"{thesis} · Kill: {kill}" if thesis else f"Kill: {kill}"
            return thesis, kill

        actions: list[dict] = []
        if execute_buys:
            # Open every priced position now: gross = capital × weight, fee off gross
            prices_sek = sek_prices_for(store, [h["ticker"] for h in priced], currency_map, native)
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
                thesis, kill = _plan_thesis(h)
                actions.append({
                    "ticker": t, "side": "buy", "target_weight_pct": h["weight_pct"],
                    "confidence": h["confidence"], "thesis": thesis, "kill_criterion": kill,
                    "cluster": h.get("cluster") or "", "sek_estimate": round(gross),
                    "stop_loss_pct": None,
                })
                log.append(f"✓ Bought {shares:g} × {t} @ {price_sek:,.2f} SEK (fee {fee:.0f})")
            if not actions:
                raise ValueError("No positions could be opened — see the skip reasons above.")
        else:
            # Plan only: record the intended trades without executing. Cash stays
            # whole; positions appear as fills are recorded later.
            for h in holdings:
                thesis, kill = _plan_thesis(h)
                actions.append({
                    "ticker": h["ticker"], "side": "buy", "target_weight_pct": h["weight_pct"],
                    "confidence": h["confidence"], "thesis": thesis, "kill_criterion": kill,
                    "cluster": h.get("cluster") or "",
                    "sek_estimate": round(capital_sek * h["weight_pct"] / 100.0),
                    "stop_loss_pct": None,
                })

        # Record the creation as this portfolio's decision run (the plan), so the
        # dashboard and — for executed books — the learnings pipeline pick it up.
        run_id = f"paper-{slug}-{now.strftime('%Y%m%d%H%M%S')}"
        actions_json = json.dumps(actions)
        summary = (
            f"Paper portfolio '{name}' seeded from {model_label.strip() or 'pasted'} "
            f"picks at live market prices."
            if execute_buys else
            f"Live sleeve '{name}' — plan imported from {model_label.strip() or 'pasted'} "
            f"picks; positions fill as you record trades."
        )
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
                "market_summary": summary,
                "notes": f"{len(actions)} {'positions' if execute_buys else 'intended positions'}"
                         f" · benchmark {benchmark}"
                         + "".join(f"\n{line}" for line in log if line.startswith("⚠")),
            }),
            guardrail_log="{}",
            actions_json=actions_json,
        ))
        if execute_buys:
            store.seed_outcomes_for_run(run_id, actions_json, prices=native)

        # Benchmark + price-history caches so tracking/evaluation have data
        from fundmgr.data.benchmark import fetch_and_cache_benchmark
        fetch_and_cache_benchmark(store, symbol=benchmark)
        if native:
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

    if execute_buys:
        log.append(f"✓ Portfolio '{name}' created — {len(actions)} positions, "
                   f"{store.get_cash():,.0f} SEK cash")
    else:
        log.append(f"✓ Plan '{name}' imported — {len(actions)} intended positions, "
                   f"no fills yet ({store.get_cash():,.0f} SEK cash). "
                   f"Record trades to build the book.")
    return slug, log


def _cost_nav(store: Store) -> float:
    return sum(p.shares * p.avg_cost_sek for p in store.get_positions()) + store.get_cash()


def _safe_confidence(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ── Fill ticker resolution ────────────────────────────────────────────────────

def plan_tickers(store: Store) -> set[str]:
    """Every Yahoo symbol this book knows: plan targets, its currency map, and
    anything already held. Used to snap a fill ticker onto the plan symbol."""
    out = set(json.loads(store.get_meta("paper_target_weights") or "{}"))
    out |= set(json.loads(store.get_meta("paper_currency_map") or "{}"))
    out |= {p.ticker for p in store.get_positions()}
    return out


def snap_ticker_to_plan(store: Store, ticker: str) -> tuple[str, str | None]:
    """Resolve a user-entered fill ticker to this book's plan symbol.

    A broker/bare symbol is snapped to the plan's Yahoo symbol when there's a
    single plan ticker with the same base (e.g. 'ASML' → 'ASML.AS', 'ENR' →
    'ENR.DE'), so a fill lands on the intended, correctly-priced instrument
    instead of a mismatched one. Returns (resolved_ticker, note) — note is a
    short human message when a snap happened or the ticker isn't in the plan.
    """
    t = (ticker or "").strip().upper()
    plan = plan_tickers(store)
    if t in plan:
        return t, None
    base = t.split(".")[0]
    matches = sorted(p for p in plan if p.split(".")[0] == base)
    if len(matches) == 1:
        return matches[0], f"→ matched to plan symbol {matches[0]}"
    if not plan:
        return t, None  # nothing to match against (e.g. a fresh book)
    return t, f"⚠ {t} isn't in this book's plan — recording as-is"


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

    # Kill-criterion watch: check fresh headlines against each position's
    # pre-registered falsification condition
    try:
        log += check_kill_criteria(slug, store=store)
    except Exception as e:
        log.append(f"⚠ kill-criterion watch failed: {e}")

    # Portfolio-level capex kill criterion (e.g. "2 of 5 hyperscalers guide
    # 2027 capex flat/down"), the master trigger for the whole sleeve.
    try:
        log += check_capex_kill(slug, store=store)
    except Exception as e:
        log.append(f"⚠ capex-kill watch failed: {e}")

    # Earnings-calendar heads-up before each print and a check-the-numbers
    # reminder after — the decision moments the whole schedule turns on.
    try:
        log += check_earnings_calendar(slug, store=store)
    except Exception as e:
        log.append(f"⚠ earnings watch failed: {e}")

    # Weight drift: alert when a position appreciates past 1.5× its target.
    try:
        log += check_weight_drift(slug, store=store)
    except Exception as e:
        log.append(f"⚠ drift watch failed: {e}")

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


# ── Kill-criterion news watch ─────────────────────────────────────────────────

def check_kill_criteria(slug: str, store: Store | None = None) -> list[str]:
    """Judge each held position's recent headlines against its pre-registered
    kill criterion (the observable fact that would falsify the thesis).

    Once per ticker per day. A hit is logged, recorded in app_meta, and sent
    to Telegram (no-op when the bot env vars are unset). Judgement uses
    gpt-4o-mini — skipped without OPENAI_API_KEY.
    """
    if store is None:
        meta, store = open_portfolio(slug)
        name = meta["name"]
    else:
        name = store.get_meta("paper_name") or slug

    kills = json.loads(store.get_meta("paper_kill_criteria") or "{}")
    # Watch the plan, not just what's been bought: a thesis can break before the
    # position is opened (held ∪ target-weight tickers).
    watch_tickers = {p.ticker for p in store.get_positions()} | set(
        json.loads(store.get_meta("paper_target_weights") or "{}"))
    kills = {t: k for t, k in kills.items() if t in watch_tickers and k}
    if not kills:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return ["kill-criterion watch skipped (no OPENAI_API_KEY)"]

    log: list[str] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hits: list[tuple[str, str]] = []
    for ticker, criterion in kills.items():
        if store.get_meta(f"paper_killwatch:{ticker}") == today:
            continue  # already checked today
        headlines = _recent_headlines(ticker)
        store.set_meta(f"paper_killwatch:{ticker}", today)
        if not headlines:
            continue
        verdict = _judge_kill_hit(ticker, criterion, headlines)
        if verdict:
            hits.append((ticker, verdict))
            store.set_meta(f"paper_killhit:{ticker}:{today}", verdict)
            log.append(f"🚨 {ticker}: kill criterion may be triggering — {verdict}")

    if hits:
        from fundmgr.notify.send import send_telegram
        lines = [f"<b>📋 Paper portfolio — {name}</b>",
                 "🚨 Kill-criterion watch: recent news may falsify a thesis"]
        for ticker, verdict in hits:
            lines.append(f"\n<b>{ticker}</b>: {verdict}")
            lines.append(f"  Pre-registered kill: {kills[ticker]}")
        lines.append("\nVerify before acting — headline-level signal only.")
        send_telegram("\n".join(lines))
    return log


def _recent_headlines(ticker: str, max_items: int = 8) -> list[str]:
    """Recent news headlines for a ticker via yfinance. Empty list on failure."""
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    titles: list[str] = []
    for item in items[: max_items * 2]:
        # yfinance has shipped two shapes: flat {'title': …} and nested {'content': {'title': …}}
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (content.get("title") or "").strip()
        publisher = ""
        prov = content.get("provider")
        if isinstance(prov, dict):
            publisher = prov.get("displayName") or ""
        publisher = publisher or item.get("publisher") or ""
        if title:
            titles.append(f"{title}" + (f" ({publisher})" if publisher else ""))
        if len(titles) >= max_items:
            break
    return titles


def _judge_kill_hit(ticker: str, criterion: str, headlines: list[str]) -> str | None:
    """Ask gpt-4o-mini whether the headlines plausibly trigger the kill criterion.

    Returns a one-line reason on a hit, None otherwise (including on any API
    failure — the watch must never break tracking)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You monitor pre-registered kill criteria (thesis-falsification "
                    "conditions) for stock positions. Given a criterion and recent "
                    "headlines, decide if any headline plausibly indicates the "
                    "criterion is being met. Be strict: general negativity, price "
                    "moves, or unrelated bad news are NOT a hit — only concrete "
                    "events matching the stated criterion. Reply with exactly "
                    "'NO' or 'YES: <one-sentence reason citing the headline>'."
                )},
                {"role": "user", "content": (
                    f"Position: {ticker}\n"
                    f"Kill criterion: {criterion}\n\n"
                    "Recent headlines:\n" + "\n".join(f"- {h}" for h in headlines)
                )},
            ],
            max_tokens=120,
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if answer.upper().startswith("YES"):
            return answer.split(":", 1)[1].strip() if ":" in answer else answer
        return None
    except Exception:
        return None


# ── Portfolio-level capex kill criterion ──────────────────────────────────────

# Five largest hyperscalers by AI capex — the default watch set when a portfolio
# stores a capex kill criterion without naming its own tickers.
DEFAULT_HYPERSCALERS = ["MSFT", "AMZN", "GOOGL", "META", "ORCL"]

# How far back a flat/down capex signal still counts toward the trigger — one
# earnings season. The prints that resolve the criterion cluster within a
# fortnight, so a ~7-week window comfortably spans a single reporting round.
_CAPEX_WINDOW_DAYS = 49


def check_capex_kill(slug: str, store: Store | None = None) -> list[str]:
    """Watch the largest hyperscalers' capex guidance — the sleeve's master kill
    criterion (e.g. "any two guide 2027 capex flat or down").

    For each hyperscaler, judge recent headlines for a flat/down 2027 capex
    signal (once per ticker per day). Count distinct flat/down names in a
    rolling earnings-season window: one is a warning to resize, two or more
    fires the portfolio kill alert quoting the pre-registered action. Each
    threshold alerts once. Judgement uses gpt-4o-mini — skipped without
    OPENAI_API_KEY. No-op on portfolios without a stored capex criterion.
    """
    if store is None:
        meta, store = open_portfolio(slug)
        name = meta["name"]
    else:
        name = store.get_meta("paper_name") or slug

    cfg = json.loads(store.get_meta("paper_capex_kill") or "{}")
    if not cfg or not (cfg.get("trigger") or "").strip():
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return ["capex-kill watch skipped (no OPENAI_API_KEY)"]

    hyperscalers = cfg.get("hyperscalers") or DEFAULT_HYPERSCALERS
    log: list[str] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for ticker in hyperscalers:
        if store.get_meta(f"paper_capexwatch:{ticker}:{today}") == today:
            continue
        headlines = _recent_headlines(ticker)
        store.set_meta(f"paper_capexwatch:{ticker}:{today}", today)
        if not headlines:
            continue
        signal = _judge_capex_signal(ticker, headlines)
        if signal:  # "down" | "flat"
            store.set_meta(f"capex_signal:{ticker}:{today}", signal)
            log.append(f"⚠ {ticker}: 2027 capex guidance reads {signal.upper()}")

    flagged = _recent_capex_signals(store, _CAPEX_WINDOW_DAYS)
    count = len(flagged)
    prior = store.get_meta("paper_capex_status") or "none"
    status = "triggered" if count >= 2 else "warning" if count == 1 else "none"

    trigger = cfg.get("trigger", "")
    action = cfg.get("action", "")
    if status == "triggered" and prior != "triggered":
        names = ", ".join(f"{t} ({s})" for t, s in sorted(flagged.items()))
        _send_capex_alert(
            name, level="🚨 KILL CRITERION TRIGGERED",
            body=[f"<b>{count} hyperscalers</b> now guiding 2027 capex flat/down: {names}",
                  f"\nTrigger: {trigger}",
                  f"Action: <b>{action}</b>" if action else ""],
        )
        log.append(f"🚨 capex kill criterion TRIGGERED — {count} hyperscalers flat/down")
    elif status == "warning" and prior == "none":
        t, s = next(iter(flagged.items()))
        _send_capex_alert(
            name, level="⚠ Capex warning (1 of 2)",
            body=[f"<b>{t}</b> guided 2027 capex {s.upper()}.",
                  "One weak print is a warning to resize, not the trigger. "
                  "A second flat/down hyperscaler fires the kill criterion.",
                  f"\nTrigger: {trigger}"],
        )
        log.append(f"⚠ capex warning — 1 of 2 ({t} {s})")
    store.set_meta("paper_capex_status", status)
    return log


def _recent_capex_signals(store: Store, window_days: int) -> dict[str, str]:
    """Distinct hyperscalers with a flat/down capex signal inside the window,
    as {ticker: latest_signal}. Reads the capex_signal:<TICKER>:<date> flags."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM app_meta WHERE key LIKE 'capex_signal:%' ORDER BY key ASC"
        ).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        _, ticker, date = r["key"].split(":", 2)
        if date >= cutoff:
            out[ticker] = r["value"]  # ascending key order → latest wins
    return out


def _judge_capex_signal(ticker: str, headlines: list[str]) -> str | None:
    """Ask gpt-4o-mini whether headlines say this hyperscaler guided next-year
    capex FLAT or DOWN. Returns 'down', 'flat', or None (raised/unclear/any API
    failure — the watch must never break tracking)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You track hyperscaler AI capital-expenditure (capex) guidance. "
                    "Given recent headlines for one company, decide the DIRECTION of "
                    "its forward (next-year) capex guidance. Be strict: only a "
                    "headline explicitly about capex/capital spending guidance "
                    "counts — spending on a single project, revenue, or generic "
                    "AI-investment enthusiasm does NOT. Reply with exactly one of: "
                    "'RAISED', 'FLAT', 'DOWN', or 'NONE' (no capex-guidance signal)."
                )},
                {"role": "user", "content": (
                    f"Company: {ticker}\nRecent headlines:\n"
                    + "\n".join(f"- {h}" for h in headlines)
                )},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("DOWN"):
            return "down"
        if answer.startswith("FLAT"):
            return "flat"
        return None
    except Exception:
        return None


def _send_capex_alert(name: str, level: str, body: list[str]) -> None:
    from fundmgr.notify.send import send_telegram
    lines = [f"<b>📋 Paper portfolio — {name}</b>", level]
    lines += [b for b in body if b]
    lines.append("\nVerify against the actual print before acting.")
    send_telegram("\n".join(lines))


# ── Earnings-calendar watch ────────────────────────────────────────────────────

def check_earnings_calendar(slug: str, store: Store | None = None) -> list[str]:
    """Heads-up the day before/of each holding's earnings, and a check-the-print
    reminder the day after — quoting that name's `watch` line and kill criterion.

    Earnings dates come from yfinance (falling back to the seeded next_earnings
    date when it's a clean YYYY-MM-DD). One alert per earnings event. No-op for
    holdings whose earnings date can't be resolved.
    """
    if store is None:
        meta, store = open_portfolio(slug)
        name = meta["name"]
    else:
        name = store.get_meta("paper_name") or slug

    # Cover the plan so the heads-up lands before you buy (held ∪ targets).
    watch_tickers = sorted({p.ticker for p in store.get_positions()} | set(
        json.loads(store.get_meta("paper_target_weights") or "{}")))
    if not watch_tickers:
        return []
    notes = json.loads(store.get_meta("paper_position_notes") or "{}")
    kills = json.loads(store.get_meta("paper_kill_criteria") or "{}")
    today = datetime.now(timezone.utc).date()

    log: list[str] = []
    for ticker in watch_tickers:
        edate = _next_earnings_date(ticker)
        if edate is None:
            seeded = (notes.get(ticker) or {}).get("next_earnings", "")
            edate = _parse_iso_date(seeded)
        if edate is None:
            continue
        delta = (edate - today).days
        estr = edate.strftime("%Y-%m-%d")
        note = notes.get(ticker) or {}
        watch = (note.get("watch") or "").strip()
        kill = (kills.get(ticker) or "").strip()

        from fundmgr.notify.send import send_telegram
        if 0 <= delta <= 1 and not store.get_meta(f"paper_earnhint:{ticker}:{estr}"):
            store.set_meta(f"paper_earnhint:{ticker}:{estr}", today.isoformat())
            when = "today" if delta == 0 else "tomorrow"
            lines = [f"<b>📅 {name} — {ticker} reports {when}</b> ({estr})"]
            if watch:
                lines.append(f"Watch: {watch}")
            if kill:
                lines.append(f"Kill criterion: {kill}")
            send_telegram("\n".join(lines))
            log.append(f"📅 {ticker} earnings heads-up ({estr}, in {delta}d)")
        elif delta == -1 and not store.get_meta(f"paper_earnpost:{ticker}:{estr}"):
            store.set_meta(f"paper_earnpost:{ticker}:{estr}", today.isoformat())
            lines = [f"<b>📊 {name} — {ticker} reported yesterday</b> ({estr})",
                     "Check the print against the thesis:"]
            if watch:
                lines.append(f"Watch: {watch}")
            if kill:
                lines.append(f"Kill criterion: {kill}")
            lines.append("If the kill criterion is met, act per the plan.")
            send_telegram("\n".join(lines))
            log.append(f"📊 {ticker} post-earnings reminder ({estr})")
    return log


def _next_earnings_date(ticker: str):
    """Next (future) earnings date for a ticker via yfinance, or None. Handles
    both the dict-shaped `.calendar` and the older DataFrame."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
    except Exception:
        return None
    dates = []
    try:
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("Earnings Date High") or []
            dates = raw if isinstance(raw, list) else [raw]
        elif cal is not None and hasattr(cal, "loc"):
            val = cal.loc["Earnings Date"]
            dates = list(val) if hasattr(val, "__iter__") else [val]
    except Exception:
        return None
    today = datetime.now(timezone.utc).date()
    parsed = []
    for d in dates:
        try:
            dd = d.date() if hasattr(d, "date") else d
            if dd >= today:
                parsed.append(dd)
        except Exception:
            continue
    return min(parsed) if parsed else None


def _parse_iso_date(text: str):
    """Pull a YYYY-MM-DD out of a fuzzy next_earnings string, else None.
    Deliberately strict — 'late August 2026 (unverified)' yields nothing."""
    import re as _re
    m = _re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%Y-%m-%d").date()
    except ValueError:
        return None


# ── Weight-drift / rebalance watch ─────────────────────────────────────────────

_DRIFT_TRIGGER = 1.5   # alert once a position exceeds 1.5× its target weight
_DRIFT_RESET = 1.4     # …and reset (re-arm) only after it falls back below 1.4×


def check_weight_drift(slug: str, store: Store | None = None) -> list[str]:
    """Alert when any position appreciates past 1.5× its target weight — the
    pre-registered rebalance rule. Transition-based (fires on crossing up, then
    re-arms below 1.4×) so a persistently-large winner isn't a daily nag.

    Uses the latest cached closes (track_portfolio refreshes them first).
    No-op on portfolios without stored target weights.
    """
    if store is None:
        meta, store = open_portfolio(slug)
    targets = json.loads(store.get_meta("paper_target_weights") or "{}")
    if not targets:
        return []
    currency_map = json.loads(store.get_meta("paper_currency_map") or "{}")
    positions = store.get_positions()
    tickers = [p.ticker for p in positions]

    native = {}
    for t in tickers:
        rows = store.get_prices(t)
        if rows:
            native[t] = rows[-1]["close"]
    prices_sek = sek_prices_for(store, tickers, currency_map, native)
    market_value = {p.ticker: p.shares * prices_sek.get(p.ticker, p.avg_cost_sek)
                    for p in positions}
    nav = sum(market_value.values()) + store.get_cash()
    if nav <= 0:
        return []

    log: list[str] = []
    breaches: list[tuple[str, float, float, float]] = []
    for p in positions:
        target = targets.get(p.ticker)
        if not target:
            continue
        weight = market_value[p.ticker] / nav * 100
        ratio = weight / target
        state = store.get_meta(f"paper_drift_state:{p.ticker}") or "under"
        if ratio >= _DRIFT_TRIGGER and state != "over":
            store.set_meta(f"paper_drift_state:{p.ticker}", "over")
            breaches.append((p.ticker, weight, target, ratio))
            log.append(f"⚖ {p.ticker} {weight:.1f}% vs {target:.1f}% target ({ratio:.2f}×)")
        elif ratio < _DRIFT_RESET and state == "over":
            store.set_meta(f"paper_drift_state:{p.ticker}", "under")

    if breaches:
        name = store.get_meta("paper_name") or slug
        from fundmgr.notify.send import send_telegram
        lines = [f"<b>📋 Paper portfolio — {name}</b>",
                 "⚖ Rebalance rule: position(s) past 1.5× target weight"]
        for ticker, weight, target, ratio in breaches:
            lines.append(f"\n<b>{ticker}</b>: {weight:.1f}% (target {target:.1f}%, "
                         f"{ratio:.2f}×) — trim back toward target")
        send_telegram("\n".join(lines))
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
