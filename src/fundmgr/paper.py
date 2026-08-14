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

import hashlib
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


def extract_json_object(text: str) -> str | None:
    """Pull the first complete JSON *object* out of prose or ``` fences.

    `_extract_json_block` looks for an array first, so on an object containing
    one — an LLM verdict with `"unchecked": [...]`, an analysis with
    `"conditions": [...]` — it returns the inner array and the caller sees a
    list where it wanted a dict. This is brace-matched and string-aware, so
    braces inside string values don't throw off the depth count.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
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
    seed_holdings: bool = False,
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

    seed_holdings=True books each holding's own `shares` at its own
    `avg_cost_sek` instead of buying at today's price — the photo-import path,
    where the screenshot already tells us what is held and what it cost. Rows
    carrying kill rules or a horizon (`max_drawdown_pct`, `price_below`,
    `price_above`, `horizon_date`/`horizon_months`) register them as the
    watch plan; see fundmgr.watchplan.

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
                # Seeding + watch-plan extras (photo import / edited review table).
                # Carried through untouched; consumed after the store exists.
                "shares": _opt_float(h.get("shares")),
                "avg_cost_sek": _opt_float(h.get("avg_cost_sek")),
                "max_drawdown_pct": h.get("max_drawdown_pct"),
                "price_below": h.get("price_below"),
                "price_above": h.get("price_above"),
                "horizon_date": h.get("horizon_date"),
                "horizon_months": h.get("horizon_months"),
                "horizon_note": h.get("horizon_note"),
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

    if seed_holdings:
        # Target weight = what you actually hold, at cost, against NAV. Deriving
        # it here rather than letting normalise_weights equal-weight the book is
        # what makes the drift watch meaningful on an imported portfolio — and
        # it sidesteps that function's fraction heuristic, which would misread a
        # mostly-cash book's small weights as 0–1 values.
        for h in holdings:
            shares, cost = h.get("shares"), h.get("avg_cost_sek")
            h["weight_pct"] = (round(shares * cost / capital_sek * 100, 2)
                               if shares and cost and capital_sek else 0.0)
    else:
        holdings = normalise_weights(holdings)
    tickers = [h["ticker"] for h in holdings]

    # Per-ticker trading currency (resolves without a live price via yfinance
    # metadata / suffix map) — needed for later fills and SEK drift math.
    currency_map = {t: detect_currency(t) for t in tickers}

    # Live native-currency prices: used to execute buys and seed the price
    # cache. In plan-only mode a missing price just means no opening fill.
    from fundmgr.data.quotes import live_prices
    native = {t: p for t, p in live_prices(tickers).items() if p}

    if seed_holdings:
        execute_buys = False  # positions come from the screenshot, not today's tape

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

        # Numeric kill lines and time horizons carried on the holding rows.
        from fundmgr import watchplan
        for h in holdings:
            if not any(h.get(k) for k in ("max_drawdown_pct", "price_below", "price_above",
                                          "horizon_date", "horizon_months")):
                continue
            watchplan.set_position_plan(
                store, h["ticker"],
                max_drawdown_pct=h.get("max_drawdown_pct"),
                price_below=h.get("price_below"),
                price_above=h.get("price_above"),
                currency=currency_map.get(h["ticker"]),
                review_date=h.get("horizon_date"),
                horizon_months=h.get("horizon_months"),
                horizon_note=h.get("horizon_note"),
            )

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
        elif seed_holdings:
            # Book what the screenshot says is already held: each holding's own
            # share count at its own SEK cost basis, fee 0 (the fees were paid
            # at the real broker long before this book existed). Same path as a
            # manually recorded fill, so cash and NAV stay consistent.
            for h in holdings:
                t = h["ticker"]
                shares, cost = h.get("shares"), h.get("avg_cost_sek")
                if not shares or not cost:
                    log.append(f"⚠ {t}: no share count / cost basis — not seeded "
                               "(record a fill to open it)")
                    continue
                store.apply_fill(Transaction(
                    ticker=t, side="buy", shares=round(shares, 4),
                    price_sek=round(cost, 4), fee_sek=0.0, source="fill",
                    currency=currency_map.get(t, "SEK"), timestamp=now,
                ))
                thesis, kill = _plan_thesis(h)
                actions.append({
                    "ticker": t, "side": "buy", "target_weight_pct": h["weight_pct"],
                    "confidence": h["confidence"], "thesis": thesis, "kill_criterion": kill,
                    "cluster": h.get("cluster") or "",
                    "sek_estimate": round(shares * cost), "stop_loss_pct": None,
                })
                log.append(f"✓ Seeded {shares:g} × {t} @ {cost:,.2f} SEK cost basis")
            if not actions:
                raise ValueError(
                    "No holdings could be seeded — every row was missing its share "
                    "count or cost basis. Fill those in and try again.")
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
        if execute_buys:
            summary = (f"Paper portfolio '{name}' seeded from "
                       f"{model_label.strip() or 'pasted'} picks at live market prices.")
        elif seed_holdings:
            summary = (f"Live sleeve '{name}' — {len(actions)} holdings imported from "
                       f"{model_label.strip() or 'a portfolio photo'} at their recorded "
                       f"cost basis.")
        else:
            summary = (f"Live sleeve '{name}' — plan imported from "
                       f"{model_label.strip() or 'pasted'} picks; positions fill as you "
                       f"record trades.")
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
            # Drawdown lines are set above, before any price is cached, so they
            # would all fall back to cost. Now that today's prices exist, anchor
            # the ones still missing one — creating a book *is* the review.
            from fundmgr import watchplan
            for tkr, rule in watchplan.get_kill_rules(store).items():
                if rule.get("max_drawdown_pct") and not rule.get("anchor_price_sek"):
                    watchplan.set_position_plan(store, tkr, re_anchor=True)

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
    elif seed_holdings:
        log.append(f"✓ Sleeve '{name}' created — {len(actions)} holdings seeded at cost, "
                   f"{store.get_cash():,.0f} SEK cash. Watched daily by 'fund paper-track'.")
    else:
        log.append(f"✓ Plan '{name}' imported — {len(actions)} intended positions, "
                   f"no fills yet ({store.get_cash():,.0f} SEK cash). "
                   f"Record trades to build the book.")
    return slug, log


def _cost_nav(store: Store) -> float:
    return sum(p.shares * p.avg_cost_sek for p in store.get_positions()) + store.get_cash()


def _opt_float(value) -> float | None:
    """Positive float or None — for optional numeric fields off a web form."""
    if value in (None, ""):
        return None
    try:
        out = float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


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


def intended_plan_symbol(store: Store, ticker: str) -> str | None:
    """The plan's target symbol a stray position should carry — a unique
    target-weight ticker sharing the base, other than the ticker itself
    (e.g. a mis-recorded 'ENR' → 'ENR.DE')."""
    t = (ticker or "").strip().upper()
    targets = set(json.loads(store.get_meta("paper_target_weights") or "{}"))
    base = t.split(".")[0]
    cands = sorted(s for s in targets if s != t and s.split(".")[0] == base)
    return cands[0] if len(cands) == 1 else None


def retag_position(store: Store, old_ticker: str, new_ticker: str) -> dict:
    """Rename a position (and its transactions) from a mis-tagged symbol to the
    correct Yahoo symbol, merging into any existing position at the new symbol.
    Cash is untouched — this only fixes the instrument, not the money. The new
    symbol's plan currency is preserved (or detected if unknown)."""
    old = (old_ticker or "").strip().upper()
    new = (new_ticker or "").strip().upper()
    if not old or not new or old == new:
        raise ValueError("Give distinct old and new tickers.")
    with store._conn() as conn:
        pos = conn.execute(
            "SELECT shares, avg_cost_sek FROM positions WHERE ticker=?", (old,)).fetchone()
        if not pos or float(pos["shares"]) <= 0:
            raise ValueError(f"No position '{old}' to retag.")
        old_shares, old_avg = float(pos["shares"]), float(pos["avg_cost_sek"])
        newpos = conn.execute(
            "SELECT shares, avg_cost_sek FROM positions WHERE ticker=?", (new,)).fetchone()
        if newpos and float(newpos["shares"]) > 0:
            tot = old_shares + float(newpos["shares"])
            avg = (old_shares * old_avg + float(newpos["shares"]) * float(newpos["avg_cost_sek"])) / tot
            conn.execute("UPDATE positions SET shares=?, avg_cost_sek=?, updated_at=? WHERE ticker=?",
                         (tot, avg, datetime.utcnow().isoformat(), new))
            conn.execute("DELETE FROM positions WHERE ticker=?", (old,))
        else:
            conn.execute("UPDATE positions SET ticker=?, updated_at=? WHERE ticker=?",
                         (new, datetime.utcnow().isoformat(), old))
        conn.execute("UPDATE transactions SET ticker=? WHERE ticker=?", (new, old))

    cmap = json.loads(store.get_meta("paper_currency_map") or "{}")
    if new not in cmap:
        cmap[new] = detect_currency(new)
        store.set_meta("paper_currency_map", json.dumps(cmap))
    return {"old": old, "new": new, "shares": old_shares}


def set_cost_basis(store: Store, ticker: str, new_avg_sek: float) -> dict:
    """Set a position's SEK average cost to the broker's real figure (Montrose
    Inköpsvärde ÷ shares), refunding or charging the cash difference so NAV
    stays consistent.

    Fixes the case where a USD/EUR holding was seeded with the purchase price
    converted at *today's* FX instead of the SEK you actually paid, which made
    the dashboard P&L diverge from the broker. Returns a summary dict.
    """
    t, _note = snap_ticker_to_plan(store, ticker)
    try:
        new_avg = float(new_avg_sek)
    except (TypeError, ValueError):
        raise ValueError("New average cost must be a number (SEK per share).")
    if new_avg <= 0:
        raise ValueError("New average cost must be a positive SEK amount.")
    pos = next((p for p in store.get_positions() if p.ticker == t), None)
    if pos is None:
        raise ValueError(f"No position '{t}' in this book.")
    old_avg = pos.avg_cost_sek
    cash_delta = pos.shares * (old_avg - new_avg)   # refund overpayment (or charge)
    store.upsert_position(t, pos.shares, round(new_avg, 4))
    store.adjust_cash(round(cash_delta, 2))
    return {"ticker": t, "shares": pos.shares, "old_avg": old_avg,
            "new_avg": new_avg, "cash_delta": cash_delta}


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

    # Fundamentals for the kill judge: refresh the cache and snapshot it, so a
    # criterion phrased around growth or margins has both a current number and a
    # direction of travel to be judged against. Optional dependency — a book on
    # a machine without financedata just keeps whatever is already cached.
    watched = sorted(set(tickers) | set(json.loads(
        store.get_meta("paper_target_weights") or "{}")))
    if watched:
        try:
            from fundmgr.data.fundamentals import fetch_and_cache_fundamentals
            stale = store.get_stale_fundamentals_tickers(watched, ttl_days=7)
            if stale:
                fetch_and_cache_fundamentals(stale, store, ttl_days=7)
        except Exception as e:
            log.append(f"⚠ fundamentals refresh skipped: {e}")
        # Capital structure comes off the statements, which `.info` never
        # carries. Merged after the financedata refresh — that call rewrites the
        # whole blob, so these have to be folded in afterwards, and before the
        # snapshot so trends include them.
        try:
            from fundmgr.data.statements import refresh as refresh_statements
            merged, seeded = refresh_statements(store, watched)
            if merged:
                log.append(f"Statement metrics merged for {merged} ticker(s)")
            if seeded:
                log.append(f"Back-filled {seeded} reported period(s) from the statements")
        except Exception as e:
            log.append(f"⚠ statement metrics skipped: {e}")
        try:
            from fundmgr.evidence import snapshot_fundamentals
            snapped = snapshot_fundamentals(store, watched)
            if snapped:
                log.append(f"Fundamentals snapshot updated for {snapped} ticker(s)")
        except Exception as e:
            log.append(f"⚠ fundamentals snapshot failed: {e}")

    # Kill-criterion watch: check fresh evidence against each position's
    # pre-registered falsification condition
    try:
        log += check_kill_criteria(slug, store=store)
    except Exception as e:
        log.append(f"⚠ kill-criterion watch failed: {e}")

    # Numeric kill lines (max drawdown / price floor / price target) — the half
    # of the kill criteria that needs no interpretation, so it runs with no API
    # key and fires the moment a line is crossed.
    from fundmgr import watchplan
    try:
        log += watchplan.check_kill_rules(slug, store=store)
    except Exception as e:
        log.append(f"⚠ kill-rule watch failed: {e}")

    # Time horizons: nudge as each position's review date approaches so a fresh
    # analysis happens before the thesis quietly expires.
    try:
        log += watchplan.check_horizons(slug, store=store)
    except Exception as e:
        log.append(f"⚠ horizon watch failed: {e}")

    # Add signals — the other half of monitoring. Runs after the kill watches
    # so a position that just tripped a kill can never also be suggested.
    try:
        from fundmgr import addsignal
        log += addsignal.check_add_signals(slug, store=store)
    except Exception as e:
        log.append(f"⚠ add-signal watch failed: {e}")

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

    from fundmgr.evidence import build_pack

    log: list[str] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hits: list[tuple[str, str]] = []
    gaps: list[tuple[str, dict]] = []
    for ticker, criterion in kills.items():
        if store.get_meta(f"paper_killwatch:{ticker}") == today:
            continue  # already checked today
        pack = build_pack(store, ticker)
        store.set_meta(f"paper_killwatch:{ticker}", today)
        if not (pack["news"] or pack["fundamentals"]):
            continue  # nothing to judge against at all

        result = _judge_kill_hit(ticker, criterion, pack)
        if result is None:
            continue
        store.set_meta(f"paper_killverdict:{ticker}", json.dumps({
            "date": today, **result, "coverage": pack["coverage"],
        }))

        if result["verdict"] == "yes":
            hits.append((ticker, result["reason"]))
            store.set_meta(f"paper_killhit:{ticker}:{today}", result["reason"])
            log.append(f"🚨 {ticker}: kill criterion may be triggering — {result['reason']}")
        elif result["verdict"] == "insufficient":
            # The criterion couldn't actually be checked. Say so once per wording
            # (rewording re-notifies) rather than daily — the point is to tell
            # you the line isn't being watched, not to nag about it.
            fingerprint = hashlib.sha1(criterion.encode()).hexdigest()[:12]
            if store.get_meta(f"paper_killgap:{ticker}") != fingerprint:
                store.set_meta(f"paper_killgap:{ticker}", fingerprint)
                gaps.append((ticker, result))
            log.append(f"❓ {ticker}: kill criterion not verifiable from available "
                       f"evidence — {result['reason']}")
        else:
            store.set_meta(f"paper_killgap:{ticker}", "")  # checkable after all

    from fundmgr.notify.send import send_telegram
    if hits:
        lines = [f"<b>📋 {name}</b>",
                 "🚨 Kill-criterion watch: the evidence may falsify a thesis"]
        for ticker, reason in hits:
            lines.append(f"\n<b>{ticker}</b>: {reason}")
            lines.append(f"  Pre-registered kill: {kills[ticker]}")
        lines.append("\nVerify against the source before acting.")
        send_telegram("\n".join(lines))

    if gaps:
        lines = [f"<b>📋 {name}</b>",
                 "❓ Kill criteria that cannot be checked automatically"]
        for ticker, result in gaps:
            lines.append(f"\n<b>{ticker}</b>: {kills[ticker]}")
            if result["unchecked"]:
                for item in result["unchecked"][:4]:
                    lines.append(f"  • can't verify: {item}")
            elif result["reason"]:
                lines.append(f"  {result['reason']}")
        lines.append("\nThese are quoted back to you at each earnings print instead. "
                     "Reword them around an observable figure if you want them watched.")
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


_JUDGE_SYSTEM = """\
You monitor pre-registered kill criteria (thesis-falsification conditions) for \
stock positions. You are given one criterion and an evidence pack: recent news \
(headlines, summaries and, where available, article text), cached fundamentals \
with their direction of travel, and the position itself.

First decompose the criterion. It often contains several conditions:
  - joined by "and" / "while" / "combined with" → ALL must hold for a hit;
  - joined by "or" → ANY one holds for a hit.
Evaluate each condition separately against the evidence, then combine them.

Then return one of three verdicts:
  "YES"          - the evidence positively shows the criterion met. Cite the
                   specific headline, article passage or figure that shows it.
  "NO"           - the evidence does cover what the criterion asks about, and
                   the criterion is not met.
  "INSUFFICIENT" - the criterion depends on facts this evidence cannot settle
                   (a metric that is not in the fundamentals block and is not
                   discussed in any article, a private figure, a judgement that
                   needs the full report). Say plainly which conditions you
                   could not check.

Be strict about YES: general negativity, share-price moves and unrelated bad
news are never a hit. Be equally strict about NO — if you could not actually
check a condition, the answer is INSUFFICIENT, not NO. A criterion that quietly
returns NO because the data was missing is the failure this exists to prevent.

Reply with a JSON object only:
{"verdict": "YES"|"NO"|"INSUFFICIENT",
 "reason": "<one or two sentences, citing the evidence>",
 "unchecked": ["<condition you could not verify>", ...]}\
"""


def _judge_kill_hit(ticker: str, criterion: str, pack: dict) -> dict | None:
    """Judge one kill criterion against an evidence pack.

    Returns {"verdict": "yes"|"no"|"insufficient", "reason": str,
    "unchecked": [str]}, or None if the call failed outright (the watch must
    never break tracking).
    """
    from fundmgr.evidence import render_pack

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        user = (
            f"# Position: {ticker}\n"
            f"# Kill criterion\n{criterion}\n\n"
            f"# Evidence\n{render_pack(pack)}"
        )
        kwargs = {
            "model": os.getenv("FUND_KILL_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": 400,
            "temperature": 0.0,
        }
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        return _parse_verdict((resp.choices[0].message.content or "").strip())
    except Exception:
        return None


def _parse_verdict(answer: str) -> dict | None:
    """Parse the judge's reply, tolerating a plain-text fallback."""
    if not answer:
        return None

    data = None
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        block = extract_json_object(answer)
        if block:
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict) and data.get("verdict"):
        verdict = str(data["verdict"]).strip().lower()
        unchecked = data.get("unchecked")
        if not isinstance(unchecked, list):
            unchecked = []
        if verdict.startswith("yes"):
            verdict = "yes"
        elif verdict.startswith("insuff"):
            verdict = "insufficient"
        else:
            verdict = "no"
        return {
            "verdict": verdict,
            "reason": str(data.get("reason") or "").strip(),
            "unchecked": [str(u).strip() for u in unchecked if str(u).strip()],
        }

    # Plain-text fallback: the old 'YES: reason' / 'NO' shape.
    upper = answer.upper()
    if upper.startswith("YES"):
        return {"verdict": "yes",
                "reason": answer.split(":", 1)[1].strip() if ":" in answer else answer,
                "unchecked": []}
    if upper.startswith("INSUFFICIENT"):
        return {"verdict": "insufficient",
                "reason": answer.split(":", 1)[1].strip() if ":" in answer else answer,
                "unchecked": []}
    if upper.startswith("NO"):
        return {"verdict": "no", "reason": "", "unchecked": []}
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
            lines += _proof_prompt(store, slug, ticker)
            send_telegram("\n".join(lines))
            log.append(f"📊 {ticker} post-earnings reminder ({estr})")
    return log


def _proof_prompt(store: Store, slug: str, ticker: str) -> list[str]:
    """The proof question for a position that has an add plan, or [].

    The reminder used to say "check the print against the thesis" and stop
    there. The one thing it never did was *ask* — so `proof_confirmed` stayed
    whatever it was, and the report that just invalidated it is the same report
    that should have prompted a new answer. This closes that loop: the question
    arrives where the alert does, with the figures that moved, and an answer
    that is one reply long.
    """
    from fundmgr import addsignal

    plan = addsignal.get_plan(store, ticker)
    if not plan:
        return []

    lines = ["", "<b>Proof check</b> — did this print confirm the thesis?"]
    state = addsignal.proof_status(store, ticker, plan)
    if state["status"] == "fresh":
        lines.append(f"Currently confirmed ({plan.get('proof_confirmed_at')}); "
                     f"this report supersedes it.")
    elif state["status"] == "stale":
        lines.append(f"⚠ {state['reason']}")
    else:
        lines.append("Not yet confirmed for any report.")

    # The numbers that moved, so the question can be answered from the message
    # rather than from memory.
    try:
        from fundmgr.evidence import fundamentals_trend
        moved = [r for r in fundamentals_trend(store, ticker)
                 if r.get("direction") in ("up", "down")][:4]
        for row in moved:
            lines.append(f"• {row['label']}: {row['text']}")
    except Exception:
        pass

    # The company-defined figures the criterion is actually written on, read out
    # of what was published. Offered for confirmation, never recorded here — the
    # alert asks a question, it does not answer it.
    try:
        from fundmgr.data.release import read_report
        found = read_report(store, ticker)
        for f in found.get("figures", [])[:4]:
            lines.append(f"• {f['label']}: <b>{f['value']:,.2f}{f['unit']}</b> "
                         f"— “{f['quote'][:90]}”")
        if found.get("rejected"):
            lines.append(f"⚠ {len(found['rejected'])} figure(s) discarded — the "
                         f"number was not in the source text.")
        if found.get("figures"):
            lines.append(f"Record them with <code>fund paper-read {slug} {ticker} "
                         f"--period &lt;quarter-end&gt; --apply</code> once checked.")
    except Exception:
        pass

    lines.append(f"Reply <code>/proof {slug} {ticker} yes</code> "
                 f"— or <code>no</code> to clear it.")
    return lines


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
