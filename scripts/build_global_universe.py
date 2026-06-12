"""
Generate config/universe_global.csv by combining the existing Nordic universe
with a curated list of global large/mid-cap names across US, UK, Europe,
Japan, Canada, and Australia.

Run: uv run python scripts/build_global_universe.py
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORDIC_CSV = ROOT / "config" / "universe.csv"
OUT_CSV    = ROOT / "config" / "universe_global.csv"

# fmt: off
GLOBAL_TICKERS: list[tuple[str, str, str, str, str, str]] = [
    # (name, yahoo_ticker, isin, country, exchange, sector)

    # ── United States ─────────────────────────────────────────────────────────
    # Technology
    ("Apple",                   "AAPL",   "US0378331005", "US", "NASDAQ", "Technology"),
    ("Microsoft",               "MSFT",   "US5949181045", "US", "NASDAQ", "Technology"),
    ("NVIDIA",                  "NVDA",   "US67066G1040", "US", "NASDAQ", "Technology"),
    ("Alphabet A",              "GOOGL",  "US02079K3059", "US", "NASDAQ", "Technology"),
    ("Meta Platforms",          "META",   "US30303M1027", "US", "NASDAQ", "Technology"),
    ("Broadcom",                "AVGO",   "US11135F1012", "US", "NASDAQ", "Technology"),
    ("AMD",                     "AMD",    "US0079031078", "US", "NASDAQ", "Technology"),
    ("Qualcomm",                "QCOM",   "US7475251036", "US", "NASDAQ", "Technology"),
    ("Texas Instruments",       "TXN",    "US8825081040", "US", "NASDAQ", "Technology"),
    ("Applied Materials",       "AMAT",   "US0382221051", "US", "NASDAQ", "Technology"),
    ("Lam Research",            "LRCX",   "US5128071082", "US", "NASDAQ", "Technology"),
    ("KLA Corporation",         "KLAC",   "US4824801009", "US", "NASDAQ", "Technology"),
    ("Micron Technology",       "MU",     "US5951121038", "US", "NASDAQ", "Technology"),
    ("Intel",                   "INTC",   "US4581401001", "US", "NASDAQ", "Technology"),
    ("ServiceNow",              "NOW",    "US81762P1021", "US", "NYSE",   "Technology"),
    ("Salesforce",              "CRM",    "US79466L3024", "US", "NYSE",   "Technology"),
    ("Oracle",                  "ORCL",   "US68389X1054", "US", "NYSE",   "Technology"),
    ("Adobe",                   "ADBE",   "US00724F1012", "US", "NASDAQ", "Technology"),
    ("Intuit",                  "INTU",   "US4612021034", "US", "NASDAQ", "Technology"),
    ("Palo Alto Networks",      "PANW",   "US6974001089", "US", "NASDAQ", "Technology"),
    ("CrowdStrike",             "CRWD",   "US22788C1053", "US", "NASDAQ", "Technology"),
    ("Snowflake",               "SNOW",   "US8334451098", "US", "NYSE",   "Technology"),
    ("Workday",                 "WDAY",   "US98138H1014", "US", "NASDAQ", "Technology"),
    ("Datadog",                 "DDOG",   "US23804L1035", "US", "NASDAQ", "Technology"),
    ("Cloudflare",              "NET",    "US18915M1071", "US", "NYSE",   "Technology"),
    ("Arista Networks",         "ANET",   "US0404131064", "US", "NYSE",   "Technology"),
    ("Fortinet",                "FTNT",   "US34959E1091", "US", "NASDAQ", "Technology"),
    ("Palantir",                "PLTR",   "US69608A1088", "US", "NYSE",   "Technology"),
    ("MongoDB",                 "MDB",    "US60937P1066", "US", "NASDAQ", "Technology"),
    ("Zscaler",                 "ZS",     "US98980G1022", "US", "NASDAQ", "Technology"),
    ("Okta",                    "OKTA",   "US6792951054", "US", "NASDAQ", "Technology"),
    ("HubSpot",                 "HUBS",   "US4435731009", "US", "NYSE",   "Technology"),
    ("Taiwan Semiconductor ADR","TSM",    "US8740391003", "US", "NYSE",   "Technology"),
    ("ASML ADR",                "ASML",   "US0485482059", "US", "NASDAQ", "Technology"),
    # Consumer Discretionary
    ("Amazon",                  "AMZN",   "US0231351067", "US", "NASDAQ", "Consumer Discretionary"),
    ("Tesla",                   "TSLA",   "US88160R1014", "US", "NASDAQ", "Consumer Discretionary"),
    ("Home Depot",              "HD",     "US4370761029", "US", "NYSE",   "Consumer Discretionary"),
    ("McDonald's",              "MCD",    "US5801351017", "US", "NYSE",   "Consumer Discretionary"),
    ("Nike",                    "NKE",    "US6541061031", "US", "NYSE",   "Consumer Discretionary"),
    ("Walt Disney",             "DIS",    "US2546871060", "US", "NYSE",   "Consumer Discretionary"),
    ("Booking Holdings",        "BKNG",   "US09857L1089", "US", "NASDAQ", "Consumer Discretionary"),
    ("Airbnb",                  "ABNB",   "US0090661010", "US", "NASDAQ", "Consumer Discretionary"),
    ("Starbucks",               "SBUX",   "US8552441094", "US", "NASDAQ", "Consumer Discretionary"),
    ("Lowe's",                  "LOW",    "US5486611073", "US", "NYSE",   "Consumer Discretionary"),
    ("Target",                  "TGT",    "US8729211030", "US", "NYSE",   "Consumer Discretionary"),
    ("Uber",                    "UBER",   "US90353T1007", "US", "NYSE",   "Consumer Discretionary"),
    ("Marriott",                "MAR",    "US5719032022", "US", "NASDAQ", "Consumer Discretionary"),
    ("Hilton",                  "HLT",    "US43300A2033", "US", "NYSE",   "Consumer Discretionary"),
    ("General Motors",          "GM",     "US37045V1008", "US", "NYSE",   "Consumer Discretionary"),
    ("Ford",                    "F",      "US3453708600", "US", "NYSE",   "Consumer Discretionary"),
    # Consumer Staples
    ("Walmart",                 "WMT",    "US9311421039", "US", "NYSE",   "Consumer Staples"),
    ("Procter & Gamble",        "PG",     "US7427181091", "US", "NYSE",   "Consumer Staples"),
    ("Coca-Cola",               "KO",     "US1912161007", "US", "NYSE",   "Consumer Staples"),
    ("PepsiCo",                 "PEP",    "US7134481081", "US", "NASDAQ", "Consumer Staples"),
    ("Costco",                  "COST",   "US22160K1051", "US", "NASDAQ", "Consumer Staples"),
    ("Philip Morris",           "PM",     "US7181721090", "US", "NYSE",   "Consumer Staples"),
    ("Altria",                  "MO",     "US02209S1033", "US", "NYSE",   "Consumer Staples"),
    ("Mondelez",                "MDLZ",   "US6092071058", "US", "NASDAQ", "Consumer Staples"),
    ("Colgate-Palmolive",       "CL",     "US1941621039", "US", "NYSE",   "Consumer Staples"),
    # Financials
    ("JPMorgan Chase",          "JPM",    "US46625H1005", "US", "NYSE",   "Financials"),
    ("Visa",                    "V",      "US92826C8394", "US", "NYSE",   "Financials"),
    ("Mastercard",              "MA",     "US57636Q1040", "US", "NYSE",   "Financials"),
    ("Berkshire Hathaway B",    "BRK-B",  "US0846707026", "US", "NYSE",   "Financials"),
    ("Bank of America",         "BAC",    "US0605051046", "US", "NYSE",   "Financials"),
    ("Wells Fargo",             "WFC",    "US9497461015", "US", "NYSE",   "Financials"),
    ("Goldman Sachs",           "GS",     "US38141G1040", "US", "NYSE",   "Financials"),
    ("Morgan Stanley",          "MS",     "US6174464486", "US", "NYSE",   "Financials"),
    ("American Express",        "AXP",    "US0258161092", "US", "NYSE",   "Financials"),
    ("Citigroup",               "C",      "US1729674242", "US", "NYSE",   "Financials"),
    ("BlackRock",               "BLK",    "US09247X1019", "US", "NYSE",   "Financials"),
    ("S&P Global",              "SPGI",   "US78409V1044", "US", "NYSE",   "Financials"),
    ("Moody's",                 "MCO",    "US6153691059", "US", "NYSE",   "Financials"),
    ("CME Group",               "CME",    "US12572Q1058", "US", "NASDAQ", "Financials"),
    ("Charles Schwab",          "SCHW",   "US8085131055", "US", "NYSE",   "Financials"),
    ("Coinbase",                "COIN",   "US19260Q1076", "US", "NASDAQ", "Financials"),
    # Healthcare
    ("UnitedHealth",            "UNH",    "US91324P1021", "US", "NYSE",   "Healthcare"),
    ("Johnson & Johnson",       "JNJ",    "US4781601046", "US", "NYSE",   "Healthcare"),
    ("Eli Lilly",               "LLY",    "US5324571036", "US", "NYSE",   "Healthcare"),
    ("AbbVie",                  "ABBV",   "US00287Y1091", "US", "NYSE",   "Healthcare"),
    ("Merck",                   "MRK",    "US58933Y1055", "US", "NYSE",   "Healthcare"),
    ("Pfizer",                  "PFE",    "US7170811035", "US", "NYSE",   "Healthcare"),
    ("Abbott Labs",             "ABT",    "US0028241000", "US", "NYSE",   "Healthcare"),
    ("Thermo Fisher",           "TMO",    "US8835561023", "US", "NYSE",   "Healthcare"),
    ("Danaher",                 "DHR",    "US2358511028", "US", "NYSE",   "Healthcare"),
    ("Intuitive Surgical",      "ISRG",   "US46120E6023", "US", "NASDAQ", "Healthcare"),
    ("Amgen",                   "AMGN",   "US0311621009", "US", "NASDAQ", "Healthcare"),
    ("Gilead Sciences",         "GILD",   "US3755581036", "US", "NASDAQ", "Healthcare"),
    ("Vertex Pharmaceuticals",  "VRTX",   "US92532F1003", "US", "NASDAQ", "Healthcare"),
    ("Regeneron",               "REGN",   "US75886F1075", "US", "NASDAQ", "Healthcare"),
    ("Moderna",                 "MRNA",   "US60770K1079", "US", "NASDAQ", "Healthcare"),
    ("Boston Scientific",       "BSX",    "US1011371077", "US", "NYSE",   "Healthcare"),
    ("Medtronic",               "MDT",    "IE00BTN1Y115", "US", "NYSE",   "Healthcare"),
    # Energy
    ("Exxon Mobil",             "XOM",    "US30231G1022", "US", "NYSE",   "Energy"),
    ("Chevron",                 "CVX",    "US1667641005", "US", "NYSE",   "Energy"),
    ("ConocoPhillips",          "COP",    "US20825C1045", "US", "NYSE",   "Energy"),
    ("EOG Resources",           "EOG",    "US26875P1012", "US", "NYSE",   "Energy"),
    ("Schlumberger",            "SLB",    "AN8068571086", "US", "NYSE",   "Energy"),
    ("Devon Energy",            "DVN",    "US25179M1036", "US", "NYSE",   "Energy"),
    ("Marathon Petroleum",      "MPC",    "US56585A1025", "US", "NYSE",   "Energy"),
    ("Valero Energy",           "VLO",    "US91913Y1001", "US", "NYSE",   "Energy"),
    # Industrials
    ("Caterpillar",             "CAT",    "US1491231015", "US", "NYSE",   "Industrials"),
    ("Boeing",                  "BA",     "US0970231058", "US", "NYSE",   "Industrials"),
    ("Honeywell",               "HON",    "US4385161066", "US", "NASDAQ", "Industrials"),
    ("Lockheed Martin",         "LMT",    "US5398301094", "US", "NYSE",   "Industrials"),
    ("General Electric",        "GE",     "US3696043013", "US", "NYSE",   "Industrials"),
    ("RTX",                     "RTX",    "US75513E1010", "US", "NYSE",   "Industrials"),
    ("Union Pacific",           "UNP",    "US9078181081", "US", "NYSE",   "Industrials"),
    ("Deere",                   "DE",     "US2441991054", "US", "NYSE",   "Industrials"),
    ("3M",                      "MMM",    "US88579Y1010", "US", "NYSE",   "Industrials"),
    ("Illinois Tool Works",     "ITW",    "US4523081093", "US", "NYSE",   "Industrials"),
    ("Northrop Grumman",        "NOC",    "US6668071029", "US", "NYSE",   "Industrials"),
    ("Emerson Electric",        "EMR",    "US2910111044", "US", "NYSE",   "Industrials"),
    # Materials
    ("Linde",                   "LIN",    "IE00BZ12WP82", "US", "NASDAQ", "Materials"),
    ("Air Products",            "APD",    "US0091581068", "US", "NYSE",   "Materials"),
    ("Newmont",                 "NEM",    "US6516391066", "US", "NYSE",   "Materials"),
    ("Freeport-McMoRan",        "FCX",    "US35671D8570", "US", "NYSE",   "Materials"),
    ("Nucor",                   "NUE",    "US6703461052", "US", "NYSE",   "Materials"),
    # Communication
    ("Netflix",                 "NFLX",   "US64110L1061", "US", "NASDAQ", "Communication Services"),
    ("T-Mobile",                "TMUS",   "US8725901040", "US", "NASDAQ", "Communication Services"),
    ("Verizon",                 "VZ",     "US92343V1044", "US", "NYSE",   "Communication Services"),
    ("AT&T",                    "T",      "US00206R1023", "US", "NYSE",   "Communication Services"),
    ("Comcast",                 "CMCSA",  "US20030N1019", "US", "NASDAQ", "Communication Services"),
    ("Spotify",                 "SPOT",   "LU1778762911", "US", "NYSE",   "Communication Services"),
    # Real Estate
    ("American Tower",          "AMT",    "US03027X1000", "US", "NYSE",   "Real Estate"),
    ("Prologis",                "PLD",    "US74340W1036", "US", "NYSE",   "Real Estate"),
    ("Equinix",                 "EQIX",   "US29444U7000", "US", "NASDAQ", "Real Estate"),
    # Utilities
    ("NextEra Energy",          "NEE",    "US65339F1012", "US", "NYSE",   "Utilities"),
    ("Duke Energy",             "DUK",    "US26441C2044", "US", "NYSE",   "Utilities"),
    ("Southern Company",        "SO",     "US8425871071", "US", "NYSE",   "Utilities"),

    # ── United Kingdom (LSE) ──────────────────────────────────────────────────
    ("AstraZeneca",             "AZN.L",  "GB0009895292", "GB", "LSE", "Healthcare"),
    ("Shell",                   "SHEL.L", "GB00BP6MXD84", "GB", "LSE", "Energy"),
    ("HSBC",                    "HSBA.L", "GB0005405286", "GB", "LSE", "Financials"),
    ("Unilever",                "ULVR.L", "GB00B10RZP78", "GB", "LSE", "Consumer Staples"),
    ("BP",                      "BP.L",   "GB0007980591", "GB", "LSE", "Energy"),
    ("GSK",                     "GSK.L",  "GB0009252882", "GB", "LSE", "Healthcare"),
    ("Rio Tinto",               "RIO.L",  "GB0007188757", "GB", "LSE", "Materials"),
    ("Diageo",                  "DGE.L",  "GB0002374006", "GB", "LSE", "Consumer Staples"),
    ("Rolls-Royce",             "RR.L",   "GB00B63H8491", "GB", "LSE", "Industrials"),
    ("Barclays",                "BARC.L", "GB0031348658", "GB", "LSE", "Financials"),
    ("Lloyds Banking Group",    "LLOY.L", "GB0008706128", "GB", "LSE", "Financials"),
    ("National Grid",           "NG.L",   "GB00BDR05C01", "GB", "LSE", "Utilities"),
    ("BAE Systems",             "BA.L",   "GB0002634946", "GB", "LSE", "Industrials"),
    ("Experian",                "EXPN.L", "IE00B19RTW58", "GB", "LSE", "Industrials"),
    ("Sage Group",              "SGE.L",  "GB00B8C3BL03", "GB", "LSE", "Technology"),
    ("LSEG",                    "LSEG.L", "GB00B0SWJX34", "GB", "LSE", "Financials"),
    ("Glencore",                "GLEN.L", "JE00B4T3BW64", "GB", "LSE", "Materials"),
    ("Anglo American",          "AAL.L",  "GB00B1XZS820", "GB", "LSE", "Materials"),
    ("Reckitt",                 "RKT.L",  "GB00B24CGK77", "GB", "LSE", "Consumer Staples"),
    ("Haleon",                  "HLN.L",  "GB00BMX86B70", "GB", "LSE", "Healthcare"),
    ("Compass Group",           "CPG.L",  "GB00BD6K4575", "GB", "LSE", "Consumer Discretionary"),
    ("InterContinental Hotels", "IHG.L",  "GB00BHJYC057", "GB", "LSE", "Consumer Discretionary"),
    ("Vodafone",                "VOD.L",  "GB00BH4HKS39", "GB", "LSE", "Communication Services"),
    ("Prudential",              "PRU.L",  "GB0007099541", "GB", "LSE", "Financials"),
    ("Standard Chartered",      "STAN.L", "GB0004082847", "GB", "LSE", "Financials"),
    ("Aviva",                   "AV.L",   "GB0002162385", "GB", "LSE", "Financials"),
    ("Legal & General",         "LGEN.L", "GB0005603997", "GB", "LSE", "Financials"),
    ("3i Group",                "III.L",  "GB00B1YW4409", "GB", "LSE", "Financials"),
    ("Melrose Industries",      "MRO.L",  "GB00BJLR0J16", "GB", "LSE", "Industrials"),
    ("Rentokil Initial",        "RTO.L",  "GB00B082RF11", "GB", "LSE", "Industrials"),

    # ── Germany (XETRA) ───────────────────────────────────────────────────────
    ("SAP",                     "SAP.DE",   "DE0007164600", "DE", "XETRA", "Technology"),
    ("Siemens",                 "SIE.DE",   "DE0007236101", "DE", "XETRA", "Industrials"),
    ("Allianz",                 "ALV.DE",   "DE0008404005", "DE", "XETRA", "Financials"),
    ("BASF",                    "BAS.DE",   "DE000BASF111", "DE", "XETRA", "Materials"),
    ("Volkswagen",              "VOW3.DE",  "DE0007664039", "DE", "XETRA", "Consumer Discretionary"),
    ("BMW",                     "BMW.DE",   "DE0005190003", "DE", "XETRA", "Consumer Discretionary"),
    ("Mercedes-Benz",           "MBG.DE",   "DE0007100000", "DE", "XETRA", "Consumer Discretionary"),
    ("Deutsche Telekom",        "DTE.DE",   "DE0005557508", "DE", "XETRA", "Communication Services"),
    ("Munich Re",               "MUV2.DE",  "DE0008430026", "DE", "XETRA", "Financials"),
    ("Bayer",                   "BAYN.DE",  "DE000BAY0017", "DE", "XETRA", "Healthcare"),
    ("Deutsche Bank",           "DBK.DE",   "DE0005140008", "DE", "XETRA", "Financials"),
    ("Airbus",                  "AIR.DE",   "NL0000235190", "DE", "XETRA", "Industrials"),
    ("adidas",                  "ADS.DE",   "DE000A1EWWW0", "DE", "XETRA", "Consumer Discretionary"),
    ("Infineon",                "IFX.DE",   "DE0006231004", "DE", "XETRA", "Technology"),
    ("RWE",                     "RWE.DE",   "DE0007037129", "DE", "XETRA", "Utilities"),
    ("E.ON",                    "EOAN.DE",  "DE000ENAG999", "DE", "XETRA", "Utilities"),
    ("Fresenius",               "FRE.DE",   "DE0005785604", "DE", "XETRA", "Healthcare"),
    ("Henkel",                  "HEN3.DE",  "DE0006048432", "DE", "XETRA", "Consumer Staples"),
    ("Daimler Truck",           "DTG.DE",   "DE000DTR0CK8", "DE", "XETRA", "Industrials"),
    ("Siemens Energy",          "ENR.DE",   "DE000ENER6Y0", "DE", "XETRA", "Energy"),
    ("Porsche AG",              "P911.DE",  "DE000PAG9113", "DE", "XETRA", "Consumer Discretionary"),
    ("Zalando",                 "ZAL.DE",   "DE000ZAL1111", "DE", "XETRA", "Consumer Discretionary"),
    ("Deutsche Boerse",         "DB1.DE",   "DE0005810055", "DE", "XETRA", "Financials"),

    # ── France (EURONEXT) ─────────────────────────────────────────────────────
    ("LVMH",                    "MC.PA",    "FR0000121014", "FR", "EURONEXT", "Consumer Discretionary"),
    ("TotalEnergies",           "TTE.PA",   "FR0014000MR3", "FR", "EURONEXT", "Energy"),
    ("L'Oréal",                 "OR.PA",    "FR0000120321", "FR", "EURONEXT", "Consumer Staples"),
    ("Sanofi",                  "SAN.PA",   "FR0000120578", "FR", "EURONEXT", "Healthcare"),
    ("BNP Paribas",             "BNP.PA",   "FR0000131104", "FR", "EURONEXT", "Financials"),
    ("Air Liquide",             "AI.PA",    "FR0000120073", "FR", "EURONEXT", "Materials"),
    ("Schneider Electric",      "SU.PA",    "FR0000121972", "FR", "EURONEXT", "Industrials"),
    ("Hermès",                  "RMS.PA",   "FR0000052292", "FR", "EURONEXT", "Consumer Discretionary"),
    ("Kering",                  "KER.PA",   "FR0000121485", "FR", "EURONEXT", "Consumer Discretionary"),
    ("Safran",                  "SAF.PA",   "FR0000073272", "FR", "EURONEXT", "Industrials"),
    ("AXA",                     "CS.PA",    "FR0000120628", "FR", "EURONEXT", "Financials"),
    ("Pernod Ricard",           "RI.PA",    "FR0000120693", "FR", "EURONEXT", "Consumer Staples"),
    ("Danone",                  "BN.PA",    "FR0000120644", "FR", "EURONEXT", "Consumer Staples"),
    ("Michelin",                "ML.PA",    "FR0000121261", "FR", "EURONEXT", "Consumer Discretionary"),
    ("Vinci",                   "DG.PA",    "FR0000125486", "FR", "EURONEXT", "Industrials"),
    ("Saint-Gobain",            "SGO.PA",   "FR0000125007", "FR", "EURONEXT", "Materials"),
    ("Engie",                   "ENGI.PA",  "FR0010208488", "FR", "EURONEXT", "Utilities"),
    ("Orange",                  "ORA.PA",   "FR0000133308", "FR", "EURONEXT", "Communication Services"),
    ("Capgemini",               "CAP.PA",   "FR0000125338", "FR", "EURONEXT", "Technology"),
    ("Dassault Systèmes",       "DSY.PA",   "FR0014003TT8", "FR", "EURONEXT", "Technology"),
    ("Publicis",                "PUB.PA",   "FR0000130577", "FR", "EURONEXT", "Communication Services"),
    ("Stellantis",              "STLAP.PA", "NL00150001Q9", "FR", "EURONEXT", "Consumer Discretionary"),
    ("Renault",                 "RNO.PA",   "FR0000131906", "FR", "EURONEXT", "Consumer Discretionary"),
    ("Legrand",                 "LR.PA",    "FR0010307819", "FR", "EURONEXT", "Industrials"),

    # ── Switzerland (SIX) ─────────────────────────────────────────────────────
    ("Nestlé",                  "NESN.SW",  "CH0038863350", "CH", "SIX", "Consumer Staples"),
    ("Novartis",                "NOVN.SW",  "CH0012221716", "CH", "SIX", "Healthcare"),
    ("Roche",                   "ROG.SW",   "CH0012032048", "CH", "SIX", "Healthcare"),
    ("UBS",                     "UBSG.SW",  "CH0244767585", "CH", "SIX", "Financials"),
    ("ABB",                     "ABBN.SW",  "CH0012221499", "CH", "SIX", "Industrials"),
    ("Zurich Insurance",        "ZURN.SW",  "CH0011075394", "CH", "SIX", "Financials"),
    ("Lonza",                   "LONN.SW",  "CH0013841017", "CH", "SIX", "Healthcare"),
    ("Partners Group",          "PGHN.SW",  "CH0024608827", "CH", "SIX", "Financials"),
    ("Richemont",               "CFR.SW",   "CH0210483332", "CH", "SIX", "Consumer Discretionary"),
    ("Holcim",                  "HOLN.SW",  "CH0012214059", "CH", "SIX", "Materials"),
    ("Swiss Re",                "SREN.SW",  "CH0126881561", "CH", "SIX", "Financials"),
    ("Geberit",                 "GEBN.SW",  "CH0030170408", "CH", "SIX", "Industrials"),

    # ── Netherlands (EURONEXT Amsterdam) ──────────────────────────────────────
    ("ASML",                    "ASML.AS",  "NL0010273215", "NL", "EURONEXT", "Technology"),
    ("ING",                     "INGA.AS",  "NL0011821202", "NL", "EURONEXT", "Financials"),
    ("Heineken",                "HEIA.AS",  "NL0000009165", "NL", "EURONEXT", "Consumer Staples"),
    ("Philips",                 "PHIA.AS",  "NL0000009538", "NL", "EURONEXT", "Healthcare"),
    ("Wolters Kluwer",          "WKL.AS",   "NL0000395903", "NL", "EURONEXT", "Industrials"),
    ("Adyen",                   "ADYEN.AS", "NL0012969182", "NL", "EURONEXT", "Technology"),
    ("Universal Music Group",   "UMG.AS",   "NL0015000IY2", "NL", "EURONEXT", "Communication Services"),
    ("NXP Semiconductors",      "NXPI.AS",  "NL0009538784", "NL", "EURONEXT", "Technology"),
    ("NN Group",                "NN.AS",    "NL0010773842", "NL", "EURONEXT", "Financials"),
    ("IMCD",                    "IMCD.AS",  "NL0010801007", "NL", "EURONEXT", "Materials"),

    # ── Japan (TSE) ───────────────────────────────────────────────────────────
    ("Toyota Motor",            "7203.T",   "JP3633400001", "JP", "TSE", "Consumer Discretionary"),
    ("Sony Group",              "6758.T",   "JP3435000009", "JP", "TSE", "Consumer Discretionary"),
    ("SoftBank Group",          "9984.T",   "JP3436100006", "JP", "TSE", "Communication Services"),
    ("Keyence",                 "6861.T",   "JP3236200006", "JP", "TSE", "Technology"),
    ("Tokyo Electron",          "8035.T",   "JP3571400005", "JP", "TSE", "Technology"),
    ("Shin-Etsu Chemical",      "4063.T",   "JP3371200001", "JP", "TSE", "Materials"),
    ("FANUC",                   "6954.T",   "JP3802400006", "JP", "TSE", "Industrials"),
    ("Murata Manufacturing",    "6981.T",   "JP3914400001", "JP", "TSE", "Technology"),
    ("Fast Retailing",          "9983.T",   "JP3802300008", "JP", "TSE", "Consumer Discretionary"),
    ("Mitsubishi UFJ Financial","8306.T",   "JP3902900004", "JP", "TSE", "Financials"),
    ("Nintendo",                "7974.T",   "JP3756600007", "JP", "TSE", "Communication Services"),
    ("Honda Motor",             "7267.T",   "JP3854600008", "JP", "TSE", "Consumer Discretionary"),
    ("Hitachi",                 "6501.T",   "JP3788600009", "JP", "TSE", "Technology"),
    ("Daikin Industries",       "6367.T",   "JP3481800005", "JP", "TSE", "Industrials"),
    ("Recruit Holdings",        "6098.T",   "JP3970300004", "JP", "TSE", "Industrials"),
    ("Hoya",                    "7741.T",   "JP3840400008", "JP", "TSE", "Healthcare"),
    ("Renesas Electronics",     "6723.T",   "JP3163130000", "JP", "TSE", "Technology"),
    ("Sumitomo Mitsui",         "8316.T",   "JP3890350006", "JP", "TSE", "Financials"),
    ("Mitsui",                  "8031.T",   "JP3893600001", "JP", "TSE", "Industrials"),
    ("Disco Corporation",       "6146.T",   "JP3548600000", "JP", "TSE", "Technology"),

    # ── Canada (TSX) ──────────────────────────────────────────────────────────
    ("Royal Bank of Canada",    "RY.TO",    "CA7800871021", "CA", "TSX", "Financials"),
    ("Toronto-Dominion Bank",   "TD.TO",    "CA8911605092", "CA", "TSX", "Financials"),
    ("Shopify",                 "SHOP.TO",  "CA82509L1076", "CA", "TSX", "Technology"),
    ("Canadian Natural Res",    "CNQ.TO",   "CA1363851017", "CA", "TSX", "Energy"),
    ("Suncor Energy",           "SU.TO",    "CA8672241079", "CA", "TSX", "Energy"),
    ("Brookfield",              "BN.TO",    "CA11271J1075", "CA", "TSX", "Financials"),
    ("Barrick Gold",            "ABX.TO",   "CA0679011084", "CA", "TSX", "Materials"),
    ("Agnico Eagle Mines",      "AEM.TO",   "CA0084741085", "CA", "TSX", "Materials"),
    ("Canadian Pacific Kansas", "CP.TO",    "CA13645T1003", "CA", "TSX", "Industrials"),
    ("Manulife Financial",      "MFC.TO",   "CA56501R1064", "CA", "TSX", "Financials"),
    ("Cenovus Energy",          "CVE.TO",   "CA15135U1093", "CA", "TSX", "Energy"),
    ("BCE",                     "BCE.TO",   "CA05534B7604", "CA", "TSX", "Communication Services"),

    # ── Australia (ASX) ───────────────────────────────────────────────────────
    ("BHP Group",               "BHP.AX",   "AU000000BHP4", "AU", "ASX", "Materials"),
    ("Commonwealth Bank",       "CBA.AX",   "AU000000CBA7", "AU", "ASX", "Financials"),
    ("CSL",                     "CSL.AX",   "AU000000CSL8", "AU", "ASX", "Healthcare"),
    ("ANZ Banking Group",       "ANZ.AX",   "AU000000ANZ3", "AU", "ASX", "Financials"),
    ("Westpac Banking",         "WBC.AX",   "AU000000WBC1", "AU", "ASX", "Financials"),
    ("Fortescue",               "FMG.AX",   "AU000000FMG4", "AU", "ASX", "Materials"),
    ("Macquarie Group",         "MQG.AX",   "AU000000MQG1", "AU", "ASX", "Financials"),
    ("Woodside Energy",         "WDS.AX",   "AU0000096929", "AU", "ASX", "Energy"),
    ("Wesfarmers",              "WES.AX",   "AU000000WES1", "AU", "ASX", "Consumer Discretionary"),
    ("National Australia Bank", "NAB.AX",   "AU000000NAB4", "AU", "ASX", "Financials"),

    # ── China ADRs (US-listed) ────────────────────────────────────────────────
    ("Alibaba Group ADR",       "BABA",     "US01609W1027", "CN", "NYSE",   "Consumer Discretionary"),
    ("JD.com ADR",              "JD",       "US47215P1066", "CN", "NASDAQ", "Consumer Discretionary"),
    ("Baidu ADR",               "BIDU",     "US0567521085", "CN", "NASDAQ", "Communication Services"),
    ("PDD Holdings ADR",        "PDD",      "US69767V1098", "CN", "NASDAQ", "Consumer Discretionary"),
    ("NIO ADR",                 "NIO",      "US62914V1061", "CN", "NYSE",   "Consumer Discretionary"),
    ("Li Auto ADR",             "LI",       "US50202M1027", "CN", "NASDAQ", "Consumer Discretionary"),
]
# fmt: on


def main():
    # Load existing Nordic universe
    nordic_rows: list[dict] = []
    with NORDIC_CSV.open() as f:
        nordic_rows = list(csv.DictReader(f))

    nordic_tickers = {r["yahoo_ticker"] for r in nordic_rows}
    print(f"Nordic tickers loaded: {len(nordic_rows)}")

    # Deduplicate global list against Nordic (e.g. ASML is in Nordic as ASML.AS)
    global_rows: list[dict] = []
    seen = set(nordic_tickers)
    for name, ticker, isin, country, exchange, sector in GLOBAL_TICKERS:
        if ticker in seen:
            print(f"  Skipping duplicate: {ticker}")
            continue
        seen.add(ticker)
        global_rows.append({
            "name": name,
            "yahoo_ticker": ticker,
            "isin": isin,
            "country": country,
            "exchange": exchange,
            "sector": sector,
            "enabled": "True",
        })

    all_rows = nordic_rows + global_rows
    print(f"Global tickers added: {len(global_rows)}")
    print(f"Total universe: {len(all_rows)}")

    fieldnames = ["name", "yahoo_ticker", "isin", "country", "exchange", "sector", "enabled"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written: {OUT_CSV}")


if __name__ == "__main__":
    main()
