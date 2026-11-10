"""
Ticker and asset universe configuration.
Configurable dictionary supporting HKEX and KRX stock exchanges.
Includes multi-lingual search terms, aliases, cashtags, and timezone details.
"""

from typing import Dict, Any, List

SUPPORTED_EXCHANGES = ["HKEX", "KRX"]

ASSET_UNIVERSE: Dict[str, Dict[str, Any]] = {
    # ---------------------------------------------------------
    # Hong Kong Stock Exchange (HKEX) - 5 Target Assets
    # ---------------------------------------------------------
    "0700.HK": {
        "ticker": "0700.HK",
        "symbol": "0700",
        "name": "Tencent Holdings",
        "name_native": "騰訊控股",
        "name_short": "Tencent",
        "exchange": "HKEX",
        "country": "Hong Kong",
        "currency": "HKD",
        "timezone": "Asia/Hong_Kong",
        "sector": "Communication Services",
        "cashtags": ["$0700", "$TCEHY", "$0700.HK", "#騰訊", "#腾讯"],
        "keywords_en": ["Tencent", "Tencent Holdings", "WeChat", "Riot Games", "Pony Ma"],
        "keywords_native": ["騰訊", "腾讯", "企鵝", "微信", "馬化騰", "700"],
        "social_search_queries": [
            "0700", "騰訊", "腾讯", "Tencent", "700.HK"
        ],
        "news_search_queries": [
            "Tencent Holdings 0700.HK earnings financial results gaming regulatory WeChat",
            "騰訊 業績 財報 遊戲 監管"
        ]
    },
    "9988.HK": {
        "ticker": "9988.HK",
        "symbol": "9988",
        "name": "Alibaba Group",
        "name_native": "阿里巴巴集團",
        "name_short": "Alibaba",
        "exchange": "HKEX",
        "country": "Hong Kong",
        "currency": "HKD",
        "timezone": "Asia/Hong_Kong",
        "sector": "Consumer Cyclical",
        "cashtags": ["$9988", "$BABA", "$9988.HK", "#阿里巴巴", "#阿里"],
        "keywords_en": ["Alibaba", "Alibaba Group", "Taobao", "AliCloud", "Jack Ma"],
        "keywords_native": ["阿里巴巴", "阿里", "淘寶", "阿里云", "菜鳥", "9988"],
        "social_search_queries": [
            "9988", "阿里巴巴", "阿里", "Alibaba", "9988.HK"
        ],
        "news_search_queries": [
            "Alibaba Group 9988.HK cloud ecommerce earnings revenue cloud computing",
            "阿里巴巴 9988 雲計算 業績 財報"
        ]
    },
    "1810.HK": {
        "ticker": "1810.HK",
        "symbol": "1810",
        "name": "Xiaomi Corporation",
        "name_native": "小米集團",
        "name_short": "Xiaomi",
        "exchange": "HKEX",
        "country": "Hong Kong",
        "currency": "HKD",
        "timezone": "Asia/Hong_Kong",
        "sector": "Technology",
        "cashtags": ["$1810", "$XIACY", "$1810.HK", "#小米", "#雷軍"],
        "keywords_en": ["Xiaomi", "Xiaomi Corp", "Lei Jun", "Xiaomi EV", "SU7"],
        "keywords_native": ["小米", "雷軍", "雷总", "小米汽車", "SU7", "1810"],
        "social_search_queries": [
            "1810", "小米", "雷軍", "Xiaomi", "SU7"
        ],
        "news_search_queries": [
            "Xiaomi Corporation 1810.HK smartphone EV SU7 electric vehicle earnings",
            "小米集團 1810 小米汽車 財報 業績"
        ]
    },
    "3690.HK": {
        "ticker": "3690.HK",
        "symbol": "3690",
        "name": "Meituan",
        "name_native": "美團",
        "name_short": "Meituan",
        "exchange": "HKEX",
        "country": "Hong Kong",
        "currency": "HKD",
        "timezone": "Asia/Hong_Kong",
        "sector": "Consumer Cyclical",
        "cashtags": ["$3690", "$MPNGF", "$3690.HK", "#美團", "#美团"],
        "keywords_en": ["Meituan", "Meituan Dianping", "KeeTa", "Wang Xing", "food delivery"],
        "keywords_native": ["美團", "美团", "外賣", "王興", "KeeTa", "3690"],
        "social_search_queries": [
            "3690", "美團", "美团", "Meituan", "外賣"
        ],
        "news_search_queries": [
            "Meituan 3690.HK food delivery KeeTa earnings expansion quarterly report",
            "美團 3690 外賣 季度業績 財報 營收"
        ]
    },
    "0005.HK": {
        "ticker": "0005.HK",
        "symbol": "0005",
        "name": "HSBC Holdings",
        "name_native": "滙豐控股",
        "name_short": "HSBC",
        "exchange": "HKEX",
        "country": "Hong Kong",
        "currency": "HKD",
        "timezone": "Asia/Hong_Kong",
        "sector": "Financial Services",
        "cashtags": ["$0005", "$HSBC", "$0005.HK", "#滙豐", "#匯豐", "#大笨象"],
        "keywords_en": ["HSBC", "HSBC Holdings", "Hongkong and Shanghai Banking Corporation", "dividend"],
        "keywords_native": ["滙豐", "匯豐", "大笨象", "收息", "5號仔", "0005"],
        "social_search_queries": [
            "0005", "滙豐", "匯豐", "HSBC", "5號仔"
        ],
        "news_search_queries": [
            "HSBC Holdings 0005.HK banking interest rates earnings dividend restructuring",
            "滙豐控股 0005 業績 派息 淨利息收入"
        ]
    },

    # ---------------------------------------------------------
    # Korea Exchange (KRX) - 5 Target Assets
    # ---------------------------------------------------------
    "005930.KS": {
        "ticker": "005930.KS",
        "symbol": "005930",
        "name": "Samsung Electronics",
        "name_native": "삼성전자",
        "name_short": "Samsung",
        "exchange": "KRX",
        "country": "South Korea",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "sector": "Technology",
        "cashtags": ["$005930", "$SSNLF", "$005930.KS", "#삼성전자", "#삼전"],
        "keywords_en": ["Samsung Electronics", "Samsung", "HBM", "semiconductor", "Galaxy"],
        "keywords_native": ["삼성전자", "삼전", "반도체", "이재용", "HBM3E", "005930"],
        "social_search_queries": [
            "삼성전자", "삼전", "005930", "Samsung Electronics"
        ],
        "news_search_queries": [
            "Samsung Electronics 005930 semiconductor HBM memory chips earnings guidance",
            "삼성전자 005930 반도체 실적 HBM 수주"
        ]
    },
    "000660.KS": {
        "ticker": "000660.KS",
        "symbol": "000660",
        "name": "SK Hynix",
        "name_native": "SK하이닉스",
        "name_short": "SK Hynix",
        "exchange": "KRX",
        "country": "South Korea",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "sector": "Technology",
        "cashtags": ["$000660", "$HXSCF", "$000660.KS", "#SK하이닉스", "#하이닉스"],
        "keywords_en": ["SK Hynix", "Hynix", "HBM3E", "Nvidia supplier", "DRAM"],
        "keywords_native": ["SK하이닉스", "하이닉스", "HBM", "엔비디아", "하닉", "000660"],
        "social_search_queries": [
            "SK하이닉스", "하이닉스", "000660", "하닉"
        ],
        "news_search_queries": [
            "SK Hynix 000660.KS HBM3E Nvidia supply semiconductor memory earnings",
            "SK하이닉스 000660 HBM 엔비디아 공급 영업이익"
        ]
    },
    "005380.KS": {
        "ticker": "005380.KS",
        "symbol": "005380",
        "name": "Hyundai Motor",
        "name_native": "현대자동차",
        "name_short": "Hyundai Motor",
        "exchange": "KRX",
        "country": "South Korea",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "sector": "Consumer Cyclical",
        "cashtags": ["$005380", "$HYMTF", "$005380.KS", "#현대차", "#현대자동차"],
        "keywords_en": ["Hyundai Motor", "Hyundai", "Genesis", "Ioniq", "Euisun Chung"],
        "keywords_native": ["현대차", "현대자동차", "제네시스", "아이오닉", "정의선", "005380"],
        "social_search_queries": [
            "현대차", "현대자동차", "005380", "Hyundai Motor"
        ],
        "news_search_queries": [
            "Hyundai Motor 005380.KS electric vehicles EV sales US market dividend",
            "현대차 005380 북미 수출 전기차 영업이익 배당"
        ]
    },
    "373220.KS": {
        "ticker": "373220.KS",
        "symbol": "373220",
        "name": "LG Energy Solution",
        "name_native": "LG에너지솔루션",
        "name_short": "LG EnSol",
        "exchange": "KRX",
        "country": "South Korea",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "sector": "Technology",
        "cashtags": ["$373220", "$373220.KS", "#LG에너지솔루션", "#엔솔", "#LG엔솔"],
        "keywords_en": ["LG Energy Solution", "LG Ensol", "battery", "EV battery", "IRA"],
        "keywords_native": ["LG에너지솔루션", "LG엔솔", "엔솔", "2차전지", "배터리", "373220"],
        "social_search_queries": [
            "LG에너지솔루션", "LG엔솔", "373220", "2차전지"
        ],
        "news_search_queries": [
            "LG Energy Solution 373220.KS EV battery IRA tax credit Tesla GM earnings",
            "LG에너지솔루션 373220 배터리 수주 보조금 실적"
        ]
    },
    "000270.KS": {
        "ticker": "000270.KS",
        "symbol": "000270",
        "name": "Kia Corporation",
        "name_native": "기아",
        "name_short": "Kia",
        "exchange": "KRX",
        "country": "South Korea",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "sector": "Consumer Cyclical",
        "cashtags": ["$000270", "$KIMTF", "$000270.KS", "#기아", "#기아차"],
        "keywords_en": ["Kia", "Kia Corporation", "EV6", "EV9", "operating margin"],
        "keywords_native": ["기아", "기아차", "EV6", "EV9", "영업이익률", "000270"],
        "social_search_queries": [
            "기아", "기아차", "000270", "Kia"
        ],
        "news_search_queries": [
            "Kia Corporation 000270.KS vehicle sales EV9 operating profit margin share buyback",
            "기아 000270 분기 실적 영업이익률 주주환원"
        ]
    }
}

def get_tickers_by_exchange(exchange: str) -> List[str]:
    """Return list of tickers for a given exchange."""
    exchange_upper = exchange.upper()
    return [
        ticker for ticker, meta in ASSET_UNIVERSE.items()
        if meta["exchange"].upper() == exchange_upper
    ]

def get_all_tickers() -> List[str]:
    """Return all configured tickers."""
    return list(ASSET_UNIVERSE.keys())

def get_asset_metadata(ticker: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific ticker."""
    if ticker not in ASSET_UNIVERSE:
        raise KeyError(f"Ticker '{ticker}' is not found in ASSET_UNIVERSE.")
    return ASSET_UNIVERSE[ticker]

def add_custom_ticker(ticker: str, metadata: Dict[str, Any]) -> None:
    """Dynamically register or update an asset in the universe."""
    ASSET_UNIVERSE[ticker] = metadata
