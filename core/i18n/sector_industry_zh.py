from __future__ import annotations

import pandas as pd


SECTOR_ZH = {
    "Basic Materials": "原材料",
    "Communication Services": "通訊服務",
    "Consumer Cyclical": "非必需消費",
    "Consumer Defensive": "必需消費",
    "Energy": "能源",
    "Financial Services": "金融服務",
    "Healthcare": "醫療健康",
    "Industrials": "工業",
    "Real Estate": "房地產",
    "Technology": "科技",
    "Utilities": "公用事業",
}


INDUSTRY_ZH = {
    "Advertising Agencies": "廣告代理",
    "Aerospace & Defense": "航太國防",
    "Agricultural Inputs": "農業投入",
    "Airlines": "航空公司",
    "Airports & Air Services": "機場服務",
    "Aluminum": "鋁業",
    "Apparel Manufacturing": "服裝製造",
    "Apparel Retail": "服裝零售",
    "Asset Management": "資產管理",
    "Auto & Truck Dealerships": "汽車經銷",
    "Auto Manufacturers": "汽車製造",
    "Auto Parts": "汽車零件",
    "Banks—Diversified": "多元銀行",
    "Banks—Regional": "地區銀行",
    "Beverages—Brewers": "啤酒飲品",
    "Beverages—Non-Alcoholic": "非酒精飲品",
    "Beverages—Wineries & Distilleries": "酒類釀造",
    "Biotechnology": "生物科技",
    "Broadcasting": "廣播媒體",
    "Building Materials": "建築材料",
    "Building Products & Equipment": "建築設備",
    "Business Equipment & Supplies": "商業設備",
    "Capital Markets": "資本市場",
    "Chemicals": "化工",
    "Coking Coal": "焦煤",
    "Communication Equipment": "通訊設備",
    "Computer Hardware": "電腦硬件",
    "Confectioners": "糖果食品",
    "Conglomerates": "綜合企業",
    "Consulting Services": "顧問服務",
    "Consumer Electronics": "消費電子",
    "Copper": "銅業",
    "Credit Services": "信貸服務",
    "Department Stores": "百貨公司",
    "Diagnostics & Research": "診斷研究",
    "Discount Stores": "折扣零售",
    "Drug Manufacturers—General": "大型藥廠",
    "Drug Manufacturers—Specialty & Generic": "專科及仿製藥",
    "Education & Training Services": "教育培訓",
    "Electrical Equipment & Parts": "電氣設備",
    "Electronic Components": "電子零件",
    "Electronic Gaming & Multimedia": "遊戲多媒體",
    "Electronics & Computer Distribution": "電子電腦分銷",
    "Engineering & Construction": "工程建設",
    "Entertainment": "娛樂",
    "Farm & Heavy Construction Machinery": "農業重機",
    "Farm Products": "農產品",
    "Financial Conglomerates": "金融控股",
    "Financial Data & Stock Exchanges": "金融數據交易所",
    "Food Distribution": "食品分銷",
    "Footwear & Accessories": "鞋履配飾",
    "Furnishings, Fixtures & Appliances": "家居家電",
    "Gambling": "博彩",
    "Gold": "黃金",
    "Grocery Stores": "雜貨零售",
    "Health Information Services": "醫療資訊",
    "Healthcare Plans": "醫療保險",
    "Home Improvement Retail": "家居修繕零售",
    "Household & Personal Products": "家居個護",
    "Industrial Distribution": "工業分銷",
    "Information Technology Services": "資訊科技服務",
    "Infrastructure Operations": "基建營運",
    "Insurance Brokers": "保險經紀",
    "Insurance—Diversified": "多元保險",
    "Insurance—Life": "人壽保險",
    "Insurance—Property & Casualty": "財產意外保險",
    "Insurance—Reinsurance": "再保險",
    "Insurance—Specialty": "專業保險",
    "Integrated Freight & Logistics": "綜合物流",
    "Internet Content & Information": "互聯網內容",
    "Internet Retail": "網上零售",
    "Leisure": "休閒",
    "Lodging": "住宿",
    "Lumber & Wood Production": "木材製品",
    "Luxury Goods": "奢侈品",
    "Marine Shipping": "海運",
    "Medical Care Facilities": "醫療設施",
    "Medical Devices": "醫療設備",
    "Medical Distribution": "醫療分銷",
    "Medical Instruments & Supplies": "醫療器材",
    "Metal Fabrication": "金屬加工",
    "Mortgage Finance": "按揭金融",
    "Oil & Gas Drilling": "油氣鑽探",
    "Oil & Gas E&P": "油氣勘探",
    "Oil & Gas Equipment & Services": "油氣設備服務",
    "Oil & Gas Integrated": "綜合油氣",
    "Oil & Gas Midstream": "油氣中游",
    "Oil & Gas Refining & Marketing": "油氣煉銷",
    "Other Industrial Metals & Mining": "其他工業金屬",
    "Other Precious Metals & Mining": "其他貴金屬",
    "Packaged Foods": "包裝食品",
    "Packaging & Containers": "包裝容器",
    "Paper & Paper Products": "紙品",
    "Personal Services": "個人服務",
    "Pharmaceutical Retailers": "藥品零售",
    "Pollution & Treatment Controls": "污染治理",
    "Publishing": "出版",
    "REIT—Diversified": "多元REIT",
    "REIT—Healthcare Facilities": "醫療REIT",
    "REIT—Hotel & Motel": "酒店REIT",
    "REIT—Industrial": "工業REIT",
    "REIT—Mortgage": "按揭REIT",
    "REIT—Office": "辦公室REIT",
    "REIT—Residential": "住宅REIT",
    "REIT—Retail": "零售REIT",
    "REIT—Specialty": "專門REIT",
    "Railroads": "鐵路",
    "Real Estate Services": "房地產服務",
    "Real Estate—Development": "房地產開發",
    "Real Estate—Diversified": "多元房地產",
    "Recreational Vehicles": "休閒車",
    "Rental & Leasing Services": "租賃服務",
    "Residential Construction": "住宅建築",
    "Resorts & Casinos": "度假博彩",
    "Restaurants": "餐飲",
    "Scientific & Technical Instruments": "科研儀器",
    "Security & Protection Services": "保安防護",
    "Semiconductor Equipment & Materials": "半導體設備材料",
    "Semiconductors": "半導體",
    "Shell Companies": "殼公司",
    "Silver": "白銀",
    "Software—Application": "應用軟件",
    "Software—Infrastructure": "基建軟件",
    "Solar": "太陽能",
    "Specialty Business Services": "專業商業服務",
    "Specialty Chemicals": "特種化工",
    "Specialty Industrial Machinery": "專用工業機械",
    "Specialty Retail": "專門零售",
    "Staffing & Employment Services": "人力資源服務",
    "Steel": "鋼鐵",
    "Telecom Services": "電訊服務",
    "Textile Manufacturing": "紡織製造",
    "Thermal Coal": "動力煤",
    "Tobacco": "煙草",
    "Tools & Accessories": "工具配件",
    "Travel Services": "旅遊服務",
    "Trucking": "貨運卡車",
    "Uranium": "鈾礦",
    "Utilities—Diversified": "多元公用",
    "Utilities—Independent Power Producers": "獨立電力",
    "Utilities—Regulated Electric": "受規管電力",
    "Utilities—Regulated Gas": "受規管燃氣",
    "Utilities—Regulated Water": "受規管水務",
    "Utilities—Renewable": "可再生公用",
    "Waste Management": "廢物管理",
}


def _clean_label(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return ""
    return text


def translate_sector(value) -> str:
    text = _clean_label(value)
    return SECTOR_ZH.get(text, text)


def translate_industry(value) -> str:
    text = _clean_label(value)
    if text in INDUSTRY_ZH:
        return INDUSTRY_ZH[text]
    dash_variant = text.replace(" - ", "—")
    return INDUSTRY_ZH.get(dash_variant, text)


def sector_industry_zh_text(row) -> str:
    parts = [
        translate_sector(row.get("sector_name")),
        translate_industry(row.get("industry_name")),
    ]
    return " | ".join(part for part in parts if part)
