"""
analysis.py — Corporate Governance & Financialization Analysis of Danieli & C. S.p.A. (BIT:DAN)
================================================================================================
Loads the five S&P Capital IQ Excel files in `Data/`, converts all monetary series from USD ($000)
to EUR (€ Millions) using exact embedded S&P exchange rates (and the irrevocable 1,936.27 ITL/EUR
conversion rate for pre-1999 Italian Lira periods), computes key financialization and ownership
concentration metrics (1990–2025), exports clean CSV datasets to `output/`, and prints a structured
analytical summary mapped to the three sections of the course report.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "Data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "bs": DATA_DIR / "SPGlobal_DanieliAndC.OfficineMeccanicheS.p.A._BalanceSheet_11-Nov-2026.xlsx",
    "is": DATA_DIR / "SPGlobal_DanieliAndC.OfficineMeccanicheS.p.A._IncomeStatement_11-Nov-2026.xlsx",
    "cf": DATA_DIR / "SPGlobal_DanieliAndC.OfficineMeccanicheS.p.A._CashFlow_11-Nov-2026.xlsx",
    "fh": DATA_DIR / "SPGlobal_DanieliAndC.OfficineMeccanicheS.p.A._FinancialHighlights_11-Nov-2026.xlsx",
    "own": DATA_DIR / "SPGlobal_DanieliAndC.OfficineMeccanicheS.p.A._OwnershipHistory_11-Nov-2026.xlsx",
}

ITL_PER_EUR = 1936.27


def load_sp_financial_sheet(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """
    Parse an S&P Capital IQ financial statement sheet (Balance Sheet, Income Statement,
    Cash Flow, or Financial Highlights) where row 13 (0-indexed row 12) holds fiscal years
    ('1990 FY' .. '2025 FY') and rows 17-19 hold exchange rates and currency codes.
    """
    raw = pd.read_excel(filepath, sheet_name=0, header=None)
    header_row = raw.iloc[12, 1:].astype(str).tolist()
    valid_cols = []
    years = []
    for idx, col_val in enumerate(header_row, start=1):
        if "FY" in col_val:
            yr = int(col_val.replace("FY", "").strip())
            if yr <= 2025:
                valid_cols.append(idx)
                years.append(yr)

    meta = {"years": years}
    for r_idx in range(14, 19):
        label = str(raw.iloc[r_idx, 0]).strip()
        if label and label != "nan":
            meta[label] = raw.iloc[r_idx, valid_cols].tolist()

    data_rows = {}
    for r_idx in range(19, len(raw)):
        label = str(raw.iloc[r_idx, 0]).strip()
        if not label or label == "nan":
            continue
        vals = pd.to_numeric(raw.iloc[r_idx, valid_cols], errors="coerce").values
        # Keep first occurrence or disambiguate duplicate labels
        if label in data_rows:
            label = f"{label} (R{r_idx + 1})"
        data_rows[label] = vals

    df = pd.DataFrame(data_rows, index=years)
    df.index.name = "Year"
    return df, meta


def compute_eur_conversion_factors(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute multiplicative conversion factors from S&P reported USD ($000) to EUR (€ Millions):
      - 1999–2025 (Reported Currency = EUR):
          Stock items (BS) = (USD_k / Spot_USD_per_EUR) / 1000
          Flow items (IS/CF) = (USD_k / Avg_USD_per_EUR) / 1000
      - 1990–1992 & 1997–1998 (Reported Currency = ITL):
          Stock items (BS) = (USD_k / Spot_USD_per_ITL) / 1936.27 / 1000
          Flow items (IS/CF) = (USD_k / Avg_USD_per_ITL) / 1936.27 / 1000
      - 1993–1996 (Reported in USD with rate=1):
          Converted to synthetic EUR/ECU using historical USD/ECU parity implied by
          Danieli's constant nominal share capital (1.2489 for 1993–1994; 1.2697 for 1995–1996).
    """
    years = meta["years"]
    spot_rates = pd.to_numeric(pd.Series(meta["Spot Exchange Rate"]), errors="coerce").values
    avg_rates = pd.to_numeric(pd.Series(meta["Average Exchange Rate"]), errors="coerce").values
    currencies = [str(c).strip() for c in meta.get("Reported Currency Code", ["EUR"] * len(years))]

    bs_factors = np.zeros(len(years), dtype=float)
    is_factors = np.zeros(len(years), dtype=float)

    for i, (yr, curr, spot, avg) in enumerate(zip(years, currencies, spot_rates, avg_rates)):
        if curr == "EUR":
            bs_factors[i] = 1.0 / (spot * 1000.0)
            is_factors[i] = 1.0 / (avg * 1000.0)
        elif curr == "ITL":
            bs_factors[i] = 1.0 / (spot * ITL_PER_EUR * 1000.0)
            is_factors[i] = 1.0 / (avg * ITL_PER_EUR * 1000.0)
        else:
            ecu_rate = 1.2489 if yr <= 1994 else 1.2697
            bs_factors[i] = 1.0 / (ecu_rate * 1000.0)
            is_factors[i] = 1.0 / (ecu_rate * 1000.0)

    return bs_factors, is_factors


def build_financial_dataset() -> pd.DataFrame:
    """Build the unified 1990–2025 EUR Millions financial dataset and financialization ratios."""
    bs_df, bs_meta = load_sp_financial_sheet(FILES["bs"])
    is_df, is_meta = load_sp_financial_sheet(FILES["is"])
    cf_df, _ = load_sp_financial_sheet(FILES["cf"])
    fh_df, _ = load_sp_financial_sheet(FILES["fh"])

    bs_fac, is_fac = compute_eur_conversion_factors(bs_meta)
    years = bs_df.index

    out = pd.DataFrame(index=years)
    out["OrigCurrency"] = bs_meta["Reported Currency Code"]
    out["SpotRateUSD"] = pd.to_numeric(pd.Series(bs_meta["Spot Exchange Rate"], index=years), errors="coerce")
    out["AvgRateUSD"] = pd.to_numeric(pd.Series(is_meta["Average Exchange Rate"], index=years), errors="coerce")

    # Income Statement (EUR Millions)
    out["Revenue_EUR_M"] = (is_df["Total Revenue"] * is_fac).round(3)
    out["GrossProfit_EUR_M"] = (is_df["Gross Profit"] * is_fac).round(3)
    out["EBITDA_EUR_M"] = (is_df["EBITDA"] * is_fac).round(3)
    out["EBIT_EUR_M"] = (is_df["Operating Income"] * is_fac).round(3)
    out["InterestExpense_EUR_M"] = (is_df["Interest Expense"] * is_fac).round(3)
    out["FinancialIncome_EUR_M"] = (is_df["Interest and Invest. Income"] * is_fac).round(3)
    out["NetInterest_EUR_M"] = (is_df["Net Interest Exp."] * is_fac).round(3)
    out["FXGainLoss_EUR_M"] = (is_df["Currency Exchange Gains"] * is_fac).round(3)
    out["EBT_EUR_M"] = (is_df["EBT Incl. Unusual Items"] * is_fac).round(3)
    out["TaxExpense_EUR_M"] = (is_df["Income Tax Expense"] * is_fac).round(3)
    out["EffectiveTaxRate_Pct"] = is_df["Effective Tax Rate (%)"].round(2)
    out["NetIncome_EUR_M"] = (is_df["Net Income"] * is_fac).round(3)

    # Balance Sheet (EUR Millions)
    out["Cash_EUR_M"] = (bs_df["Cash and Equivalents"] * bs_fac).round(3)
    out["ShortTermInv_EUR_M"] = (bs_df["Short-term Investments"] * bs_fac).round(3)
    out["CashAndSTI_EUR_M"] = (bs_df["Cash & Short-term Investments"] * bs_fac).round(3)
    out["LongTermInv_EUR_M"] = (bs_df["Long-term Investments"] * bs_fac).round(3)
    out["TotalFinAssets_EUR_M"] = (out["CashAndSTI_EUR_M"] + out["LongTermInv_EUR_M"].fillna(0)).round(3)
    out["Receivables_EUR_M"] = (bs_df["Total Receivables"] * bs_fac).round(3)
    out["Inventory_EUR_M"] = (bs_df["Inventory"] * bs_fac).round(3)
    out["NetPPE_EUR_M"] = (bs_df["Net Property, Plant & Equipment"] * bs_fac).round(3)
    out["TotalAssets_EUR_M"] = (bs_df["Total Assets"] * bs_fac).round(3)
    out["UnearnedRevCurr_EUR_M"] = (bs_df["Unearned Revenue, Current"] * bs_fac).round(3)
    out["TotalDebt_EUR_M"] = (bs_df["Total Debt"] * bs_fac).round(3)
    out["NetDebt_EUR_M"] = (bs_df["Net Debt"] * bs_fac).round(3)
    out["CommonStock_EUR_M"] = (bs_df["Common Stock"] * bs_fac).round(3)
    out["RetainedEarnings_EUR_M"] = (bs_df["Retained Earnings"] * bs_fac).round(3)
    out["TreasuryStock_EUR_M"] = (bs_df["Treasury Stock"] * bs_fac).round(3)
    out["TotalEquity_EUR_M"] = (bs_df["Total Equity"] * bs_fac).round(3)
    out["SharesFiling_M"] = (bs_df["ECS Total Shares Outstanding on Filing Date (actual)"] / 1e6).round(3)
    out["CommonShares_M"] = (bs_df["ECS Total Common Shares Outstanding (actual)"] / 1e6).round(3)
    out["Employees"] = bs_df["Full Time Employees (actual)"]
    out["OrderBacklog_EUR_M"] = (bs_df["Order Backlog"] * bs_fac).round(3)
    out["NOLCarryforward_EUR_M"] = (bs_df["Total NOL Carryforward"] * bs_fac).round(3)

    # Cash Flow Statement (EUR Millions)
    out["CFO_EUR_M"] = (cf_df["Cash from Ops."] * is_fac).round(3)
    out["CapEx_EUR_M"] = (cf_df["Capital Expenditure"].abs() * is_fac).round(3)
    out["DividendsPaid_EUR_M"] = (cf_df["Total Dividends Paid"].abs() * is_fac).round(3)
    out["ShareBuybacks_EUR_M"] = (cf_df["Repurchase of Common Stock"].abs().fillna(0) * is_fac).round(3)
    out["TotalPayout_EUR_M"] = (out["DividendsPaid_EUR_M"] + out["ShareBuybacks_EUR_M"]).round(3)
    out["CashTaxesPaid_EUR_M"] = (cf_df["Cash Taxes Paid"] * is_fac).round(3)
    out["FCF_EUR_M"] = (cf_df["Levered Free Cash Flow"] * is_fac).round(3)

    # Profitability & Key Financialization Ratios (%)
    out["ROA_Pct"] = fh_df["Return on Assets"].round(3)
    out["ROC_Pct"] = fh_df["Return on Capital"].round(3)
    out["ROE_Pct"] = fh_df["Return on Equity"].round(3)
    out["EBITDAMargin_Pct"] = fh_df["EBITDA Margin"].round(3)
    out["NetMargin_Pct"] = fh_df["Net Income Margin"].round(3)

    out["CashAndSTI_PctAssets"] = (out["CashAndSTI_EUR_M"] / out["TotalAssets_EUR_M"] * 100).round(2)
    out["TotalFinAssets_PctAssets"] = (out["TotalFinAssets_EUR_M"] / out["TotalAssets_EUR_M"] * 100).round(2)
    out["NetPPE_PctAssets"] = (out["NetPPE_EUR_M"] / out["TotalAssets_EUR_M"] * 100).round(2)
    out["UnearnedRev_PctAssets"] = (out["UnearnedRevCurr_EUR_M"] / out["TotalAssets_EUR_M"] * 100).round(2)
    out["FinInc_PctEBIT"] = np.where(
        out["EBIT_EUR_M"] > 0, (out["FinancialIncome_EUR_M"] / out["EBIT_EUR_M"] * 100).round(2), np.nan
    )
    out["FinInc_PctNetInc"] = np.where(
        out["NetIncome_EUR_M"] > 0, (out["FinancialIncome_EUR_M"] / out["NetIncome_EUR_M"] * 100).round(2), np.nan
    )
    out["TotalPayout_PctNetInc"] = np.where(
        out["NetIncome_EUR_M"] > 0, (out["TotalPayout_EUR_M"] / out["NetIncome_EUR_M"] * 100).round(2), np.nan
    )
    out["CapEx_PctNetInc"] = np.where(
        out["NetIncome_EUR_M"] > 0, (out["CapEx_EUR_M"] / out["NetIncome_EUR_M"] * 100).round(2), np.nan
    )

    return out


def build_ownership_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse S&P Capital IQ Ownership History (2004–2025 + Latest) and classify holders into:
      1. Sind International S.R.L. & Family/Executive Insiders
      2. Italian Institutional Investors (SGRs, domestic asset managers)
      3. Foreign Active / Value Funds
      4. Passive Indexers (Vanguard, BlackRock, Dimensional, State Street, etc.)
      5. Retail / Unallocated Float
    """
    raw = pd.read_excel(FILES["own"], sheet_name=0, header=None)
    headers = raw.iloc[14].tolist()
    date_cols = headers[2:25]
    period_labels = ["2026_Latest" if str(d) == "Latest" else str(d).replace("12/31/", "") for d in date_cols]

    holders_df = raw.iloc[15:].copy()
    holders_df.columns = ["Holder", "MI_Key"] + period_labels + ["Activist"]
    for p in period_labels:
        holders_df[p] = pd.to_numeric(holders_df[p], errors="coerce").fillna(0.0)

    insider_kw = "Benedetti|Alzetta|De Colle|Cappugi|Angeli|Tomasin|Venturini|Ravasio|Martegani|Piussi"
    italian_kw = (
        "SGR|S\\.G\\.R|SpA|S\\.p\\.A|S\\.p\\.a|Fideuram|Anima|Azimut|Eurizon|Generali|"
        "Mediolanum|Arca|Kairos|Intermonte|Pramerica|Anthilia|Finlabo|Cassa Lombarda|Marzotto"
    )
    passive_kw = "Vanguard|BlackRock|Dimensional|State Street|Geode|Invesco Capital|Schwab|DBX|First Trust"

    summary_rows = []
    for p in period_labels:
        vals = holders_df[p]
        names = holders_df["Holder"].astype(str)
        pos_mask = vals > 0

        is_sind = names.str.contains("Sind International", case=False, regex=True)
        is_insider = (~is_sind) & names.str.contains(insider_kw, case=False, regex=True)
        is_inst = (~is_sind) & (~is_insider) & pos_mask

        is_ital = is_inst & names.str.contains(italian_kw, case=False, regex=True)
        is_pass = is_inst & (~is_ital) & names.str.contains(passive_kw, case=False, regex=True)
        is_for_active = is_inst & (~is_ital) & (~is_pass)

        sind_pct = vals[is_sind].sum()
        ins_pct = vals[is_insider].sum()
        ital_pct = vals[is_ital].sum()
        pass_pct = vals[is_pass].sum()
        for_pct = vals[is_for_active].sum()
        tot_inst = ital_pct + pass_pct + for_pct
        tot_tracked = sind_pct + ins_pct + tot_inst
        inst_hhi = (vals[is_inst] ** 2).sum()

        summary_rows.append(
            {
                "Period": p,
                "SindInternational_Pct": round(sind_pct, 2),
                "FamilyAndInsiders_Pct": round(sind_pct + ins_pct, 2),
                "ItalianInstitutions_Pct": round(ital_pct, 2),
                "ForeignActiveInst_Pct": round(for_pct, 2),
                "PassiveIndexInst_Pct": round(pass_pct, 2),
                "TotalInstitutional_Pct": round(tot_inst, 2),
                "TotalTracked_Pct": round(tot_tracked, 2),
                "UnallocatedFloat_Pct": round(max(0.0, 100.0 - tot_tracked), 2),
                "ActiveInstCount": int(is_inst.sum()),
                "InstHHI": round(inst_hhi, 2),
            }
        )

    own_summary_df = pd.DataFrame(summary_rows)

    # Top shareholders ever exceeding 1.5% CSO
    max_stake = holders_df[period_labels].max(axis=1)
    top_holders_df = holders_df.loc[max_stake >= 1.5, ["Holder", "Activist"] + period_labels[1:]].copy()
    return own_summary_df, top_holders_df


def build_period_summary(fin_df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-year regime averages in EUR Millions across five historical phases."""
    regimes = [
        ("1990-1998 (Pre-Euro & High Rates)", 1990, 1998),
        ("1999-2007 (Euro Adoption & Supercycle)", 1999, 2007),
        ("2008-2014 (Post-GFC & Cash Build-up)", 2008, 2014),
        ("2015-2020 (Steel Downcycle & CapEx Push)", 2015, 2020),
        ("2021-2025 (Green Steel Boom & Rate Hikes)", 2021, 2025),
    ]
    rows = []
    for label, start_yr, end_yr in regimes:
        sub = fin_df.loc[start_yr:end_yr]
        rows.append(
            {
                "Period": label,
                "Avg_Revenue_EUR_M": round(sub["Revenue_EUR_M"].mean(), 2),
                "Avg_EBITDA_EUR_M": round(sub["EBITDA_EUR_M"].mean(), 2),
                "Avg_EBIT_EUR_M": round(sub["EBIT_EUR_M"].mean(), 2),
                "Avg_FinIncome_EUR_M": round(sub["FinancialIncome_EUR_M"].mean(), 2),
                "Avg_NetIncome_EUR_M": round(sub["NetIncome_EUR_M"].mean(), 2),
                "Avg_CashAndSTI_EUR_M": round(sub["CashAndSTI_EUR_M"].mean(), 2),
                "Avg_NetPPE_EUR_M": round(sub["NetPPE_EUR_M"].mean(), 2),
                "Avg_UnearnedRev_EUR_M": round(sub["UnearnedRevCurr_EUR_M"].mean(), 2),
                "Avg_CapEx_EUR_M": round(sub["CapEx_EUR_M"].mean(), 2),
                "Avg_Dividends_EUR_M": round(sub["DividendsPaid_EUR_M"].mean(), 2),
                "Avg_Buybacks_EUR_M": round(sub["ShareBuybacks_EUR_M"].mean(), 2),
                "Avg_CashSTI_PctAssets": round(sub["CashAndSTI_PctAssets"].mean(), 2),
                "Avg_NetPPE_PctAssets": round(sub["NetPPE_PctAssets"].mean(), 2),
                "Avg_FinInc_PctEBIT": round(sub["FinInc_PctEBIT"].mean(), 2),
                "Avg_TotalPayout_PctNetInc": round(sub["TotalPayout_PctNetInc"].mean(), 2),
                "Avg_EffectiveTaxRate_Pct": round(sub["EffectiveTaxRate_Pct"].mean(), 2),
                "Avg_ROE_Pct": round(sub["ROE_Pct"].mean(), 2),
                "Avg_Employees": round(sub["Employees"].mean(), 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("Building Danieli EUR Financial Dataset (1990–2025)...")
    fin_df = build_financial_dataset()
    fin_df.to_csv(OUTPUT_DIR / "danieli_financials_eur_1990_2025.csv")

    print("Building Multi-Period Regime Summary (EUR Millions)...")
    period_df = build_period_summary(fin_df)
    period_df.to_csv(OUTPUT_DIR / "danieli_period_summary_eur.csv", index=False)

    print("Building Ownership Structure & Top Shareholders Datasets (2004–2025)...")
    own_summary_df, top_holders_df = build_ownership_datasets()
    own_summary_df.to_csv(OUTPUT_DIR / "danieli_ownership_summary_2004_2025.csv", index=False)
    top_holders_df.to_csv(OUTPUT_DIR / "danieli_top_shareholders.csv", index=False)

    # Dual-Class Voting Distortion Dataset (2005–2026_Latest)
    dual_df = own_summary_df[own_summary_df["Period"] != "2004"].copy()
    shares_map = fin_df["SharesFiling_M"].to_dict()
    shares_map["2026_Latest"] = shares_map.get(2025, 71.573)
    dual_df["Combined_CSO_Millions"] = dual_df["Period"].map(
        lambda p: shares_map.get(int(p) if str(p).isdigit() else p, 71.573)
    ).round(2)
    dual_df["Treasury_Buyback_Reduction_M"] = (81.305 - dual_df["Combined_CSO_Millions"]).round(2)
    dual_df["Sind_And_Family_CSO_Pct"] = dual_df["FamilyAndInsiders_Pct"]
    dual_df["Institutional_CSO_Pct"] = dual_df["TotalInstitutional_Pct"]
    dual_df["Implied_Family_Ord_Shares_M"] = (
        (dual_df["Sind_And_Family_CSO_Pct"] / 100.0) * dual_df["Combined_CSO_Millions"]
    ).round(3)
    dual_df["Family_Voting_Share_IssuedOrd_Pct"] = (
        dual_df["Implied_Family_Ord_Shares_M"] / 40.652283 * 100.0
    ).round(2)
    dual_df["Family_Voting_Share_NetOrd_Pct"] = (dual_df["Sind_And_Family_CSO_Pct"] * 2.0).clip(upper=78.5).round(2)
    dual_cols = [
        "Period",
        "Combined_CSO_Millions",
        "Treasury_Buyback_Reduction_M",
        "Sind_And_Family_CSO_Pct",
        "Institutional_CSO_Pct",
        "Implied_Family_Ord_Shares_M",
        "Family_Voting_Share_IssuedOrd_Pct",
        "Family_Voting_Share_NetOrd_Pct",
    ]
    dual_df[dual_cols].to_csv(OUTPUT_DIR / "danieli_dual_class_voting.csv", index=False)

    print("\n=== MULTI-PERIOD REGIME SUMMARY (EUR MILLIONS) ===")
    print(period_df.to_string(index=False))

    print("\n=== OWNERSHIP STRUCTURE EVOLUTION (2005–2026_Latest) ===")
    print(own_summary_df[own_summary_df["Period"] != "2004"].to_string(index=False))


if __name__ == "__main__":
    main()
