"""
plots.py — Publication-Ready EUR Figures for the Danieli Financialization & Governance Report
==============================================================================================
Generates all 8 figures in `plots/` from the cleaned EUR CSV datasets in `output/`:
  1. fig1_ownership_evolution.png       — Ownership Structure Breakdown (2005–2025)
  2. fig2_top_institutional_holders.png — Top Non-Family Institutional Blockholders (2005–2025)
  3. fig3_asset_financialization.png    — Cash & Short-Term Investments vs. Net PP&E (€M & % Assets)
  4. fig4_profit_financialization.png   — Financial Income vs. Operating Income / EBIT (€M & %)
  5. fig5_retain_invest_vs_distribute.png — CapEx vs. Dividends & Share Buybacks (€M & % Net Income)
  6. fig6_real_accumulation_profitability.png — Employment Growth vs. ROE, ROC & EBITDA Margin
  7. fig7_customer_advances_treasury.png — Unearned Revenue (Customer Advances) vs. Liquid Assets (€M)
  8. fig8_tax_rate_and_carryforwards.png — Effective Tax Rate (%), Tax Expense & NOL Carryforwards (€M)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 8),
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})


def load_processed_data():
    fin = pd.read_csv(OUTPUT_DIR / "danieli_financials_eur_1990_2025.csv")
    own = pd.read_csv(OUTPUT_DIR / "danieli_ownership_summary_2004_2025.csv")
    own = own[own["Period"] != "2004"].copy()
    own["Label"] = own["Period"].str.replace("_Latest", "*", regex=False)
    top_h = pd.read_csv(OUTPUT_DIR / "danieli_top_shareholders.csv")
    return fin, own, top_h


def plot_fig1_ownership(own: pd.DataFrame):
    fig, ax = plt.subplots()
    x = np.arange(len(own))
    categories = [
        ("FamilyAndInsiders_Pct", "Sind International & Family Insiders (~34–39%)", "#1b4f72"),
        ("ItalianInstitutions_Pct", "Italian Institutional Funds (SGRs)", "#2874a6"),
        ("ForeignActiveInst_Pct", "Foreign Active / Value Funds", "#d35400"),
        ("PassiveIndexInst_Pct", "Passive Indexers (Vanguard/BlackRock/DFA)", "#27ae60"),
        ("UnallocatedFloat_Pct", "Retail / Unallocated Float", "#d5dbdb"),
    ]
    bottom = np.zeros(len(own))
    for col, label, color in categories:
        vals = own[col].values
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.72)
        bottom += vals

    ax.set_title("Figure 1: Evolution of Danieli's Ownership Structure (2005–2026*)")
    ax.set_ylabel("Share of Common Shares Outstanding (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(own["Label"], rotation=45)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=9.5)
    fig.text(0.01, 0.01, "Source: S&P Capital IQ Ownership History. *2026* = Latest snapshot as of 11-Nov-2026 (all prior years are 12/31 year-end figures).", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(PLOTS_DIR / "fig1_ownership_evolution.png")
    plt.close(fig)


def plot_fig2_top_institutions(own: pd.DataFrame, top_h: pd.DataFrame):
    fig, ax = plt.subplots()
    period_cols = own["Period"].tolist()
    x = np.arange(len(period_cols))
    targets = [
        ("Cobas", "Cobas Asset Mgmt (Value)", "#c0392b"),
        ("Capital Research", "Capital Research & Mgmt", "#8e44ad"),
        ("Invesco Ltd", "Invesco Ltd.", "#d35400"),
        ("Fideuram", "Fideuram - Intesa Sanpaolo SGR", "#2980b9"),
        ("Anima SGR", "Anima SGR SpA", "#16a085"),
        ("Vanguard Group", "Vanguard Group", "#27ae60"),
        ("JP Morgan", "JP Morgan Asset Mgmt", "#2c3e50"),
        ("Norges", "Norges Bank (SWF)", "#f39c12"),
    ]
    for pattern, label, color in targets:
        row = top_h[top_h["Holder"].str.contains(pattern, case=False, na=False)]
        if not row.empty:
            vals = row.iloc[0][period_cols].astype(float).values
            ax.plot(x, vals, marker="o", linewidth=2.4, label=label, color=color)

    ax.set_title("Figure 2: Trajectories of Major Non-Family Institutional Blockholders (2005–2026*)")
    ax.set_ylabel("Ownership Stake (% of CSO)")
    ax.set_xticks(x)
    ax.set_xticklabels(own["Label"], rotation=45)
    ax.set_ylim(0, 6.0)
    ax.legend(loc="upper left", ncol=2)
    fig.text(0.01, 0.01, "Source: S&P Capital IQ Ownership History. *2026* = Latest snapshot as of 11-Nov-2026 (all prior years are 12/31 year-end figures).", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(PLOTS_DIR / "fig2_top_institutional_holders.png")
    plt.close(fig)


def plot_fig3_asset_financialization(fin: pd.DataFrame):
    fig, ax1 = plt.subplots()
    x = np.arange(len(fin))
    w = 0.38
    ax1.bar(x - w / 2, fin["CashAndSTI_EUR_M"], width=w, color="#2980b9", alpha=0.8, label="Cash & ST Investments (€M, Left)")
    ax1.bar(x + w / 2, fin["NetPPE_EUR_M"], width=w, color="#7f8c8d", alpha=0.8, label="Net PP&E (€M, Left)")
    ax1.set_ylabel("EUR Millions (€M)")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(fin["Year"].iloc[::2])

    ax2 = ax1.twinx()
    ax2.plot(x, fin["CashAndSTI_PctAssets"], color="#c0392b", marker="o", linewidth=2.5, label="Cash & STI / Total Assets (%, Right)")
    ax2.plot(x, fin["NetPPE_PctAssets"], color="#27ae60", marker="s", linewidth=2.5, label="Net PP&E / Total Assets (%, Right)")
    ax2.set_ylabel("Share of Total Assets (%)")
    ax2.set_ylim(0, 45)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)
    ax1.set_title("Figure 3: Balance Sheet Financialization — Liquid Financial Assets vs. Net PP&E (1990–2025)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig3_asset_financialization.png")
    plt.close(fig)


def plot_fig4_profit_financialization(fin: pd.DataFrame):
    fig, ax1 = plt.subplots()
    x = np.arange(len(fin))
    w = 0.38
    ax1.bar(x - w / 2, fin["EBIT_EUR_M"], width=w, color="#2c3e50", alpha=0.85, label="Operating Income / EBIT (€M, Left)")
    ax1.bar(x + w / 2, fin["FinancialIncome_EUR_M"], width=w, color="#e67e22", alpha=0.9, label="Interest & Investment Income (€M, Left)")
    ax1.set_ylabel("EUR Millions (€M)")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(fin["Year"].iloc[::2])

    ax2 = ax1.twinx()
    ratio_capped = fin["FinInc_PctEBIT"].clip(upper=120)
    ax2.plot(x, ratio_capped, color="#c0392b", marker="o", linewidth=2.5, label="Financial Income / EBIT Ratio (%, Right, capped at 120%)")
    ax2.set_ylabel("Financial Income / EBIT (%)")
    ax2.set_ylim(0, 125)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Figure 4: Income Financialization — Financial Income vs. Operating Income (1990–2025)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4_profit_financialization.png")
    plt.close(fig)


def plot_fig5_retain_vs_distribute(fin: pd.DataFrame):
    sub = fin[fin["Year"] >= 1997].copy()
    fig, ax1 = plt.subplots()
    x = np.arange(len(sub))
    w = 0.38
    div = sub["DividendsPaid_EUR_M"].fillna(0)
    buy = sub["ShareBuybacks_EUR_M"].fillna(0)

    ax1.bar(x - w / 2, sub["CapEx_EUR_M"], width=w, color="#27ae60", alpha=0.85, label="Capital Expenditure (CapEx, €M)")
    ax1.bar(x + w / 2, div, width=w, color="#2980b9", alpha=0.9, label="Dividends Paid (€M)")
    ax1.bar(x + w / 2, buy, bottom=div, width=w, color="#c0392b", alpha=0.9, label="Share Buybacks (€M)")
    ax1.set_ylabel("EUR Millions (€M)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["Year"], rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x, sub["TotalPayout_PctNetInc"].clip(upper=60), color="#8e44ad", marker="o", linewidth=2.5, label="(Dividends + Buybacks) / Net Income (%, Right)")
    ax2.set_ylabel("Total Payout / Net Income (%)")
    ax2.set_ylim(0, 60)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)
    ax1.set_title("Figure 5: 'Retain & Invest' vs. 'Downsize & Distribute' (1997–2025)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig5_retain_invest_vs_distribute.png")
    plt.close(fig)


def plot_fig6_accumulation_profitability(fin: pd.DataFrame):
    fig, ax1 = plt.subplots()
    x = np.arange(len(fin))
    ax1.bar(x, fin["Employees"], width=0.6, color="#3498db", alpha=0.55, label="Full-Time Employees (Left)")
    ax1.set_ylabel("Full-Time Employees (Headcount)")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(fin["Year"].iloc[::2])
    ax1.set_ylim(0, 12000)

    ax2 = ax1.twinx()
    ax2.plot(x, fin["ROE_Pct"], color="#c0392b", marker="o", linewidth=2.4, label="Return on Equity (ROE %, Right)")
    ax2.plot(x, fin["ROC_Pct"], color="#27ae60", marker="s", linewidth=2.4, label="Return on Capital (ROC %, Right)")
    ax2.plot(x, fin["EBITDAMargin_Pct"], color="#d35400", marker="^", linewidth=2.4, label="EBITDA Margin (%, Right)")
    ax2.set_ylabel("Profitability & Margins (%)")
    ax2.set_ylim(-2, 30)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)
    ax1.set_title("Figure 6: Industrial Accumulation, Employment & Profitability Dynamics (1990–2025)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig6_real_accumulation_profitability.png")
    plt.close(fig)


def plot_fig7_customer_advances(fin: pd.DataFrame):
    sub = fin[fin["Year"] >= 1997].copy()
    fig, ax = plt.subplots()
    x = np.arange(len(sub))
    ax.plot(x, sub["UnearnedRevCurr_EUR_M"], color="#c0392b", marker="o", linewidth=3.0, label="Unearned Revenue / Customer Contract Advances (€2,871M in 2025)")
    ax.plot(x, sub["CashAndSTI_EUR_M"], color="#2980b9", marker="s", linewidth=3.0, label="Gross Cash & Short-Term Investments (€2,773M in 2025)")
    ax.plot(x, -sub["NetDebt_EUR_M"], color="#27ae60", marker="^", linewidth=2.6, label="Net Cash Position (-Net Debt: €2,180M in 2025)")
    ax.set_title("Figure 7: The 'Customer-Financed Treasury' — Customer Advances vs. Liquid Assets (1997–2025)")
    ax.set_ylabel("EUR Millions (€M)")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Year"], rotation=45)
    ax.set_ylim(0, 3200)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig7_customer_advances_treasury.png")
    plt.close(fig)


def plot_fig8_tax_strategy(fin: pd.DataFrame):
    fig, ax1 = plt.subplots()
    x = np.arange(len(fin))
    w = 0.38
    ax1.bar(x - w / 2, fin["TaxExpense_EUR_M"], width=w, color="#34495e", alpha=0.85, label="Income Tax Expense (€M, Left)")
    ax1.bar(x + w / 2, fin["NOLCarryforward_EUR_M"], width=w, color="#f39c12", alpha=0.85, label="Total NOL Tax-Loss Carryforwards (€M, Left)")
    ax1.set_ylabel("EUR Millions (€M)")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(fin["Year"].iloc[::2])
    ax1.set_ylim(0, 360)

    ax2 = ax1.twinx()
    ax2.plot(x, fin["EffectiveTaxRate_Pct"].clip(lower=0, upper=60), color="#c0392b", marker="o", linewidth=2.5, label="Effective Tax Rate (%, Right)")
    ax2.set_ylabel("Effective Tax Rate (%)")
    ax2.set_ylim(0, 60)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Figure 8: Corporate Tax Strategy, Effective Tax Rate & Tax-Loss Carryforwards (1990–2025)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig8_tax_rate_and_carryforwards.png")
    plt.close(fig)


def plot_fig9_dual_class_voting():
    dual = pd.read_csv(OUTPUT_DIR / "danieli_dual_class_voting.csv")
    dual["Label"] = dual["Period"].str.replace("_Latest", "*", regex=False)
    x = np.arange(len(dual))

    fig, ax1 = plt.subplots()
    ax1.fill_between(
        x,
        dual["Sind_And_Family_CSO_Pct"],
        dual["Family_Voting_Share_IssuedOrd_Pct"],
        color="#c0392b",
        alpha=0.18,
        label="Dual-Class Control Wedge (Voting % minus CSO %)",
    )
    ax1.plot(
        x,
        dual["Family_Voting_Share_NetOrd_Pct"],
        color="#922b21",
        linestyle="--",
        marker="o",
        linewidth=2.3,
        label="Sind & Family Voting Rights (% Net Ordinary Shares: up to 78.5%)",
    )
    ax1.plot(
        x,
        dual["Family_Voting_Share_IssuedOrd_Pct"],
        color="#c0392b",
        marker="o",
        linewidth=3.0,
        label="Sind & Family Voting Rights (% Issued Ordinary Capital: ~68–69%)",
    )
    ax1.plot(
        x,
        dual["Sind_And_Family_CSO_Pct"],
        color="#1b4f72",
        marker="s",
        linewidth=2.8,
        label="Sind & Family Apparent Stake (% Combined CSO: ~34–39%)",
    )
    ax1.plot(
        x,
        dual["Institutional_CSO_Pct"],
        color="#d35400",
        marker="^",
        linewidth=2.5,
        label="Total Institutional Investors (% Combined CSO: peak 37.4% in 2018)",
    )
    ax1.axhline(66.67, color="#8e44ad", linestyle="-.", linewidth=1.6, label="66.7% Extraordinary AGM Supermajority")
    ax1.axhline(50.0, color="#566573", linestyle="--", linewidth=1.4, label="50.0% Ordinary AGM Majority")
    ax1.set_ylabel("Share of Capital / Voting Rights (%)")
    ax1.set_ylim(0, 85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dual["Label"], rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(
        x,
        dual["Combined_CSO_Millions"],
        width=0.45,
        color="#95a5a6",
        alpha=0.3,
        label="Combined Shares Outstanding net of Buybacks (Millions, Right)",
    )
    ax2.set_ylabel("Combined Shares Outstanding (Millions, net of Treasury)")
    ax2.set_ylim(65, 85)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2, fontsize=9)
    ax1.set_title("Figure 9: Dual-Class Share Distortion — Family Voting Control vs. Apparent % of CSO (2005–2026*)")
    fig.text(0.01, 0.01, "Source: S&P Capital IQ Ownership History & Balance Sheet. *2026* = Latest ownership snapshot (11-Nov-2026) paired with FY 2025 share count (71.57M shares).", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(PLOTS_DIR / "fig9_dual_class_voting_distortion.png")
    plt.close(fig)


def main():
    fin, own, top_h = load_processed_data()
    plot_fig1_ownership(own)
    plot_fig2_top_institutions(own, top_h)
    plot_fig3_asset_financialization(fin)
    plot_fig4_profit_financialization(fin)
    plot_fig5_retain_vs_distribute(fin)
    plot_fig6_accumulation_profitability(fin)
    plot_fig7_customer_advances(fin)
    plot_fig8_tax_strategy(fin)
    plot_fig9_dual_class_voting()
    print(f"All 9 EUR figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
