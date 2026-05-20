from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _fmt_feature_name(name):
    return (
        str(name)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("_l1", "")
        .replace("_", " ")
        .replace("ff ", "Fama-French ")
        .title()
    )


def plot_esg_distribution(sample):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=220)
    sns.kdeplot(data=sample, x="ESG_l1", hue="region", common_norm=False, fill=True, alpha=0.28, ax=ax)
    ax.set_title("Lagged ESG score distribution by region")
    ax.set_xlabel("Lagged ESG score")
    ax.set_ylabel("Density")
    fig.savefig(FIGURES / "fig1_esg_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_pillar_trends(sample):
    trend = (
        sample.groupby("year")[["ESG_l1", "E_l1", "S_l1", "G_l1"]]
        .mean()
        .reset_index()
        .melt("year", var_name="series", value_name="score")
    )
    trend["series"] = trend["series"].str.replace("_l1", "", regex=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=220)
    sns.lineplot(data=trend, x="year", y="score", hue="series", marker="o", ax=ax)
    ax.set_title("ESG and pillar trends")
    ax.set_xlabel("Return year")
    ax.set_ylabel("Lagged score")
    fig.savefig(FIGURES / "fig2_pillar_trends.png", bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(pooled, fe):
    rows = []
    terms = ["z_ESG_l1", "z_E_l1", "z_S_l1", "z_G_l1", "srs"]
    for source, frame in [("Pooled OLS", pooled), ("Firm-year FE", fe)]:
        sub = frame[(frame["outcome"] == "region_return") & (frame["term"].isin(terms))].copy()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "model": source,
                    "term": row["term"].replace("z_", "").replace("_l1", "").replace("srs", "SRS"),
                    "coef": row["coef"],
                    "lo": row["coef"] - 1.96 * row["se"],
                    "hi": row["coef"] + 1.96 * row["se"],
                }
            )
    coef = pd.DataFrame(rows).drop_duplicates(["model", "term"])
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=220)
    sns.pointplot(data=coef, y="term", x="coef", hue="model", dodge=0.45, linestyle="none", ax=ax)
    for _, row in coef.iterrows():
        ytick = list(coef["term"].drop_duplicates()).index(row["term"])
        offset = -0.12 if row["model"] == "Pooled OLS" else 0.12
        ax.plot([row["lo"], row["hi"]], [ytick + offset, ytick + offset], color="0.25", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("ESG coefficients for region-adjusted returns")
    ax.set_xlabel("Standardised coefficient with approximate 95% CI")
    ax.set_ylabel("")
    fig.savefig(FIGURES / "fig3_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def plot_ml_rmse(ml):
    reg = ml[ml["task"] == "regression"].sort_values("rmse").head(14).copy()
    reg["label"] = reg["model"] + " | " + reg["feature_set"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=220)
    sns.barplot(data=reg, y="label", x="rmse", color="#3A6EA5", ax=ax)
    ax.set_title("Walk-forward return-prediction RMSE")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("")
    fig.savefig(FIGURES / "fig4_rmse.png", bbox_inches="tight")
    plt.close(fig)


def plot_ml_auc(ml):
    cls = ml[ml["task"] == "classification"].sort_values("auc", ascending=False).head(14).copy()
    cls["label"] = cls["model"] + " | " + cls["feature_set"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=220)
    sns.barplot(data=cls, y="label", x="auc", color="#2F855A", ax=ax)
    ax.axvline(0.5, color="black", lw=1)
    ax.set_title("Walk-forward direction-prediction AUC")
    ax.set_xlabel("AUC")
    ax.set_ylabel("")
    fig.savefig(FIGURES / "fig5_auc.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(feature_importance):
    if feature_importance.empty:
        return
    imp = (
        feature_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(18)
    )
    imp["feature"] = imp["feature"].map(_fmt_feature_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=220)
    sns.barplot(data=imp, y="feature", x="importance", color="#B7791F", ax=ax)
    ax.set_title("Tree-model feature importance")
    ax.set_xlabel("Mean importance")
    ax.set_ylabel("")
    fig.savefig(FIGURES / "fig6_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_spreads(portfolio):
    top = portfolio.sort_values("mean_spread", ascending=False).head(10).copy()
    top["label"] = top["model"] + " | " + top["feature_set"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=220)
    sns.barplot(data=top, y="label", x="mean_spread", color="#6C5B7B", ax=ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Top-minus-bottom quintile portfolio spread")
    ax.set_xlabel("Mean raw return spread")
    ax.set_ylabel("")
    fig.savefig(FIGURES / "fig7_portfolio.png", bbox_inches="tight")
    plt.close(fig)


def main():
    sample = pd.read_csv(TABLES / "model_sample.csv", low_memory=False)
    pooled = pd.read_csv(TABLES / "clustered_ols.csv")
    fe = pd.read_csv(TABLES / "panel_fe.csv")
    ml = pd.read_csv(TABLES / "ml_aggregate_results.csv")
    feature_importance = pd.read_csv(TABLES / "feature_importance.csv")
    portfolio = pd.read_csv(TABLES / "portfolio_aggregate.csv")

    plot_esg_distribution(sample)
    plot_pillar_trends(sample)
    plot_coefficients(pooled, fe)
    plot_ml_rmse(ml)
    plot_ml_auc(ml)
    plot_feature_importance(feature_importance)
    plot_portfolio_spreads(portfolio)

    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
