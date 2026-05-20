from pathlib import Path
import io
import json
import re
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = LGBMRegressor = None


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / "data" / "Asia_us_eu_1.csv"
OUT = PROJECT / "results" / "tables"
EXT = OUT / "external"
OUT.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)

YEARS = range(2017, 2025)
META = {
    "Identifier (RIC)": "firm",
    "Company Name": "company",
    "Country of Exchange": "country",
    "Khu v_c": "region",
    "Developed Asia": "developed_asia",
    "GICS Industry Name": "industry",
    "TRBC Economic Sector Name": "sector",
}
SERIES = [
    "ESG", "E", "S", "G", "ESGC", "OM", "YTD", "ROA", "ROE", "MTB", "PTB",
    "LEV", "PROF", "RDINT", "CASH", "QUICK", "GR", "beta", "AZDis",
    "TA", "MC", "Rev", "Debt", "TL", "NI", "FA"
]
FF_URLS = {
    "US": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "Europe": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_CSV.zip",
    "AsiaPacExJapan": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_CSV.zip",
    "Japan": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_CSV.zip",
    "Developed": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_5_Factors_CSV.zip",
}


def clean_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s or s in {"#DIV/0!", "#VALUE!", "#N/A", "nan", "NaN"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    pct = s.endswith("%")
    s = s.replace("%", "").replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if neg:
        v = -v
    if pct:
        v /= 100.0
    return v


def winsor(s, lo=0.01, hi=0.99):
    qlo, qhi = s.quantile([lo, hi])
    return s.clip(qlo, qhi)


def zscore(s):
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - s.mean()) / sd


def parse_ff_zip(name, url):
    target = EXT / f"{name}.zip"
    if not target.exists():
        target.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    with zipfile.ZipFile(target) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("latin1")
    lines = raw.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    end_idx = next(i for i in range(header_idx + 1, len(lines)) if not re.match(r"^\d{6},", lines[i]))
    text = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={df.columns[0]: "yyyymm"})
    df = df[df["yyyymm"].astype(str).str.match(r"^\d{6}$")].copy()
    df["year"] = df["yyyymm"].astype(str).str[:4].astype(int)
    for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    annual = df.groupby("year")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].sum().reset_index()
    annual["ff_source"] = name
    return annual


def get_ff_factors():
    frames = []
    sources = []
    for name, url in FF_URLS.items():
        annual = parse_ff_zip(name, url)
        frames.append(annual)
        sources.append({"dataset": name, "url": url, "retrieval_note": "Downloaded from Kenneth R. French Data Library FTP zip endpoint and aggregated from monthly percent factors to annual decimal sums."})
    ff = pd.concat(frames, ignore_index=True)
    ff.to_csv(OUT / "fama_french_annual_factors.csv", index=False)
    (OUT / "external_data_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    return ff


def make_panel():
    raw = pd.read_csv(DATA, header=1, low_memory=False)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    raw = raw.rename(columns=META)
    raw = raw.dropna(subset=["firm"]).copy()
    rows = []
    for _, r in raw.iterrows():
        base = {v: r.get(v, np.nan) for v in META.values()}
        for year in YEARS:
            row = dict(base)
            row["year"] = int(year)
            for var in SERIES:
                row[var] = clean_number(r.get(f"{var}{year}", np.nan))
            rows.append(row)
    p = pd.DataFrame(rows).sort_values(["firm", "year"])
    p["log_mcap"] = np.log1p(p["MC"].clip(lower=0))
    p["log_assets"] = np.log1p(p["TA"].clip(lower=0))
    p["valuation"] = p["MTB"].combine_first(p["PTB"])
    p["debt_assets"] = p["Debt"] / p["TA"].replace(0, np.nan)
    p["fixed_assets_assets"] = p["FA"] / p["TA"].replace(0, np.nan)
    lag_vars = ["ESG", "E", "S", "G", "ESGC", "ROA", "ROE", "valuation", "LEV", "PROF", "RDINT", "CASH", "QUICK", "GR", "beta", "AZDis", "log_mcap", "log_assets", "debt_assets", "fixed_assets_assets"]
    for v in lag_vars:
        p[f"{v}_l1"] = p.groupby("firm")[v].shift(1)
    p["ESG_momentum_l1"] = p.groupby("firm")["ESG"].transform(lambda s: s.diff().shift(1))
    p["pillar_dispersion_l1"] = p[["E_l1", "S_l1", "G_l1"]].std(axis=1)
    p["EG_gap_l1"] = p["E_l1"] - p["G_l1"]
    p["return_raw"] = p["YTD"]
    p["region_year_return"] = p.groupby(["region", "year"])["return_raw"].transform("mean")
    p["sector_year_return"] = p.groupby(["sector", "year"])["return_raw"].transform("mean")
    p["excess_return_region"] = p["return_raw"] - p["region_year_return"]
    p["excess_return_sector"] = p["return_raw"] - p["sector_year_return"]
    p["outperform_region"] = (p["excess_return_region"] > 0).astype(int)
    return raw, p


def add_ff(sample, ff):
    sample = sample.copy()
    def source(row):
        if row["region"] == "US":
            return "US"
        if row["region"] == "EU":
            return "Europe"
        if str(row.get("country", "")).lower() == "japan":
            return "Japan"
        if row["region"] == "Asia":
            return "AsiaPacExJapan"
        return "Developed"
    sample["ff_source"] = sample.apply(source, axis=1)
    out = sample.merge(ff, on=["year", "ff_source"], how="left")
    out = out.rename(columns={"Mkt-RF": "ff_mkt_rf", "SMB": "ff_smb", "HML": "ff_hml", "RMW": "ff_rmw", "CMA": "ff_cma", "RF": "ff_rf"})
    return out


def model_sample(panel, ff):
    s = panel[(panel["year"] >= 2018) & (panel["year"] <= 2024)].copy()
    s = s.dropna(subset=["firm", "region", "sector", "return_raw", "excess_return_region"])
    core = ["ESG_l1", "E_l1", "S_l1", "G_l1", "ROA_l1", "LEV_l1", "log_mcap_l1", "valuation_l1", "beta_l1"]
    s = s[s[core].notna().sum(axis=1) >= 6].copy()
    s = add_ff(s, ff)
    cont = ["excess_return_region", "excess_return_sector", "ROA", "valuation", "ESG_l1", "E_l1", "S_l1", "G_l1", "ESGC_l1", "ROA_l1", "ROE_l1", "valuation_l1", "LEV_l1", "PROF_l1", "RDINT_l1", "CASH_l1", "QUICK_l1", "GR_l1", "beta_l1", "AZDis_l1", "log_mcap_l1", "log_assets_l1", "debt_assets_l1", "fixed_assets_assets_l1", "ESG_momentum_l1", "pillar_dispersion_l1", "EG_gap_l1", "ff_mkt_rf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_rf"]
    for c in cont:
        if c in s:
            s[f"w_{c}"] = winsor(s[c].astype(float))
            s[f"z_{c}"] = zscore(s[f"w_{c}"])
    quality = (s["z_ROA_l1"].fillna(0) + s["z_CASH_l1"].fillna(0) - s["z_LEV_l1"].fillna(0) - s["z_AZDis_l1"].fillna(0)) / 4
    s["srs"] = zscore(0.30*s["z_ESG_l1"].fillna(0) + 0.20*s["z_ESG_momentum_l1"].fillna(0) + 0.20*s["z_E_l1"].fillna(0) - 0.15*s["z_pillar_dispersion_l1"].fillna(0) + 0.15*quality.fillna(0))
    return s


def clustered_ols(sample):
    specs = {
        "ESG": ["z_ESG_l1", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1"],
        "Pillars": ["z_E_l1", "z_S_l1", "z_G_l1", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1"],
        "SRS": ["srs", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1"],
        "ESG+FF": ["z_ESG_l1", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1", "z_ff_mkt_rf", "z_ff_smb", "z_ff_hml", "z_ff_rmw", "z_ff_cma"],
    }
    outcomes = {"region_return": "z_excess_return_region", "sector_return": "z_excess_return_sector", "ROA": "z_ROA", "valuation": "z_valuation"}
    rows = []
    for oname, y in outcomes.items():
        for sname, xs in specs.items():
            d = sample[[y, "firm", "year", "region", "sector"] + xs].replace([np.inf, -np.inf], np.nan).dropna()
            fit = smf.ols(f"{y} ~ {' + '.join(xs)} + C(year) + C(region) + C(sector)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["firm"]})
            for term in xs:
                rows.append({"family": "clustered pooled OLS", "outcome": oname, "spec": sname, "term": term, "coef": fit.params[term], "se": fit.bse[term], "t": fit.tvalues[term], "p": fit.pvalues[term], "n": int(fit.nobs), "r2": fit.rsquared})
    return pd.DataFrame(rows)


def panel_fe(sample):
    specs = {
        "ESG": ["z_ESG_l1", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1"],
        "Pillars": ["z_E_l1", "z_S_l1", "z_G_l1", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1"],
        "SRS": ["srs", "z_log_mcap_l1", "z_LEV_l1", "z_ROA_l1", "z_valuation_l1", "z_beta_l1"],
    }
    rows = []
    for oname, y in {"region_return": "z_excess_return_region", "ROA": "z_ROA", "valuation": "z_valuation"}.items():
        for sname, xs in specs.items():
            d = sample[["firm", "year", y] + xs].replace([np.inf, -np.inf], np.nan).dropna().set_index(["firm", "year"])
            fit = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
            for term in xs:
                if term in fit.params.index:
                    rows.append({"family": "firm-year FE", "outcome": oname, "spec": sname, "term": term, "coef": fit.params[term], "se": fit.std_errors[term], "t": fit.tstats[term], "p": fit.pvalues[term], "n": int(fit.nobs), "r2": fit.rsquared_within})
    return pd.DataFrame(rows)


def ml_models():
    regressors = {
        "Ridge": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=0.005, l1_ratio=0.15, max_iter=20000, random_state=7),
        "LinearSVR": LinearSVR(C=0.25, random_state=7, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=250, min_samples_leaf=25, max_features="sqrt", random_state=7, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=160, learning_rate=0.03, max_depth=2, random_state=7),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 16), alpha=0.01, early_stopping=True, random_state=7, max_iter=500),
    }
    classifiers = {
        "Logistic": LogisticRegression(C=0.5, max_iter=3000, random_state=7),
        "LinearSVC": LinearSVC(C=0.25, random_state=7, max_iter=10000),
        "RandomForest": RandomForestClassifier(n_estimators=250, min_samples_leaf=25, max_features="sqrt", random_state=7, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=160, learning_rate=0.03, max_depth=2, random_state=7),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 16), alpha=0.01, early_stopping=True, random_state=7, max_iter=500),
    }
    if XGBRegressor is not None:
        regressors["XGBoost"] = XGBRegressor(n_estimators=180, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=7, n_jobs=-1)
        classifiers["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=7, n_jobs=-1)
    if LGBMRegressor is not None:
        regressors["LightGBM"] = LGBMRegressor(n_estimators=180, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=7, n_jobs=-1, verbose=-1)
        classifiers["LightGBM"] = LGBMClassifier(n_estimators=180, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=7, n_jobs=-1, verbose=-1)
    return regressors, classifiers


def ml_eval(sample):
    fin = ["ROA_l1", "ROE_l1", "valuation_l1", "LEV_l1", "PROF_l1", "RDINT_l1", "CASH_l1", "QUICK_l1", "GR_l1", "beta_l1", "AZDis_l1", "log_mcap_l1", "debt_assets_l1", "fixed_assets_assets_l1"]
    esg = ["ESG_l1", "E_l1", "S_l1", "G_l1", "ESG_momentum_l1", "pillar_dispersion_l1", "EG_gap_l1", "srs"]
    ff = ["ff_mkt_rf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_rf"]
    cat = ["region", "sector"]
    sets = {"financial": fin, "ESG": esg, "financial+ESG": fin+esg, "financial+SRS": fin+["srs"], "financial+ESG+FF": fin+esg+ff}
    regressors, classifiers = ml_models()
    rows, preds, imps = [], [], []
    for year in [2021, 2022, 2023, 2024]:
        train, test = sample[sample.year < year].copy(), sample[sample.year == year].copy()
        ytr, yte = train["excess_return_region"].astype(float), test["excess_return_region"].astype(float)
        base_mean = ytr.mean()
        rmean = train.groupby("region")["excess_return_region"].mean()
        bp = test["region"].map(rmean).fillna(base_mean).to_numpy()
        rows.append({"task": "regression", "test_year": year, "feature_set": "baseline", "model": "regional historical mean", "rmse": mean_squared_error(yte, bp)**0.5, "mae": mean_absolute_error(yte, bp), "r2": r2_score(yte, bp), "direction_accuracy": accuracy_score(yte > 0, bp > 0), "auc": roc_auc_score(yte > 0, bp)})
        preds.append(pd.DataFrame({"firm": test.firm.values, "year": year, "region": test.region.values, "model": "baseline", "feature_set": "baseline", "pred": bp, "actual": yte.values, "raw_return": test.return_raw.values}))
        for fs, nums in sets.items():
            pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), nums), ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]), cat)])
            Xtr, Xte = train[nums+cat], test[nums+cat]
            for name, model in regressors.items():
                pipe = Pipeline([("pre", pre), ("model", model)])
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                rows.append({"task": "regression", "test_year": year, "feature_set": fs, "model": name, "rmse": mean_squared_error(yte, pred)**0.5, "mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred), "direction_accuracy": accuracy_score(yte > 0, pred > 0), "auc": roc_auc_score(yte > 0, pred)})
                if fs in {"financial+ESG", "financial+ESG+FF"} and name in {"LightGBM", "XGBoost", "RandomForest"}:
                    try:
                        names = pipe.named_steps["pre"].get_feature_names_out()
                        vals = getattr(pipe.named_steps["model"], "feature_importances_", None)
                        if vals is not None:
                            for f, v in zip(names, vals):
                                imps.append({"test_year": year, "model": name, "feature_set": fs, "feature": f, "importance": float(v)})
                    except Exception:
                        pass
                if fs in {"financial+ESG", "financial+SRS", "financial+ESG+FF"} and name in {"Ridge", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "MLP"}:
                    preds.append(pd.DataFrame({"firm": test.firm.values, "year": year, "region": test.region.values, "model": name, "feature_set": fs, "pred": pred, "actual": yte.values, "raw_return": test.return_raw.values}))
            ytrc, ytec = (ytr > 0).astype(int), (yte > 0).astype(int)
            for name, model in classifiers.items():
                pipe = Pipeline([("pre", pre), ("model", model)])
                pipe.fit(Xtr, ytrc)
                predc = pipe.predict(Xte)
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    score = pipe.predict_proba(Xte)[:, 1]
                elif hasattr(pipe.named_steps["model"], "decision_function"):
                    score = pipe.decision_function(Xte)
                else:
                    score = predc
                rows.append({"task": "classification", "test_year": year, "feature_set": fs, "model": name, "direction_accuracy": accuracy_score(ytec, predc), "f1": f1_score(ytec, predc), "auc": roc_auc_score(ytec, score)})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(imps)


def portfolio(preds):
    rows = []
    for (model, fs, year, region), d in preds.groupby(["model", "feature_set", "year", "region"]):
        if len(d) < 40:
            continue
        lo, hi = d.pred.quantile([0.2, 0.8])
        long = d[d.pred >= hi].raw_return.mean()
        short = d[d.pred <= lo].raw_return.mean()
        rows.append({"model": model, "feature_set": fs, "year": year, "region": region, "long_return": long, "short_return": short, "spread": long-short, "n": len(d)})
    out = pd.DataFrame(rows)
    agg = out.groupby(["model", "feature_set"]).agg(mean_long=("long_return", "mean"), mean_short=("short_return", "mean"), mean_spread=("spread", "mean"), positive_share=("spread", lambda x: float((x > 0).mean())), tests=("spread", "count")).reset_index()
    return out, agg.sort_values("mean_spread", ascending=False)


def main():
    ff = get_ff_factors()
    raw, panel = make_panel()
    sample = model_sample(panel, ff)
    pooled = clustered_ols(sample)
    fe = panel_fe(sample)
    ml, preds, imp = ml_eval(sample)
    port_y, port_agg = portfolio(preds)
    ml_agg = ml.groupby(["task", "feature_set", "model"], dropna=False).agg(rmse=("rmse", "mean"), mae=("mae", "mean"), r2=("r2", "mean"), direction_accuracy=("direction_accuracy", "mean"), f1=("f1", "mean"), auc=("auc", "mean")).reset_index()
    desc = sample[["return_raw", "excess_return_region", "excess_return_sector", "ROA", "valuation", "ESG_l1", "E_l1", "S_l1", "G_l1", "srs", "ff_mkt_rf", "ff_smb", "ff_hml"]].describe(percentiles=[.01, .1, .5, .9, .99]).T
    coverage = {"raw_firms": int(raw.firm.nunique()), "countries": int(raw.country.nunique()), "panel_firm_years": int(len(panel)), "model_firm_years": int(len(sample)), "regions_model": sample.region.value_counts().to_dict(), "years": [int(x) for x in sorted(sample.year.unique())], "fama_french_sources": FF_URLS}
    sample.to_csv(OUT / "model_sample.csv", index=False)
    pooled.to_csv(OUT / "clustered_ols.csv", index=False)
    fe.to_csv(OUT / "panel_fe.csv", index=False)
    ml.to_csv(OUT / "ml_yearly_results.csv", index=False)
    ml_agg.to_csv(OUT / "ml_aggregate_results.csv", index=False)
    preds.to_csv(OUT / "predictions.csv", index=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    port_y.to_csv(OUT / "portfolio_yearly.csv", index=False)
    port_agg.to_csv(OUT / "portfolio_aggregate.csv", index=False)
    desc.to_csv(OUT / "descriptive_statistics.csv")
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": coverage, "top_ml": ml_agg.sort_values(["task", "rmse", "auc"], ascending=[True, True, False]).head(12).to_dict("records"), "portfolio": port_agg.head(8).to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
