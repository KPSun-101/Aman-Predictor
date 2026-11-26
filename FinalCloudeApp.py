"""
Aman Rice Production Predictor - ONE FILE (Flask + HTML + CSS + JS)
Run:  python app.py     then open http://127.0.0.1:5000
Needs: pip install flask pandas numpy scikit-learn

HERO PHOTO: save a photo of a ripe Aman rice field as  hero.jpg  in the SAME folder as app.py.
(Optional) save a second photo as  about.jpg  for the "About Aman Rice" page.
Without them the site shows a golden-green gradient instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).parent


def first_existing(*candidates):
    paths = [Path(c) for c in candidates]
    return next((p for p in paths if p.exists()), paths[0])


# ------------------------------------------------------------------
# CONFIG - file paths (edit if needed)
# ------------------------------------------------------------------
DATA_FILE = first_existing(r"D:\Rice Reaserch\Rice+JJAS(total_rain).csv")
RESULTS_FILE = first_existing(r"D:\Project Purpose\Temperature_Feature_Comparison.csv")
WB_FILE = first_existing(
    r"D:\Rice Reaserch\World_Bank_JJAS_2023_2100.csv",
    r"D:\Project Purpose\World_Bank_JJAS_2023_2100.csv",
    BASE / "World_Bank_JJAS_2023_2100.csv",
    BASE / "data" / "World_Bank_JJAS_2023_2100.csv",
)

# HERO PHOTO (ripe Aman rice field). Easiest: save the photo as hero.jpg next to app.py.
# Or paste a direct image link (ending in .jpg/.jpeg/.png/.webp) here:
HERO_IMAGE_URL = ""

# Shown in the UI - set to the real model / scenario of your WBCCKP download
CLIMATE_MODEL = "WBCCKP CMIP6 (as provided in local dataset)"
CLIMATE_SCENARIO = "Scenario of local WBCCKP dataset"

YEAR_COL = "Year"
TARGET_COL = "production(LMT)"
RAIN, TMAX, TMIN, LAND = "Total_Rainfall_(mm)", "Maximum_Temperature", "Minimum_Temperature", "Total_Land"

FEATURE_SETS = {
    "with_temp": [YEAR_COL, RAIN, TMAX, TMIN, LAND],
    "without_temp": [YEAR_COL, RAIN, LAND],
}
FEATURE_LABELS = {"with_temp": "LR+RF with Temperature", "without_temp": "LR+RF without Temperature"}

RANDOM_STATE = 42
N_ESTIMATORS = 300
FIRST_YEAR, TRAIN_END, LAST_YEAR = 1970, 2022, 2100


# ------------------------------------------------------------------
# MODEL (same architecture as the research code)
# ------------------------------------------------------------------
def make_model(model_type):
    if model_type == "LR":
        return LinearRegression()
    if model_type == "RF":
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                     n_jobs=-1, max_features="sqrt")
    raise ValueError(model_type)


def make_scaled_pipeline(model_type):
    return Pipeline([("scaler", MinMaxScaler()), ("model", make_model(model_type))])


class ResidualHybrid:
    def __init__(self, residual_model_type="RF", random_state=42):
        self.residual_model_type = residual_model_type
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True).copy()
        y = pd.Series(np.asarray(y, dtype=float).reshape(-1)).reset_index(drop=True)
        n = len(X)
        cv = TimeSeriesSplit(n_splits=min(5, n - 1))
        oof = np.full(n, np.nan)
        for tr, va in cv.split(X):
            lr = make_scaled_pipeline("LR")
            lr.fit(X.iloc[tr], y.iloc[tr])
            oof[va] = lr.predict(X.iloc[va])
        mask = np.isfinite(oof)
        resid = y.to_numpy()[mask] - oof[mask]
        self.residual_model_ = make_model(self.residual_model_type)
        self.residual_model_.fit(X.loc[mask].reset_index(drop=True), resid)
        self.final_lr_ = make_scaled_pipeline("LR")
        self.final_lr_.fit(X, y)
        return self

    def predict(self, X):
        X = pd.DataFrame(X).copy()
        return np.asarray(self.final_lr_.predict(X)) + np.asarray(self.residual_model_.predict(X))


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def load_historical():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Historical dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    need = [YEAR_COL, TARGET_COL, RAIN, TMAX, TMIN, LAND]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in historical CSV: {miss}")
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df[YEAR_COL] != 2023]
    df = df[(df[YEAR_COL] >= FIRST_YEAR) & (df[YEAR_COL] <= TRAIN_END)]
    if df[need].isna().any().any():
        raise ValueError("Missing/invalid values in historical dataset.")
    return df.sort_values(YEAR_COL).reset_index(drop=True)


def load_world_bank():
    """1st weather col = rainfall, 2nd = max temp, 3rd = min temp.
    Uses a 'Year' column if present, otherwise rows = 2023, 2024, ... in order."""
    if not WB_FILE.exists():
        raise FileNotFoundError(f"World Bank dataset not found: {WB_FILE}")
    raw = pd.read_csv(WB_FILE)
    year_name = next((c for c in raw.columns if str(c).strip().lower() == "year"), None)
    if year_name is not None:
        years = pd.to_numeric(raw[year_name], errors="coerce")
        weather = raw.drop(columns=[year_name])
    else:
        weather = raw
        years = pd.Series(range(2023, 2023 + len(raw)))
    weather = weather.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if weather.shape[1] < 3:
        raise ValueError("World Bank CSV needs at least 3 numeric columns (rain, max temp, min temp).")
    wb = pd.DataFrame({
        YEAR_COL: years.values,
        RAIN: weather.iloc[:, 0].values,
        TMAX: weather.iloc[:, 1].values,
        TMIN: weather.iloc[:, 2].values,
    }).dropna()
    wb[YEAR_COL] = wb[YEAR_COL].astype(int)
    wb = wb[(wb[YEAR_COL] >= 2023) & (wb[YEAR_COL] <= LAST_YEAR)]
    return wb.drop_duplicates(YEAR_COL).set_index(YEAR_COL)


hist = load_historical()
wb = load_world_bank()
hist_idx = hist.set_index(YEAR_COL)

land_model = LinearRegression().fit(hist[[YEAR_COL]], hist[LAND])
models = {k: ResidualHybrid("RF", RANDOM_STATE).fit(hist[c], hist[TARGET_COL]) for k, c in FEATURE_SETS.items()}


def inputs_for_year(year):
    if year < FIRST_YEAR or year > LAST_YEAR:
        raise ValueError(f"Year must be between {FIRST_YEAR} and {LAST_YEAR}.")
    if year <= TRAIN_END:
        if year not in hist_idx.index:
            raise ValueError(f"No historical data for {year}.")
        r = hist_idx.loc[year]
        return {YEAR_COL: year, RAIN: float(r[RAIN]), TMAX: float(r[TMAX]), TMIN: float(r[TMIN]),
                LAND: float(r[LAND]), "source": "Historical dataset (1970-2022)",
                "land_source": "Observed cultivated land", "period": "historical"}
    if year not in wb.index:
        raise ValueError(f"The World Bank dataset has no data for {year}.")
    r = wb.loc[year]
    land = float(land_model.predict(pd.DataFrame({YEAR_COL: [year]}))[0])
    return {YEAR_COL: year, RAIN: float(r[RAIN]), TMAX: float(r[TMAX]), TMIN: float(r[TMIN]),
            LAND: land, "source": "World Bank WBCCKP CMIP6 (local file)",
            "land_source": "Predicted by cultivated-land trend model", "period": "future"}


def predict_value(year, config):
    inp = inputs_for_year(year)
    cols = FEATURE_SETS[config]
    X = pd.DataFrame([{c: inp[c] for c in cols}])[cols]
    return inp, float(models[config].predict(X)[0])


curves = {}
climate_curve = []
for key in FEATURE_SETS:
    pts = []
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        try:
            inp, p = predict_value(y, key)
            pts.append({"year": y, "pred": round(p, 3)})
            if key == "with_temp":
                climate_curve.append({"year": y, "rain": inp[RAIN], "tmax": inp[TMAX], "tmin": inp[TMIN]})
        except ValueError:
            pass
    curves[key] = pts

# ------------------------------------------------------------------
# FLASK API
# ------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/hero.jpg")
def hero_image():
    # 1) your own photo saved next to app.py (hero.jpg / hero.jpeg / hero.png / hero.webp)
    for name in ("hero.jpg", "hero.jpeg", "hero.png", "hero.webp"):
        if (BASE / name).exists():
            return send_from_directory(BASE, name)
    # 2) or a direct image link set in HERO_IMAGE_URL at the top of this file
    if HERO_IMAGE_URL:
        return redirect(HERO_IMAGE_URL)
    abort(404)


@app.route("/about.jpg")
def about_image():
    # about.* if you added one, otherwise reuse your hero photo so the section is never empty
    for name in ("about.jpg", "about.jpeg", "about.png", "about.webp",
                 "hero.jpg", "hero.jpeg", "hero.png", "hero.webp"):
        if (BASE / name).exists():
            return send_from_directory(BASE, name)
    if HERO_IMAGE_URL:
        return redirect(HERO_IMAGE_URL)
    abort(404)


@app.route("/api/meta")
def meta():
    return jsonify({"first_year": FIRST_YEAR, "last_year": LAST_YEAR, "train_end": TRAIN_END})


@app.route("/api/inputs")
def api_inputs():
    try:
        return jsonify(inputs_for_year(int(request.args.get("year", ""))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict")
def api_predict():
    config = request.args.get("config", "with_temp")
    if config not in FEATURE_SETS:
        return jsonify({"error": "Unknown model configuration."}), 400
    try:
        year = int(request.args.get("year", ""))
        inp, pred = predict_value(year, config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    actual = float(hist_idx.loc[year][TARGET_COL]) if year in hist_idx.index else None
    future = inp["period"] == "future"
    return jsonify({
        "year": year, "config_label": FEATURE_LABELS[config], "prediction_lmt": round(pred, 3),
        "actual_lmt": actual,
        "climate_model": CLIMATE_MODEL if future else "Observed (historical record)",
        "climate_scenario": CLIMATE_SCENARIO if future else "N/A (observed)",
        "inputs": {"rainfall": inp[RAIN], "tmax": inp[TMAX], "tmin": inp[TMIN], "land": inp[LAND],
                   "weather_source": inp["source"], "land_source": inp["land_source"]},
    })


@app.route("/api/series")
def api_series():
    config = request.args.get("config", "with_temp")
    if config not in curves:
        return jsonify({"error": "Unknown model configuration."}), 400
    actual = [{"year": int(y), "actual": float(v)} for y, v in zip(hist[YEAR_COL], hist[TARGET_COL])]
    return jsonify({"actual": actual, "predicted": curves[config], "train_end": TRAIN_END})


@app.route("/api/climate")
def api_climate():
    return jsonify({"series": climate_curve, "train_end": TRAIN_END})


@app.route("/api/metrics")
def api_metrics():
    if not RESULTS_FILE.exists():
        return jsonify({"available": False, "rows": []})
    r = pd.read_csv(RESULTS_FILE)
    r = r[r["Split"].str.contains("Chronological", case=False)]
    names = {"Linear Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF",
             "Gradient Boosting": "GB", "XGBoost": "XGB", "LR + RF": "LR+RF (Proposed)"}
    r = r[r["Model"].isin(names)]
    rows = [{"feature_set": "with_temp" if str(x["Feature_Set"]).lower().startswith("with_") else "without_temp",
             "model": names[x["Model"]], "r2": float(x["R2"]),
             "mae": float(x["MAE (LMT)"]), "rmse": float(x["RMSE (LMT)"])} for _, x in r.iterrows()]
    return jsonify({"available": True, "rows": rows})


# ------------------------------------------------------------------
# WEB PAGE (HTML + CSS + JS in one string)
# ------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aman Rice Production Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--forest:#1f4d2e;--field:#5b8c3a;--sage:#a8bfa0;--sage2:#e6eee1;--gold:#d9a520;--gold2:#f3e2b0;--cream:#faf6ea;--ink:#2b2f2c;--mut:#667068;--maxw:1200px;--pad:20px;--navh:64px}
@media(min-width:768px){:root{--pad:32px}}
@media(min-width:1024px){:root{--pad:48px;--navh:68px}}
*{box-sizing:border-box}
.grid-2>*,.grid-3>*,.grid-4>*{min-width:0}
img,canvas{max-width:100%}
html{scroll-behavior:smooth}
body{margin:0;font-family:Inter,Arial,sans-serif;color:var(--ink);background:var(--cream);line-height:1.65;overflow-x:hidden}
h1,h2,h3,h4{font-family:Fraunces,Georgia,serif;margin:0 0 .5rem;color:var(--forest)}
p{margin:0 0 1rem}
a{color:inherit}

/* NAV */
.nav{position:fixed;top:0;left:0;right:0;z-index:50;height:var(--navh);background:rgba(250,246,234,.94);backdrop-filter:blur(12px);border-bottom:1px solid #e7dfc6;box-shadow:0 1px 0 rgba(255,255,255,.6) inset}
.nav-in{max-width:var(--maxw);height:100%;margin:auto;padding:0 var(--pad);display:flex;justify-content:space-between;align-items:center;gap:1rem}
.brand{display:flex;align-items:center;gap:.55rem;font-family:Fraunces,serif;font-weight:700;color:var(--forest);font-size:1.15rem;text-decoration:none;white-space:nowrap;letter-spacing:-.01em}
.brand i{font-style:normal;display:grid;place-items:center;width:32px;height:32px;border-radius:9px;background:var(--forest);color:var(--gold2);font-size:1rem}
.menu{display:flex;align-items:center;gap:.25rem;height:100%}
.menu a{position:relative;display:flex;align-items:center;gap:.45rem;height:100%;padding:0 .85rem;text-decoration:none;color:#4a524c;font-size:.875rem;font-weight:500;letter-spacing:.005em;white-space:nowrap;transition:color .25s}
.menu a svg{width:17px;height:17px;flex:none;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;opacity:.75;transition:.25s}
.menu a::after{content:"";position:absolute;left:.85rem;right:.85rem;bottom:-1px;height:3px;border-radius:3px 3px 0 0;background:var(--gold);transform:scaleX(0);transform-origin:center;transition:transform .3s ease}
.menu a:hover{color:var(--forest)}
.menu a:hover svg{opacity:1;transform:translateY(-1px)}
.menu a:hover::after{transform:scaleX(.45)}
.menu a.active{color:var(--forest);font-weight:600}
.menu a.active svg{opacity:1;color:var(--field)}
.menu a.active::after{transform:scaleX(1)}
.menu a:focus-visible{outline:2px solid var(--gold);outline-offset:-4px;border-radius:8px}
.menu .sm{display:none}
@media(min-width:1024px) and (max-width:1279px){.menu .lg{display:none}.menu .sm{display:inline}.menu a{padding:0 .7rem}.menu a::after{left:.7rem;right:.7rem}}
.burger{display:none;width:44px;height:44px;border:1px solid #e7dfc6;background:#fff;border-radius:12px;cursor:pointer;color:var(--forest);align-items:center;justify-content:center;transition:.2s}
.burger:hover{background:var(--sage2)}
.burger svg{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round}

/* PAGES */
.page{display:none;padding-top:var(--navh);min-height:80vh;animation:fade .5s ease}
.page.active{display:block}
@keyframes fade{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.wrap{max-width:var(--maxw);margin:auto;padding:2rem var(--pad) 3rem}
@media(min-width:768px){.wrap{padding-top:2.75rem;padding-bottom:3.5rem}}
@media(min-width:1024px){.wrap{padding-top:3.5rem;padding-bottom:4.5rem}}
.page-title{font-size:clamp(1.8rem,4vw,2.6rem)}
.lead{color:var(--mut);max-width:720px;margin-bottom:2rem}

/* HERO */
.hero{position:relative;height:clamp(520px,calc(100vh - var(--navh)),760px);display:flex;align-items:center;color:#fff;background-color:#8a8a2b;
 /* light overlay: dark on the text side, almost clear at the bottom so the field stays visible */
 background-image:linear-gradient(90deg,rgba(14,32,18,.74) 0%,rgba(14,32,18,.42) 48%,rgba(14,32,18,.08) 100%),
 linear-gradient(180deg,rgba(0,0,0,.18) 0%,rgba(0,0,0,0) 30%,rgba(60,40,0,.12) 100%),
 url('/hero.jpg'),
 /* fallback while no photo is added: flat golden plain under a warm sky */
 linear-gradient(180deg,#bcd2dc 0%,#efe3b8 34%,#e4c768 47%,#cfa73a 58%,#a98f2c 78%,#6f7a25 100%);
 background-size:cover;background-position:center,center,center 62%,center}
.hero-in{max-width:var(--maxw);margin:auto;padding:0 var(--pad);width:100%}
@media(max-width:767px){.hero{height:clamp(500px,82vh,620px);background-image:linear-gradient(180deg,rgba(14,32,18,.7) 0%,rgba(14,32,18,.5) 55%,rgba(14,32,18,.18) 100%),url('/hero.jpg'),linear-gradient(180deg,#bcd2dc 0%,#efe3b8 34%,#e4c768 47%,#cfa73a 58%,#a98f2c 78%,#6f7a25 100%);background-position:center,center 60%,center}}
@media(min-width:768px) and (max-width:1023px){.hero{height:clamp(520px,68vh,640px)}}
.tag{display:inline-block;background:rgba(217,165,32,.2);border:1px solid rgba(243,226,176,.6);color:var(--gold2);padding:.3rem .9rem;border-radius:999px;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;margin-bottom:1.2rem}
.hero h1{color:#fff;font-size:clamp(2.4rem,6.5vw,4.6rem);line-height:1.05;max-width:760px}
.hero .sub{font-size:clamp(1.05rem,2.2vw,1.35rem);color:var(--gold2);max-width:640px;margin:1rem 0 .8rem}
.hero .desc{max-width:600px;color:#e8eee4}
.cta{display:flex;gap:.8rem;flex-wrap:wrap;margin-top:1.6rem}
.btn{background:var(--gold);color:#2b2200;border:0;border-radius:12px;padding:.85rem 1.7rem;font-size:1rem;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block;transition:.2s;font-family:inherit}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.2)}
.btn.ghost{background:transparent;color:#fff;border:1.5px solid rgba(255,255,255,.7)}
.btn.green{background:var(--forest);color:#fff}
.btn.full{width:100%;padding:1rem;font-size:1.05rem}

/* FEATURE CARDS */
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
.card{background:#fff;border-radius:18px;padding:1.5rem;box-shadow:0 6px 24px rgba(40,50,30,.07);border:1px solid #eee7d0}
.card h3{font-size:1.15rem}
.ico{font-size:1.8rem;margin-bottom:.4rem}
.feat{transition:.25s}.feat:hover{transform:translateY(-4px)}
.feat a{color:var(--field);font-weight:600;text-decoration:none;font-size:.9rem}

/* DASHBOARD */
label.l{display:block;font-weight:600;margin:1.1rem 0 .4rem;font-size:.85rem;color:var(--forest);text-transform:uppercase;letter-spacing:.05em}
.year-row{display:flex;gap:1rem;align-items:center}
input[type=range]{flex:1;accent-color:var(--field);height:30px}
input[type=number]{width:100px;padding:.6rem;border:1.5px solid var(--sage);border-radius:10px;font-size:1.05rem;font-family:inherit;background:var(--cream)}
.chips{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.6rem}
.chip{border:1px solid var(--sage);background:#fff;border-radius:999px;padding:.3rem .8rem;font-size:.82rem;cursor:pointer;font-family:inherit}
.chip:hover{background:var(--sage2)}
.badge{display:inline-block;padding:.15rem .7rem;border-radius:999px;font-size:.75rem;font-weight:600;background:var(--gold);color:#2b2200}
.badge.hist{background:var(--field);color:#fff}
.seg{display:flex;gap:.4rem}
.seg button{flex:1;padding:.7rem .6rem;border:1.5px solid var(--sage);background:#fff;border-radius:12px;cursor:pointer;font-weight:500;font-family:inherit;font-size:.9rem}
.seg button.active{background:var(--forest);color:#fff;border-color:var(--forest)}
.seg.c{justify-content:center;max-width:420px;margin-bottom:1.5rem}
.inputs{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin:1.2rem 0 .4rem}
.stat{background:var(--cream);border:1px solid #eee7d0;border-radius:14px;padding:.8rem 1rem}
.stat span{display:block;font-size:.72rem;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.stat b{font-size:1.4rem;color:var(--forest);font-family:Fraunces,serif}
.stat small{margin-left:.25rem;color:var(--mut)}
.stat.off{opacity:.35}
.src{font-size:.8rem;color:var(--mut);margin:.4rem 0 1.2rem}
.error{margin-top:1rem;background:#fdecea;color:#a12622;padding:.8rem 1rem;border-radius:10px;font-size:.9rem}
.result{background:linear-gradient(155deg,var(--forest),#2f6b3f);color:#fff;border:0;display:flex;flex-direction:column;justify-content:center}
.result h4,.result .eyebrow{color:var(--gold2)}
.eyebrow{font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem}
.big{display:flex;align-items:baseline;gap:.6rem;font-family:Fraunces,serif}
.big span{font-size:clamp(3.2rem,8vw,5rem);font-weight:700;line-height:1;color:#fff}
.big em{font-style:normal;color:var(--gold2);font-size:1.3rem}
.actual{color:var(--gold2);font-size:.92rem;margin:.3rem 0 0}
.rgrid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:1.2rem}
.rgrid div{background:rgba(255,255,255,.1);border-radius:12px;padding:.6rem .8rem}
.rgrid span{display:block;font-size:.7rem;color:var(--gold2);text-transform:uppercase;letter-spacing:.05em}
.rgrid b{font-size:1.1rem;color:#fff}
.meta{display:grid;grid-template-columns:auto 1fr;gap:.3rem 1rem;margin-top:1.1rem;font-size:.88rem}
.meta dt{color:var(--gold2)}.meta dd{margin:0;font-weight:600;color:#fff}
.chart-wrap{height:360px;position:relative}.chart-wrap.sm{height:280px}
.note{font-size:.82rem;color:var(--mut);margin:.8rem 0 0}
.stack{display:grid;gap:1.2rem;margin-top:1.2rem}
#resultCard{scroll-margin-top:calc(var(--navh) + 16px)}
#resultCard.pulse{animation:pulse 1.2s ease}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(217,165,32,.75)}70%{box-shadow:0 0 0 18px rgba(217,165,32,0)}100%{box-shadow:0 0 0 0 rgba(217,165,32,0)}}

/* MODEL FLOW */
.flow{display:flex;flex-wrap:wrap;align-items:stretch;justify-content:center;gap:.5rem;margin:1.5rem 0 2rem}
.node{background:#fff;border:1px solid #eee7d0;border-radius:16px;padding:1rem .8rem;width:150px;text-align:center;box-shadow:0 6px 18px rgba(40,50,30,.06)}
.node .ico{font-size:1.7rem}
.node h4{font-size:.92rem;margin-bottom:.2rem}
.node p{font-size:.76rem;color:var(--mut);margin:0;line-height:1.4}
.node.key{background:var(--forest);border-color:var(--forest)}.node.key h4{color:#fff}.node.key p{color:var(--gold2)}
.arr{align-self:center;color:var(--gold);font-size:1.4rem;font-weight:700}
.formula{background:var(--forest);color:#fff;border-radius:16px;padding:1.2rem 1.5rem;text-align:center;font-family:Fraunces,serif;font-size:clamp(1rem,2.5vw,1.4rem)}
.formula b{color:var(--gold2)}

/* AMAN */
.photo{border-radius:18px;min-height:300px;background:url('/about.jpg') center/cover,linear-gradient(160deg,#e9d9a0,#d9a520 55%,#6f8a2b)}
.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin:1.2rem 0}
.tl{background:#fff;border-radius:14px;padding:1rem;border-top:4px solid var(--gold);box-shadow:0 4px 14px rgba(40,50,30,.06)}
.tl b{display:block;color:var(--forest);font-family:Fraunces,serif}
.tl span{font-size:.85rem;color:var(--mut)}

/* TABLE */
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{padding:.65rem .5rem;text-align:left;border-bottom:1px solid #eee7d0}
th{color:var(--forest);font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
tr.best td{background:var(--gold2);font-weight:600}
ul.clean{padding-left:1.1rem;margin:0 0 1rem}ul.clean li{margin:.3rem 0}

footer{background:var(--forest);color:#d9e6d4;text-align:center;padding:2.2rem 1rem}
footer h4{color:#fff;margin-bottom:.3rem}footer p{margin:.15rem 0}.small{font-size:.8rem;opacity:.75}

/* ===== Standard responsive system =====
   Mobile 320-767 | Tablet 768-1023 | Laptop/Desktop 1024+ ; content max width 1200px */
.btn{min-height:48px;display:inline-flex;align-items:center;justify-content:center}
input[type=number]{min-height:46px}
.seg button,.chip{min-height:44px}
.chip{min-height:38px}
.hero h1{font-size:clamp(2.1rem,5.5vw,4.2rem)}
.page-title{font-size:clamp(1.65rem,3.6vw,2.4rem)}
.card{padding:1.25rem}
@media(min-width:768px){.card{padding:1.5rem}}
@media(min-width:1024px){.card{padding:1.75rem}}
.gap-l{gap:1.2rem}
table{min-width:420px}

/* Tablet and below: hamburger drawer, 2-column grids */
@media(max-width:1023px){
 .burger{display:flex}
 .menu{display:none;position:absolute;top:100%;left:0;right:0;height:auto;flex-direction:column;align-items:stretch;gap:.15rem;background:var(--cream);padding:.6rem var(--pad) 1rem;border-bottom:1px solid #e7dfc6;box-shadow:0 14px 24px rgba(30,40,20,.12)}
 .menu.open{display:flex;animation:fade .25s ease}
 .menu a{height:auto;min-height:48px;padding:0 1rem;border-radius:12px;font-size:1rem}
 .menu a::after{left:0;right:auto;top:10px;bottom:10px;width:4px;height:auto;border-radius:4px;transform:scaleY(0)}
 .menu a:hover::after{transform:scaleY(.5)}
 .menu a.active{background:var(--sage2)}
 .menu a.active::after{transform:scaleY(1)}
 .grid-2{grid-template-columns:1fr}
 .grid-3,.grid-4,.timeline{grid-template-columns:1fr 1fr}
 .grid-3>:last-child:nth-child(odd){grid-column:1/-1}
 .chart-wrap{height:340px}.chart-wrap.sm{height:300px}
}
/* Tablet only */
@media(min-width:768px) and (max-width:1023px){
 .grid-2.keep2{grid-template-columns:1fr 1fr}
 .node{width:calc(33.33% - .5rem)}
 .arr{display:none}
}
/* Mobile */
@media(max-width:767px){
 body{font-size:15.5px}
 .brand{font-size:1.05rem}
 .grid-3,.grid-4,.timeline{grid-template-columns:1fr}
 .grid-3>:last-child:nth-child(odd){grid-column:auto}
 .node{width:100%}.arr{transform:rotate(90deg);width:100%;text-align:center}
 .chart-wrap{height:280px}.chart-wrap.sm{height:260px}
 .btn{width:100%}
 .cta{flex-direction:column}
 .year-row{flex-wrap:wrap}
 .year-row input[type=number]{width:100%}
 .seg{flex-direction:column}
 .inputs,.rgrid{grid-template-columns:1fr 1fr;gap:.5rem}
 .stat b{font-size:1.2rem}
 .meta{grid-template-columns:1fr;gap:0}.meta dd{margin-bottom:.5rem}
 .big span{font-size:3.2rem}
}
@media(max-width:360px){.inputs,.rgrid{grid-template-columns:1fr}.brand span{font-size:.95rem}}
.hero-photo{display:none}
@media(max-width:767px){
 .hero{display:block;height:auto;background:var(--forest)}
 .hero-photo{display:block;width:100%;height:auto;max-height:none}
 .hero-in{padding:1.5rem var(--pad) 2.2rem}
 .hero h1{font-size:clamp(1.9rem,8.5vw,2.4rem)}
 /* no photo added yet: keep the golden placeholder */
 .hero.nophoto{height:clamp(500px,82vh,620px);display:flex;align-items:center;
  background:linear-gradient(180deg,rgba(14,32,18,.7),rgba(14,32,18,.35)),linear-gradient(180deg,#bcd2dc 0%,#efe3b8 34%,#e4c768 47%,#cfa73a 58%,#a98f2c 78%,#6f7a25 100%)}
 .hero.nophoto .hero-photo{display:none}
}
.quote{font-family:Fraunces,Georgia,serif;font-style:italic;color:var(--gold2);font-size:clamp(1.05rem,2.4vw,1.35rem);border-left:3px solid var(--gold);padding-left:.9rem;margin:1.3rem 0 0;max-width:620px;line-height:1.4}
footer .fq{font-family:Fraunces,Georgia,serif;font-style:italic;color:var(--gold2);margin:.3rem 0 .9rem}
.season{display:flex;gap:1rem;align-items:flex-start}
.season .ico{margin:0}
.months{display:flex;gap:.4rem;flex-wrap:wrap;margin:.2rem 0 .8rem}
.months span{background:var(--gold2);color:#4a3a00;border-radius:999px;padding:.2rem .8rem;font-size:.82rem;font-weight:600}
</style>
</head>
<body>

<nav class="nav" aria-label="Main navigation">
  <div class="nav-in">
    <a class="brand" href="#home"><i>🌾</i><span>Aman Predictor</span></a>
    <button class="burger" id="burger" aria-label="Open menu" aria-expanded="false">
      <svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
    </button>
    <div class="menu" id="menu">
      <a href="#home" data-p="home"><svg viewBox="0 0 24 24"><path d="M3 11.5 12 4l9 7.5M5.5 10v9.5h13V10M10 19.5v-5h4v5"/></svg>Home</a>
      <a href="#predict" data-p="predict"><svg viewBox="0 0 24 24"><path d="M4 20V4M4 20h16M8 16v-4M12 16V8M16 16v-6"/></svg>Prediction</a>
      <a href="#model" data-p="model"><svg viewBox="0 0 24 24"><circle cx="6" cy="6" r="2.2"/><circle cx="18" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/><path d="M7.7 7.5 10.8 16M16.3 7.5 13.2 16M8.2 6h7.6"/></svg><span class="lg">Model Architecture</span><span class="sm">Model</span></a>
      <a href="#aman" data-p="aman"><svg viewBox="0 0 24 24"><path d="M12 21V9M12 9c0-3 1.6-5 4-6 .3 3-1 5.2-4 6zM12 13c0-2.6-1.5-4.4-4-5 0 2.8 1.4 4.6 4 5zM12 17c0-2.2-1.3-3.7-3.5-4.2.1 2.4 1.3 3.9 3.5 4.2zM12 17c2.2-.4 3.4-1.8 3.5-4.2-2.2.5-3.5 2-3.5 4.2z"/></svg><span class="lg">About Aman Rice</span><span class="sm">About Aman</span></a>
      <a href="#performance" data-p="performance"><svg viewBox="0 0 24 24"><path d="M3 17l5-6 4 4 8-9M15 6h5v5"/></svg><span class="lg">Performance &amp; Comparison</span><span class="sm">Performance</span></a>
      <a href="#data" data-p="data"><svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>Data Sources</a>
    </div>
  </div>
</nav>

<!-- ================= HOME ================= -->
<main class="page" id="page-home">
  <section class="hero">
    <img class="hero-photo" src="/hero.jpg" alt="Ripe Aman rice field in Bangladesh" onerror="this.parentNode.classList.add('nophoto')">
    <div class="hero-in">
      <span class="tag">Agricultural Intelligence · Bangladesh</span>
      <h1>Aman Rice Production Predictor</h1>
      <p class="sub">Data-Driven Forecasting for Bangladesh's Aman Rice Production</p>
      <p class="desc">This platform estimates Aman rice production using historical agricultural data, cultivated land, and climate information, with forecasts from 1970 to 2100.</p>
      <p class="quote">“Where Algorithms Meet the Art of Agriculture”</p>
      <div class="cta">
        <a class="btn" href="#predict">Explore Prediction</a>
        <a class="btn ghost" href="#model">Explore Model</a>
      </div>
    </div>
  </section>
  <section class="wrap">
    <h2 class="page-title">What you can do here</h2>
    <p class="lead">Pick a year, choose a model configuration, and see the predicted production together with the rainfall and temperature behind it.</p>
    <div class="grid-3">
      <div class="card feat"><div class="ico">📊</div><h3>Predict production</h3><p>Forecast Aman production in Lakh Metric Tonnes (LMT) for any year from 1970 to 2100.</p><a href="#predict">Open prediction →</a></div>
      <div class="card feat"><div class="ico">🧠</div><h3>Understand the model</h3><p>See how Linear Regression and Random Forest work together as a residual hybrid.</p><a href="#model">View architecture →</a></div>
      <div class="card feat"><div class="ico">📈</div><h3>Compare performance</h3><p>Compare LR+RF with LR, DT, RF, GB and XGB under chronological validation.</p><a href="#performance">See results →</a></div>
    </div>
  </section>
</main>

<!-- ================= PREDICTION ================= -->
<main class="page" id="page-predict">
  <section class="wrap">
    <h2 class="page-title">Production Prediction</h2>
    <p class="lead">Years up to 2022 use observed weather and land. Years 2023–2100 use the World Bank climate projection and a cultivated-land trend model. Weather inputs cover the monsoon months only (June–September).</p>
    <div class="grid-2">
      <div class="card">
        <label class="l">Year</label>
        <div class="year-row">
          <input type="range" id="yearRange" min="1970" max="2100" value="2050">
          <input type="number" id="yearNum" min="1970" max="2100" value="2050">
        </div>
        <div class="chips">
          <button class="chip" data-y="1990">1990</button><button class="chip" data-y="2010">2010</button>
          <button class="chip" data-y="2022">2022</button><button class="chip" data-y="2030">2030</button>
          <button class="chip" data-y="2050">2050</button><button class="chip" data-y="2075">2075</button>
          <button class="chip" data-y="2100">2100</button>
        </div>
        <p style="margin:.7rem 0 0;font-size:.85rem"><span id="periodBadge" class="badge">Future</span> <span id="periodNote" style="color:var(--mut)"></span></p>

        <label class="l">Model configuration</label>
        <div class="seg" id="configSeg">
          <button class="active" data-v="with_temp">With Temperature</button>
          <button data-v="without_temp">Without Temperature</button>
        </div>

        <label class="l">Inputs for this year</label>
        <div class="inputs">
          <div class="stat"><span>🌧️ Rainfall (JJAS)</span><b id="inRain">–</b><small>mm</small></div>
          <div class="stat" data-temp><span>🌡️ Max temperature</span><b id="inTmax">–</b><small>°C</small></div>
          <div class="stat" data-temp><span>❄️ Min temperature</span><b id="inTmin">–</b><small>°C</small></div>
          <div class="stat"><span>🌾 Cultivated land</span><b id="inLand">–</b></div>
        </div>
        <p class="src" id="srcNote"></p>
        <button class="btn green full" id="predictBtn">Predict Production</button>
        <div id="errorBox" class="error" hidden></div>
      </div>

      <div class="card result" id="resultCard">
        <p class="eyebrow">Predicted Aman production</p>
        <div class="big"><span id="resultVal">—</span><em>LMT</em></div>
        <p id="actualLine" class="actual" hidden></p>
        <div class="rgrid">
          <div><span>🌧️ Rainfall</span><b id="rRain">—</b></div>
          <div><span>🌾 Cultivated land</span><b id="rLand">—</b></div>
          <div><span>🌡️ Max temp</span><b id="rTmax">—</b></div>
          <div><span>❄️ Min temp</span><b id="rTmin">—</b></div>
        </div>
        <dl class="meta">
          <dt>Year</dt><dd id="rYear">—</dd>
          <dt>Configuration</dt><dd id="rConfig">—</dd>
          <dt>Climate model</dt><dd id="rModel">—</dd>
          <dt>Climate scenario</dt><dd id="rScen">—</dd>
        </dl>
      </div>
    </div>

    <div class="stack">
      <div class="card">
        <h3>Aman production, 1970–2100</h3>
        <div class="chart-wrap"><canvas id="prodChart"></canvas></div>
        <p class="note">Solid green = actual production. Light dashed = model fit (1970–2022). Gold dashed = forecast (2023–2100). Red dot = your predicted year.</p>
      </div>
      <div class="card">
        <h3>Rainfall and temperature, 1970–2100</h3>
        <div class="chart-wrap"><canvas id="climChart"></canvas></div>
        <p class="note">Rainfall (JJAS total, mm) on the left axis; maximum and minimum temperature (°C) on the right. 1970–2022 is observed, 2023–2100 is from the World Bank file. The dot marks the selected year.</p>
      </div>
    </div>
  </section>
</main>

<!-- ================= MODEL ================= -->
<main class="page" id="page-model">
  <section class="wrap">
    <h2 class="page-title">Model Architecture: LR + RF Residual Hybrid</h2>
    <p class="lead">A simple trend model does the heavy lifting; a Random Forest then learns what the trend model gets wrong.</p>

    <div class="flow">
      <div class="node"><div class="ico">🧮</div><h4>Input Features</h4><p>Year, rainfall, (temperatures), land</p></div><div class="arr">➜</div>
      <div class="node"><div class="ico">📏</div><h4>MinMax Scaling</h4><p>All inputs scaled to 0–1</p></div><div class="arr">➜</div>
      <div class="node"><div class="ico">📈</div><h4>Linear Regression</h4><p>Baseline trend prediction</p></div><div class="arr">➜</div>
      <div class="node"><div class="ico">➖</div><h4>Residual Calculation</h4><p>Actual − out-of-fold LR prediction</p></div><div class="arr">➜</div>
      <div class="node"><div class="ico">🌳</div><h4>Random Forest</h4><p>Learns the residual patterns</p></div><div class="arr">➜</div>
      <div class="node key"><div class="ico">🌾</div><h4>Final Prediction</h4><p>LR + predicted residual</p></div>
    </div>

    <div class="formula">Final prediction = <b>Linear Regression</b> prediction + <b>Random Forest</b> predicted residual</div>

    <div class="grid-2" style="margin-top:1.5rem">
      <div class="card"><h3>📈 Linear Regression: the baseline</h3><p>Finds the best straight-line relationship between the inputs and production. It captures the long-term upward trend and extends it smoothly into the future.</p></div>
      <div class="card"><h3>➖ Residuals: what the baseline misses</h3><p>A residual is the actual value minus the baseline prediction. To keep residuals honest, they are computed out-of-fold using chronological (time-ordered) splits, so each year is predicted only from earlier years.</p></div>
      <div class="card"><h3>🌳 Random Forest: learning the leftovers</h3><p>An ensemble of decision trees that captures non-linear patterns, such as how unusual rainfall or temperature in a given year pushes production above or below the trend.</p></div>
      <div class="card"><h3>🌾 Why it suits Aman rice</h3><p>Aman production has a strong long-term trend plus year-to-year swings from monsoon weather. The hybrid keeps the stable trend and adds a learned correction for the swings.</p></div>
    </div>

    <div class="card" style="margin-top:1.2rem">
      <h3>Two configurations</h3>
      <ul class="clean">
        <li><b>With Temperature:</b> Year, rainfall, maximum temperature, minimum temperature, cultivated land.</li>
        <li><b>Without Temperature:</b> Year, rainfall, cultivated land.</li>
      </ul>
      <p class="note">Training period: 1970–2022. Settings: 300 trees, random_state 42, residuals from chronological out-of-fold Linear Regression predictions.</p>
    </div>
  </section>
</main>

<!-- ================= ABOUT AMAN ================= -->
<main class="page" id="page-aman">
  <section class="wrap">
    <h2 class="page-title">About Aman Rice</h2>
    <p class="lead">One of Bangladesh's major rice seasons, grown with the monsoon rains.</p>
    <div class="grid-2">
      <div>
        <p>Aman is one of Bangladesh's three main rice-growing seasons, alongside Aus and Boro. It is a rainy-season crop that depends largely on monsoon rainfall rather than irrigation, and it covers a very large share of the country's rice land.</p>
        <p>Most Aman is transplanted: seedlings are raised in beds and moved to the field during the monsoon. A smaller share is broadcast-sown directly. Because it grows through the rains, its yield is sensitive to how much, and when, the rain falls.</p>
        <p><b>Food security:</b> rice is the staple food of Bangladesh, so Aman harvests directly affect national food supply, prices and import planning.</p>
      </div>
      <div class="photo" role="img" aria-label="Aman rice field"></div>
    </div>
    <div class="timeline">
      <div class="tl"><b>Seedbed</b><span>Seedlings are raised before the monsoon peak.</span></div>
      <div class="tl"><b>Transplanting</b><span>Generally July–August, during the monsoon.</span></div>
      <div class="tl"><b>Growth</b><span>Plants grow and flower through the rainy season.</span></div>
      <div class="tl"><b>Harvest</b><span>Generally November–December.</span></div>
    </div>
    <div class="grid-3">
      <div class="card"><div class="ico">🌧️</div><h3>Rainfall</h3><p>Supplies most of the water the crop needs. Too little causes drought stress; too much can cause flooding and crop damage.</p></div>
      <div class="card"><div class="ico">🌡️</div><h3>Temperature</h3><p>Affects growth, flowering and grain filling. Very high temperatures can reduce yield.</p></div>
      <div class="card"><div class="ico">🌾</div><h3>Cultivated land</h3><p>The area under Aman is a basic driver of total production, along with yield per unit of land.</p></div>
    </div>
    <div class="card" style="margin-top:1.2rem">
      <div class="season"><div class="ico">🌧️</div><div>
        <h3>Why this model uses only June–September weather</h3>
        <div class="months"><span>June</span><span>July</span><span>August</span><span>September</span></div>
        <p>Aman is a monsoon-season rice, so its weather inputs are taken from the monsoon months only: <b>June, July, August and September (JJAS)</b>. The rainfall value is the JJAS total, and the temperature values come from the same four months. The same months are used for the historical data (1970–2022) and for the World Bank projections (2023–2100).</p>
      </div></div>
    </div>
    <p class="note" style="margin-top:1rem">Dates are general; exact timing varies by region and variety.</p>
  </section>
</main>

<!-- ================= PERFORMANCE ================= -->
<main class="page" id="page-performance">
  <section class="wrap">
    <h2 class="page-title">Performance &amp; Comparison</h2>
    <p class="lead">Test-set results from the research experiments using <b>chronological 80/20 validation</b>, read directly from your results file.</p>
    <div class="seg c" id="cmpSeg">
      <button class="active" data-v="with_temp">With Temperature</button>
      <button data-v="without_temp">Without Temperature</button>
    </div>
    <div class="grid-2">
      <div class="card"><h3>R² (higher is better)</h3><div class="chart-wrap sm"><canvas id="r2Chart"></canvas></div></div>
      <div class="card"><h3>MAE and RMSE (lower is better)</h3><div class="chart-wrap sm"><canvas id="errChart"></canvas></div></div>
    </div>
    <div class="card" style="margin-top:1.2rem">
      <h3>Metrics table</h3>
      <div style="overflow-x:auto"><table id="cmpTable"><thead><tr><th>Model</th><th>R²</th><th>MAE (LMT)</th><th>RMSE (LMT)</th></tr></thead><tbody></tbody></table></div>
      <p class="note">Highlighted row = highest R². Chronological validation: the model is trained on earlier years and tested on the latest years.</p>
    </div>
    <div class="stack">
      <div class="card"><h3>Actual vs predicted production (1970–2022)</h3><div class="chart-wrap"><canvas id="fitChart"></canvas></div>
        <p class="note">This final model is fitted on all 1970–2022 data, so this chart shows fit, not out-of-sample accuracy. Use the table above for validated performance.</p></div>
      <div class="card"><h3>With vs without temperature: forecasts to 2100</h3><div class="chart-wrap"><canvas id="wwChart"></canvas></div>
        <p class="note">Both configurations use the same World Bank rainfall and land estimates; only the temperature inputs differ.</p></div>
    </div>
  </section>
</main>

<!-- ================= DATA ================= -->
<main class="page" id="page-data">
  <section class="wrap">
    <h2 class="page-title">Data Sources</h2>
    <p class="lead">Everything shown comes from the datasets below. No live weather API is used.</p>
    <div class="grid-2">
      <div class="card"><div class="ico">🗂️</div><h3>Historical dataset (1970–2022)</h3><p>Aman production (LMT), total rainfall, maximum and minimum temperature, and cultivated land. Rainfall and temperature are for the monsoon months June–September (JJAS). This is the training data for the model.</p></div>
      <div class="card"><div class="ico">🌍</div><h3>World Bank Climate Portal (2023–2100)</h3><p>Future JJAS (June–September) rainfall, maximum temperature and minimum temperature from a pre-downloaded WBCCKP CMIP6 projection file, read locally.</p></div>
      <div class="card"><div class="ico">🌾</div><h3>Cultivated land (future)</h3><p>For 2023–2100, land area is estimated with a linear trend fitted on the historical land data.</p></div>
      <div class="card"><div class="ico">📊</div><h3>Model performance</h3><p>Metrics come from the research experiment results file (chronological 80/20 split). Nothing is invented.</p></div>
    </div>
    <div class="card" style="margin-top:1.2rem"><h3>Please note</h3>
      <ul class="clean">
        <li>2023–2100 values are scenario-based estimates, not observed data.</li>
        <li>Future temperatures may fall outside the range seen during training, which makes forecasts less certain.</li>
        <li>Future cultivated land is a trend estimate.</li>
      </ul>
    </div>
  </section>
</main>

<footer>
  <h4>Forecasting Aman Rice Production in Bangladesh</h4>
  <p class="fq">“Where Algorithms Meet the Art of Agriculture”</p>
  <p>Developed by Sun</p>
  <p class="small">© 2026 All rights reserved.</p>
</footer>

<script>
const $ = id => document.getElementById(id);
const PAGES = ["home", "predict", "model", "aman", "performance", "data"];
let config = "with_temp", cmpFs = "with_temp", meta = null, sel = null, cur = null, metrics = [];
const charts = {};
const f = (v, d = 2) => Number(v).toFixed(d);
const C = {forest: "#1f4d2e", field: "#5b8c3a", sage: "#a8bfa0", gold: "#d9a520", red: "#c0392b", blue: "#4a7fa8"};

async function getJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}
function showError(m) { $("errorBox").textContent = m; $("errorBox").hidden = !m; }
function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), cfg);
}
const baseOpts = {responsive: true, maintainAspectRatio: false};

/* ---------- routing ---------- */
function route() {
  let p = location.hash.replace("#", "") || "home";
  if (!PAGES.includes(p)) p = "home";
  PAGES.forEach(x => $("page-" + x).classList.toggle("active", x === p));
  document.querySelectorAll(".menu a").forEach(a => a.classList.toggle("active", a.dataset.p === p));
  $("menu").classList.remove("open");
  window.scrollTo(0, 0);
  if (p === "predict") initPredict();
  if (p === "performance") initPerf();
}
window.addEventListener("hashchange", route);
// The browser does not fire "hashchange" when the link points to the page you are already on
// (e.g. clicking Home while on Home), so handle that case ourselves.
document.querySelectorAll('a[href^="#"]').forEach(a => a.addEventListener("click", e => {
  const target = a.getAttribute("href");
  if (target === location.hash || (target === "#home" && !location.hash)) {
    e.preventDefault();
    route();
  }
}));
$("burger").addEventListener("click", () => {
  const open = $("menu").classList.toggle("open");
  $("burger").setAttribute("aria-expanded", open);
});
document.addEventListener("click", e => {
  if (!e.target.closest(".nav")) $("menu").classList.remove("open");
});

/* ---------- prediction page ---------- */
let predictReady = false;
function initPredict() {
  if (!predictReady) { predictReady = true; applyConfig(); }
  loadInputs().then(() => { loadSeries(); loadClimate(); });
}
function setYear(v) { $("yearRange").value = v; $("yearNum").value = v; sel = null; initPredict(); }
$("yearRange").addEventListener("input", e => setYear(e.target.value));
$("yearNum").addEventListener("change", e => {
  const v = parseInt(e.target.value);
  if (isNaN(v) || v < meta.first_year || v > meta.last_year) {
    showError("Please enter a year between " + meta.first_year + " and " + meta.last_year + "."); return;
  }
  setYear(v);
});
document.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => setYear(c.dataset.y)));
document.querySelectorAll("#configSeg button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#configSeg button").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); config = b.dataset.v; sel = null; applyConfig(); loadSeries();
}));
function applyConfig() {
  document.querySelectorAll("[data-temp]").forEach(el => el.classList.toggle("off", config === "without_temp"));
}

async function loadInputs() {
  showError("");
  try {
    const d = await getJSON("/api/inputs?year=" + $("yearNum").value);
    cur = {year: d.Year, rain: d["Total_Rainfall_(mm)"], tmax: d["Maximum_Temperature"], tmin: d["Minimum_Temperature"]};
    $("inRain").textContent = f(d["Total_Rainfall_(mm)"], 1);
    $("inTmax").textContent = f(d["Maximum_Temperature"]);
    $("inTmin").textContent = f(d["Minimum_Temperature"]);
    $("inLand").textContent = f(d["Total_Land"], 0);
    const h = d.period === "historical";
    $("periodBadge").textContent = h ? "Historical" : "Future";
    $("periodBadge").classList.toggle("hist", h);
    $("periodNote").textContent = h ? "Observed weather and land (training period)" : "World Bank projected climate";
    $("srcNote").textContent = "Weather: " + d.source + ". Land: " + d.land_source + ".";
  } catch (e) { showError(e.message); }
}

$("predictBtn").addEventListener("click", async () => {
  showError("");
  try {
    const d = await getJSON("/api/predict?year=" + $("yearNum").value + "&config=" + config);
    $("resultVal").textContent = f(d.prediction_lmt);
    $("rYear").textContent = d.year;
    $("rConfig").textContent = d.config_label;
    $("rModel").textContent = d.climate_model;
    $("rScen").textContent = d.climate_scenario;
    $("rRain").textContent = f(d.inputs.rainfall, 1) + " mm";
    $("rTmax").textContent = f(d.inputs.tmax) + " °C";
    $("rTmin").textContent = f(d.inputs.tmin) + " °C";
    $("rLand").textContent = f(d.inputs.land, 0);
    const a = $("actualLine");
    if (d.actual_lmt !== null) { a.hidden = false; a.textContent = "Actual recorded production: " + f(d.actual_lmt) + " LMT"; }
    else a.hidden = true;
    sel = {x: d.year, y: d.prediction_lmt};
    loadSeries();
    const rc = $("resultCard");
    rc.scrollIntoView({behavior: "smooth", block: "center"});
    rc.classList.remove("pulse"); void rc.offsetWidth; rc.classList.add("pulse");
  } catch (e) { showError(e.message); }
});

async function loadSeries() {
  const s = await getJSON("/api/series?config=" + config);
  const actual = s.actual.map(p => ({x: p.year, y: p.actual}));
  const fitted = s.predicted.filter(p => p.year <= s.train_end).map(p => ({x: p.year, y: p.pred}));
  const future = s.predicted.filter(p => p.year >= s.train_end).map(p => ({x: p.year, y: p.pred}));
  const ds = [
    {label: "Actual (historical)", data: actual, borderColor: C.forest, backgroundColor: C.forest, pointRadius: 2, tension: .2},
    {label: "Model fit (1970–2022)", data: fitted, borderColor: C.sage, borderDash: [4, 3], pointRadius: 0, tension: .2},
    {label: "Forecast (2023–2100)", data: future, borderColor: C.gold, borderDash: [8, 4], borderWidth: 2.5, pointRadius: 0, tension: .2}
  ];
  if (sel) ds.push({label: "Predicted year", data: [sel], showLine: false, pointRadius: 9, pointBackgroundColor: C.red, pointBorderColor: "#fff", pointBorderWidth: 2});
  mk("prodChart", {type: "line", data: {datasets: ds}, options: {...baseOpts, parsing: false, interaction: {mode: "nearest", intersect: false},
    scales: {x: {type: "linear", min: 1970, max: 2100, title: {display: true, text: "Year"}},
             y: {title: {display: true, text: "Production (LMT)"}}},
    plugins: {legend: {position: "bottom"}}}});
}

async function loadClimate() {
  const s = (await getJSON("/api/climate")).series;
  const ds = [
    {label: "Rainfall (mm)", data: s.map(p => ({x: p.year, y: p.rain})), borderColor: C.blue, yAxisID: "y", pointRadius: 0, tension: .2},
    {label: "Max temp (°C)", data: s.map(p => ({x: p.year, y: p.tmax})), borderColor: C.red, yAxisID: "y2", pointRadius: 0, tension: .2},
    {label: "Min temp (°C)", data: s.map(p => ({x: p.year, y: p.tmin})), borderColor: C.gold, yAxisID: "y2", pointRadius: 0, tension: .2}
  ];
  if (cur) ds.push({label: "Selected year", data: [{x: cur.year, y: cur.rain}], yAxisID: "y", showLine: false, pointRadius: 8, pointBackgroundColor: C.forest, pointBorderColor: "#fff", pointBorderWidth: 2});
  mk("climChart", {type: "line", data: {datasets: ds}, options: {...baseOpts, parsing: false, interaction: {mode: "nearest", intersect: false},
    scales: {x: {type: "linear", min: 1970, max: 2100, title: {display: true, text: "Year"}},
             y: {position: "left", title: {display: true, text: "Rainfall (mm)"}},
             y2: {position: "right", grid: {drawOnChartArea: false}, title: {display: true, text: "Temperature (°C)"}}},
    plugins: {legend: {position: "bottom"}}}});
}

/* ---------- performance page ---------- */
let perfReady = false;
async function initPerf() {
  if (!perfReady) {
    perfReady = true;
    try { const d = await getJSON("/api/metrics"); metrics = d.rows; if (!d.available) metrics = null; } catch (e) { metrics = null; }
  }
  renderCompare();
  loadFit(); loadWW();
}
function renderCompare() {
  const tb = document.querySelector("#cmpTable tbody");
  if (!metrics) {
    tb.innerHTML = '<tr><td colspan="4">Results file not found. Run your original research script to create Temperature_Feature_Comparison.csv.</td></tr>';
    return;
  }
  const order = ["LR", "DT", "RF", "GB", "XGB", "LR+RF (Proposed)"];
  const rows = order.map(m => metrics.find(r => r.model === m && r.feature_set === cmpFs)).filter(Boolean);
  const best = Math.max(...rows.map(r => r.r2));
  tb.innerHTML = rows.map(r => '<tr class="' + (r.r2 === best ? "best" : "") + '"><td>' + r.model + "</td><td>" + f(r.r2, 4) + "</td><td>" + f(r.mae, 3) + "</td><td>" + f(r.rmse, 3) + "</td></tr>").join("");
  const colors = rows.map(r => r.model.indexOf("Proposed") >= 0 ? C.gold : C.field);
  mk("r2Chart", {type: "bar", data: {labels: rows.map(r => r.model), datasets: [{label: "R²", data: rows.map(r => r.r2), backgroundColor: colors}]},
    options: {...baseOpts, plugins: {legend: {display: false}}}});
  mk("errChart", {type: "bar", data: {labels: rows.map(r => r.model), datasets: [
    {label: "MAE (LMT)", data: rows.map(r => r.mae), backgroundColor: C.sage},
    {label: "RMSE (LMT)", data: rows.map(r => r.rmse), backgroundColor: C.forest}]},
    options: {...baseOpts, plugins: {legend: {position: "bottom"}}}});
}
document.querySelectorAll("#cmpSeg button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#cmpSeg button").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); cmpFs = b.dataset.v; renderCompare(); loadFit();
}));
async function loadFit() {
  const s = await getJSON("/api/series?config=" + cmpFs);
  mk("fitChart", {type: "line", data: {datasets: [
    {label: "Actual", data: s.actual.map(p => ({x: p.year, y: p.actual})), borderColor: C.forest, pointRadius: 2, tension: .2},
    {label: "Predicted (fit)", data: s.predicted.filter(p => p.year <= s.train_end).map(p => ({x: p.year, y: p.pred})), borderColor: C.gold, borderDash: [6, 3], pointRadius: 0, tension: .2}]},
    options: {...baseOpts, parsing: false, scales: {x: {type: "linear", min: 1970, max: 2022, title: {display: true, text: "Year"}}, y: {title: {display: true, text: "Production (LMT)"}}}, plugins: {legend: {position: "bottom"}}}});
}
async function loadWW() {
  const a = await getJSON("/api/series?config=with_temp");
  const b = await getJSON("/api/series?config=without_temp");
  const fut = s => s.predicted.filter(p => p.year >= 2023).map(p => ({x: p.year, y: p.pred}));
  mk("wwChart", {type: "line", data: {datasets: [
    {label: "With temperature", data: fut(a), borderColor: C.red, pointRadius: 0, tension: .2},
    {label: "Without temperature", data: fut(b), borderColor: C.forest, pointRadius: 0, tension: .2}]},
    options: {...baseOpts, parsing: false, scales: {x: {type: "linear", min: 2023, max: 2100, title: {display: true, text: "Year"}}, y: {title: {display: true, text: "Production (LMT)"}}}, plugins: {legend: {position: "bottom"}}}});
}

/* ---------- start ---------- */
(async function init() {
  try { meta = await getJSON("/api/meta"); } catch (e) { meta = {first_year: 1970, last_year: 2100}; }
  route();
})();
</script>
</body>
</html>
"""

def lan_ip():
    """Best-effort guess of this computer's Wi-Fi/LAN address (for opening the site on a phone)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no data is sent; just picks the active network adapter
        return s.getsockname()[0]
    except OSError:
        return "YOUR-PC-IP"
    finally:
        s.close()


if __name__ == "__main__":
    print(f"Trained on {len(hist)} rows ({FIRST_YEAR}-{TRAIN_END}); World Bank rows: {len(wb)}")
    print("\nOpen on this computer :  http://127.0.0.1:5000")
    print(f"Open on your phone    :  http://{lan_ip()}:5000   (same Wi-Fi)\n")
    app.run(host="0.0.0.0", debug=False, port=5000)