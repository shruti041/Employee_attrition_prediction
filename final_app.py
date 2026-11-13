"""Employee Attrition Prediction - Streamlit app.

Two pages:
- Team (first): upload a CSV exported from the HR system, score every employee
  and rank them. Selecting an employee opens their individual assessment.
- Individual employee: HR records are filled in from the file; HR adds the
  engagement answers only they can provide, then predicts. The new score
  flows back into the Team table. The page also works as a blank form.

Run `python train_model.py` first to create attrition_model.pkl.
"""
import base64
import datetime as dt
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon=":material/groups:",
    layout="wide",
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("attrition_model.pkl")


bundle = load_bundle()
MODEL = bundle["model"]
FEATURES = bundle["features"]
CATEGORICAL = bundle["categorical"]
SURVEY_DEFAULTS = bundle["survey_defaults"]
BASELINE = bundle["baseline"]
COEF = MODEL.named_steps["model"].coef_[0]
ENCODED_NAMES = MODEL.named_steps["preprocess"].get_feature_names_out()
# Which encoded model columns belong to each of the 16 questions.
FEATURE_COLUMNS = {
    f: [i for i, n in enumerate(ENCODED_NAMES) if n == f"num__{f}" or n.startswith(f"cat__{f}_")]
    for f in FEATURES
}
SURVEY = list(SURVEY_DEFAULTS)                          # 5 answers only HR can provide
HR_RECORDS = [f for f in FEATURES if f not in SURVEY]   # 11 answers from the HR system

JOB_ROLES = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources",
]
MARITAL = ["Single", "Married", "Divorced"]
OVERTIME = ["No", "Yes"]
TRAVEL = {"Non-Travel": "None", "Travel_Rarely": "Rarely", "Travel_Frequently": "Frequently"}
RATING = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}
BALANCE = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
SURVEY_QUESTIONS = {
    "JobSatisfaction": ("Job satisfaction", "How satisfied are you with your job?", RATING),
    "EnvironmentSatisfaction": ("Environment satisfaction", "How satisfied are you with your work environment?", RATING),
    "RelationshipSatisfaction": ("Relationship satisfaction", "How satisfied are you with your working relationships?", RATING),
    "JobInvolvement": ("Job involvement", "How involved do you feel in your work?", RATING),
    "WorkLifeBalance": ("Work-life balance", "How would you rate your work-life balance?", BALANCE),
}
# Number inputs: (label, min, max). Ranges are wider than the training data so
# other companies' records still load.
NUMBERS = {
    "Age": ("Age", 18, 70),
    "DistanceFromHome": ("Distance from home (km)", 1, 50),
    "NumCompaniesWorked": ("Companies worked at", 0, 20),
    "YearsInCurrentRole": ("Years in current role", 0, 40),
    "YearsWithCurrManager": ("Years with current manager", 0, 40),
    "YearsSinceLastPromotion": ("Years since last promotion", 0, 40),
    "TrainingTimesLastYear": ("Training sessions last year", 0, 20),
}
BLANK_FORM = {
    "OverTime": "No", "JobRole": "Sales Executive", "MaritalStatus": "Single",
    "BusinessTravel": "Travel_Rarely", "Age": 30, "DistanceFromHome": 10, "NumCompaniesWorked": 2,
    "TrainingTimesLastYear": 3, "YearsInCurrentRole": 3, "YearsWithCurrManager": 4,
    "YearsSinceLastPromotion": 2, **{col: 3 for col in SURVEY},
}
HIGH_RISK, WATCH_RISK = 0.5, 0.3


def contributions(df):
    """How much each answer pushes risk up (+) or down (-), in log-odds.

    The model is linear, so coefficient x (employee - average employee) is an
    exact breakdown of the score, summed over each question's encoded columns.
    """
    encoded = MODEL.named_steps["preprocess"].transform(df[FEATURES])
    encoded = encoded.toarray() if hasattr(encoded, "toarray") else np.asarray(encoded)
    parts = (encoded - BASELINE) * COEF
    return pd.DataFrame(
        {f: parts[:, cols].sum(axis=1) for f, cols in FEATURE_COLUMNS.items()},
        index=df.index,
    )


def count(n, word, plural=None):
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def describe(feature, value):
    """A short, plain-English version of one answer."""
    if feature == "OverTime":
        return "Works overtime" if value == "Yes" else "No overtime"
    if feature == "JobRole":
        return f"{value} role"
    if feature == "MaritalStatus":
        return value
    if feature == "BusinessTravel":
        return {"Travel_Frequently": "Travels frequently", "Travel_Rarely": "Travels rarely",
                "Non-Travel": "Doesn't travel"}[value]
    if feature in SURVEY_QUESTIONS:
        label, _, scale = SURVEY_QUESTIONS[feature]
        return f"{scale[int(value)]} {label.lower()}"
    v = int(value)
    return {
        "Age": f"Age {v}",
        "DistanceFromHome": f"Lives {v} km away",
        "NumCompaniesWorked": f"Worked at {count(v, 'company', 'companies')}",
        "TrainingTimesLastYear": f"{count(v, 'training session')} last year",
        "YearsInCurrentRole": f"{count(v, 'year')} in current role",
        "YearsWithCurrManager": f"{count(v, 'year')} with current manager",
        "YearsSinceLastPromotion": f"{count(v, 'year')} since last promotion",
    }[feature]


def key_signals(answers, contrib, direction=1, limit=3, threshold=0.15):
    """The answers that pushed risk most in one direction (1 = up, -1 = down)."""
    ranked = (contrib * direction).sort_values(ascending=False)
    return [describe(f, answers[f]) for f, v in ranked.items() if v > threshold][:limit]


def table_score(record):
    """(chance of leaving, used common answers?) for one employee, as the Team table scores them."""
    guessed = any(record[c] is None for c in SURVEY)
    filled = {c: (SURVEY_DEFAULTS[c] if c in SURVEY and record[c] is None else record[c]) for c in FEATURES}
    return float(MODEL.predict_proba(pd.DataFrame([filled])[FEATURES])[0, 1]), guessed


def risk_band(p):
    return "High" if p >= HIGH_RISK else "Watch" if p >= WATCH_RISK else "Low"


# --------------------------------------------------------------------------
# Icons. Streamlit strips <svg> from st.html, so each icon is drawn as a CSS
# mask over a span coloured with currentColor (it follows the text colour).
# --------------------------------------------------------------------------
ICONS = {
    "team": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7"/><path d="M18.5 14.4c1.8.8 3 2.8 3 5.6"/>',
    "person": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "file": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6Z"/><path d="M14 3v6h6"/>',
    "survey": '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 2h6v4H9zM9 12h6M9 16h4"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16v4Z"/><path d="m13 7 4 4"/>',
    "up": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "down": '<path d="M12 5v14M6 13l6 6 6-6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "warn": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z"/>',
    "building": '<path d="M3 21h18M5 21V7l7-4 7 4v14"/><path d="M9 21v-6h6v6"/>',
    "balance": '<path d="M12 3v18M5 8h14M7 8l-3 7a3 3 0 0 0 6 0L7 8ZM17 8l-3 7a3 3 0 0 0 6 0l-3-7Z"/>',
    "plane": '<path d="M17.8 19.2 16 11l3.5-3.5a2.1 2.1 0 0 0-3-3L13 8 4.8 6.2 3.5 7.5 10 11l-3 3H4l-1 1 3 2 2 3 1-1v-3l3-3 3.5 6.5Z"/>',
    "growth": '<path d="m3 17 6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    "people": '<circle cx="8" cy="8" r="3"/><circle cx="16" cy="8" r="3"/><path d="M2 20c0-3 2.7-5 6-5s6 2 6 5M14 15.2c.6-.1 1.3-.2 2-.2 3.3 0 6 2 6 5"/>',
    "learn": '<path d="M2 7l10-4 10 4-10 4L2 7Z"/><path d="M6 9v5c0 1.7 2.7 3 6 3s6-1.3 6-3V9"/>',
    "bulb": '<path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2Z"/>',
    "result": '<path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 12 20 4M15 4h5v5"/>',
}


def _icon_css(name, paths):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
           'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">' + paths + "</svg>")
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'.ar-i-{name}{{-webkit-mask-image:url("{uri}");mask-image:url("{uri}")}}'


ICON_CSS = "".join(_icon_css(name, paths) for name, paths in ICONS.items())


def icon(name, size=16):
    return f'<span class="ar-i ar-i-{name}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>'


# --------------------------------------------------------------------------
# Theme-aware styling for the custom cards (Streamlit widgets use config.toml).
# Note: Streamlit drops a <style> block containing "<" plus a letter, so the
# CSS below must never contain one (use the CSS escape \3C instead).
# --------------------------------------------------------------------------
PALETTES = {
    "light": {
        "surface": "#FFFFFF", "surface2": "#EEF0F6", "border": "#E2E5EE", "text": "#151A2D",
        "muted": "#586078", "primary": "#4338CA", "primary-soft": "#EEEEFF", "primary-text": "#3730A3",
        "on-primary": "#FFFFFF", "high": "#C2410C", "high-soft": "#FFF1E8", "low": "#0F766E",
        "low-soft": "#E6F5F2", "track": "#E2E5EE", "focus": "rgba(67,56,202,.35)",
        "shadow": "0 1px 2px rgba(21,26,45,.05),0 8px 24px rgba(21,26,45,.06)",
        "seg-shadow": "0 1px 3px rgba(21,26,45,.14)",
    },
    "dark": {
        "surface": "#151B27", "surface2": "#1C2333", "border": "#272F42", "text": "#E7EAF3",
        "muted": "#9AA3BD", "primary": "#5B5EE8", "primary-soft": "#23264A", "primary-text": "#C3C4FF",
        "on-primary": "#FFFFFF", "high": "#FB923C", "high-soft": "#2B1D13", "low": "#2DD4BF",
        "low-soft": "#0F2A28", "track": "#272F42", "focus": "rgba(155,156,255,.45)",
        "shadow": "0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.28)",
        "seg-shadow": "0 1px 3px rgba(0,0,0,.45)",
    },
}
theme = st.context.theme.type if st.context.theme.type in PALETTES else "light"
css_vars = ";".join(f"--{k}:{v}" for k, v in PALETTES[theme].items())

st.html("<style>.stApp{" + css_vars + "}" + ICON_CSS + """
[data-testid="stMainBlockContainer"]{max-width:1240px;padding-top:4.5rem}
.ar-i{display:inline-block;flex-shrink:0;background-color:currentColor;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;-webkit-mask-size:contain;mask-size:contain}
[class*="st-key-card-"]{background:var(--surface);border:1px solid var(--border) !important;border-radius:16px !important;box-shadow:var(--shadow);padding:24px !important;animation:ar-fade .55s cubic-bezier(.2,.8,.2,1) both}
[data-testid="stButtonGroup"] [role="radiogroup"]{background:var(--surface2);padding:4px;border-radius:12px;gap:4px;flex-wrap:nowrap}
button[data-variant="segmented_control"]{border:0 !important;margin:0 !important;border-radius:9px !important;background:transparent !important;color:var(--muted) !important;min-height:40px;padding:4px 6px !important;font-size:13px;transition:background-color .2s ease,color .2s ease,box-shadow .2s ease}
button[data-variant="segmented_control"] p{font-size:13px;white-space:nowrap}
button[data-variant="segmented_control"]:hover{color:var(--text) !important}
button[data-variant="segmented_control"][aria-checked="true"]{background:var(--surface) !important;color:var(--text) !important;box-shadow:var(--seg-shadow)}
button[data-variant="segmented_control"][aria-checked="true"] p{font-weight:600}
button[data-variant="segmented_control"]:focus-visible{outline:3px solid var(--focus) !important;outline-offset:2px}
.ar{color:var(--text);font-size:15px;line-height:1.5}
.ar *{box-sizing:border-box}
.ar-muted{color:var(--muted);font-size:13px}
.ar-header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:4px}
.ar-logo{width:40px;height:40px;border-radius:12px;background:var(--primary);color:var(--on-primary);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ar-app-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:17px;line-height:1.2}
.ar-model-chip{margin-left:auto;display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:var(--surface2);color:var(--muted);font-size:13px;font-weight:500}
.ar-hero{display:flex;flex-direction:column;gap:6px;max-width:760px;margin:8px 0 4px;animation:ar-fade .55s cubic-bezier(.2,.8,.2,1) both}
.ar-hero h1{margin:0;padding:0;font-family:'Space Grotesk',sans-serif;font-size:clamp(28px,4vw,38px);font-weight:600;letter-spacing:-.02em;line-height:1.15;color:var(--text)}
.ar-hero p{margin:0;font-size:16px;color:var(--muted)}
.ar-section-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.ar-step{width:32px;height:32px;border-radius:10px;background:var(--primary-soft);color:var(--primary-text);display:flex;align-items:center;justify-content:center;font-weight:600;font-size:14px;flex-shrink:0}
.ar-h2{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;line-height:1.3}
.ar-tag{margin-left:auto;display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;border:1px solid var(--border);color:var(--muted);font-size:12px;font-weight:500;white-space:nowrap}
.ar-tag.accent{border-color:transparent;background:var(--primary-soft);color:var(--primary-text)}
.ar-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px;display:flex;flex-direction:column;gap:14px;animation:ar-fade .55s cubic-bezier(.2,.8,.2,1) both}
.ar-card h3{margin:0;padding:0;font-size:15px;font-weight:600;color:var(--text)}
.ar-result{align-items:center;text-align:center;animation-name:ar-pop}
.ar-empty{align-items:center;text-align:center;padding:40px 24px;border-style:dashed}
.ar-empty-icon{width:48px;height:48px;border-radius:14px;background:var(--primary-soft);color:var(--primary-text);display:flex;align-items:center;justify-content:center}
.ar-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;font-size:13px;font-weight:600}
.ar-badge.high{background:var(--high-soft);color:var(--high)}
.ar-badge.low{background:var(--low-soft);color:var(--low)}
.ar-dot{width:8px;height:8px;border-radius:50%;background:currentColor;animation:ar-pulse 2s ease-out infinite}
.ar-gauge{position:relative;width:168px;height:168px}
.ar-ring{position:absolute;inset:0;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--ar-p) * 1%),var(--track) 0);-webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 14px),#000 calc(100% - 13px));mask:radial-gradient(farthest-side,transparent calc(100% - 14px),#000 calc(100% - 13px));animation:ar-ring 1.2s cubic-bezier(.2,.8,.2,1) .15s both}
.ar-gauge-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ar-pct{font-family:'Space Grotesk',sans-serif;font-size:36px;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}
.ar-verdict{margin:0;font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:600}
.ar-note{display:flex;gap:8px;align-items:flex-start;text-align:left;padding:10px 12px;border-radius:12px;background:var(--surface2);font-size:13px;color:var(--muted);width:100%}
.ar-note.warn{background:var(--high-soft);color:var(--text)}
.ar-note.ok{background:var(--low-soft);color:var(--text)}
.ar-note .ar-i{margin-top:2px}
.ar-chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 !important;padding:0 !important;list-style:none}
.ar-chip{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:13px;font-weight:500;margin:0 !important}
.ar-chip.up{background:var(--high-soft);color:var(--high)}
.ar-chip.down{background:var(--low-soft);color:var(--low)}
.ar-recs{margin:0 !important;padding:0 !important;list-style:none;display:flex;flex-direction:column;gap:2px}
.ar-rec{margin:0 !important;display:flex;gap:12px;align-items:flex-start;padding:8px;border-radius:12px;font-size:14px;animation:ar-fade .5s cubic-bezier(.2,.8,.2,1) both;transition:background-color .2s ease}
.ar-rec:hover{background:var(--surface2)}
.ar-rec-icon{width:32px;height:32px;border-radius:10px;background:var(--primary-soft);color:var(--primary-text);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ar-howto{margin:0 !important;padding:0 !important;list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px}
.ar-howto-step{margin:0 !important;display:flex;gap:12px;align-items:flex-start;font-size:14px}
.ar-count{font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;background:var(--primary-soft);color:var(--primary-text)}
.ar-tip{background:var(--primary-soft);border-color:transparent;flex-direction:row;align-items:flex-start;gap:14px;padding:20px 24px}
.ar-tip strong{color:var(--primary-text)}
.ar-q{display:flex;flex-direction:column;gap:2px;margin-bottom:-6px}
.ar-q-label{font-size:14px;font-weight:500;display:flex;justify-content:space-between;align-items:center;gap:8px}
.ar-q-source{font-size:12px;font-weight:400;color:var(--muted)}
.ar-needs{font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;background:var(--high-soft);color:var(--high);white-space:nowrap}
.ar-qbadge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;white-space:nowrap;animation:ar-swap .3s cubic-bezier(.2,.8,.2,1) both}
.ar-qbadge.needs{background:var(--high-soft);color:var(--high)}
.ar-qbadge.done{background:var(--low-soft);color:var(--low)}
.ar-qbadge.skip{background:var(--surface2);color:var(--muted)}
.ar-qbadge.file{box-shadow:inset 0 0 0 1px var(--border);color:var(--muted)}
.ar-fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.ar-field{display:flex;flex-direction:column;gap:2px;padding:10px 12px;border-radius:10px;background:var(--surface2)}
.ar-field span:first-child{font-size:12px;color:var(--muted)}
.ar-field span:last-child{font-size:14px;font-weight:500}
.ar-subhead{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:6px}
.ar-was{margin:0;font-size:13px;color:var(--muted)}
.ar-was strong{color:var(--text)}
.ar-result.stale .ar-ring,.ar-result.stale .ar-pct{opacity:.45}
.st-key-records [data-testid="stExpander"] details{background:var(--surface);border:1px solid var(--border);border-radius:16px;box-shadow:var(--shadow);animation:ar-fade .55s cubic-bezier(.2,.8,.2,1) both}
.st-key-records [data-testid="stExpander"] summary{min-height:56px;padding-left:20px;padding-right:20px}
.st-key-records [data-testid="stExpander"] summary:hover{background:var(--surface2);border-radius:16px}
.st-key-predict-error [data-testid="stAlert"]{animation:ar-shake .4s ease both}
[data-testid="stProgress"] > div > div > div > div{transition:width .45s cubic-bezier(.2,.8,.2,1)}
.ar-person{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.ar-avatar{width:48px;height:48px;border-radius:14px;background:var(--primary-soft);color:var(--primary-text);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ar-person-name{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:600;line-height:1.2}
@property --ar-p{syntax:'\\3C number>';inherits:false;initial-value:0}
@keyframes ar-fade{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes ar-swap{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:none}}
@keyframes ar-shake{0%,100%{transform:none}20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-4px)}80%{transform:translateX(4px)}}
@keyframes ar-pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:none}}
@keyframes ar-ring{from{--ar-p:0}}
@keyframes ar-pulse{0%{box-shadow:0 0 0 0 currentColor}70%{box-shadow:0 0 0 9px transparent}100%{box-shadow:0 0 0 0 transparent}}
@media (max-width:640px){.ar-model-chip{margin-left:0}.ar-card{padding:20px}}
@media (prefers-reduced-motion:reduce){.ar,.ar *,[class*="st-key-card-"],button[data-variant="segmented_control"],.st-key-records details,.st-key-predict-error [data-testid="stAlert"]{animation:none !important;transition:none !important}}
</style>""")


def section_header(step, title, subtitle, tag, tag_icon, accent=False):
    st.html(
        '<div class="ar ar-section-head">'
        + (f'<span class="ar-step">{step}</span>' if step else "")
        + f'<div style="flex:1 1 160px;min-width:0"><div class="ar-h2">{title}</div>'
        f'<div class="ar-muted">{subtitle}</div></div>'
        f'<span class="ar-tag{" accent" if accent else ""}">{icon(tag_icon, 13)}{tag}</span></div>'
    )


def hero(title, text):
    st.html(f'<div class="ar ar-hero"><h1>{title}</h1><p>{text}</p></div>')


# --------------------------------------------------------------------------
# State shared by both pages
# --------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("team_raw", None)        # the uploaded (or sample) DataFrame
ss.setdefault("team_source", None)     # its file name, shown to the user
ss.setdefault("team_file_id", None)    # detects when a different file is uploaded
ss.setdefault("upload_version", 0)     # bumping it clears the file uploader
ss.setdefault("table_version", 0)      # bumping it clears the table selection
ss.setdefault("overrides", {})         # employee id -> answers HR entered/corrected
ss.setdefault("selected", None)        # the employee opened from the Team page
ss.setdefault("results", {})           # assessment key -> last prediction


def clean_value(v):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


# --------------------------------------------------------------------------
# Team page
# --------------------------------------------------------------------------
def template_csv():
    example = {"EmployeeNumber": 1001, "Department": "Sales", **BLANK_FORM}
    example["JobSatisfaction"] = None  # shows that survey answers can be left blank
    return pd.DataFrame([example]).to_csv(index=False).encode("utf-8")


def prepare_upload(raw):
    """Check and clean an uploaded file. Returns (clean_df, bad_rows, error).

    Survey answers may be blank (kept as NaN, filled only when scoring).
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in HR_RECORDS if c not in df.columns]
    if missing:
        return None, 0, "The file is missing these columns: " + ", ".join(missing)
    for col in SURVEY:
        if col not in df.columns:
            df[col] = np.nan
    for col in FEATURES:
        if col in CATEGORICAL:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "EmployeeNumber" in df.columns:
        df["_id"] = df["EmployeeNumber"].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        df["_id"] = [str(i) for i in range(1, len(df) + 1)]
    allowed = {"OverTime": OVERTIME, "JobRole": JOB_ROLES, "MaritalStatus": MARITAL,
               "BusinessTravel": list(TRAVEL)}
    bad = df[HR_RECORDS].isna().any(axis=1)
    for col, values in allowed.items():
        bad |= ~df[col].isin(values)
    for col in SURVEY:
        bad |= df[col].notna() & ~df[col].between(1, 4)
    return df[~bad].reset_index(drop=True), int(bad.sum()), None


def apply_overrides(df):
    """Replace file values with anything HR entered on the Individual page."""
    work = df.copy()
    for emp_id, answers in ss.overrides.items():
        mask = work["_id"] == emp_id
        if mask.any():
            for col, value in answers.items():
                work.loc[mask, col] = np.nan if value is None else value
    work["_survey_known"] = work[SURVEY].notna().sum(axis=1)
    return work


def score_team(work):
    filled = work.fillna(SURVEY_DEFAULTS)
    probs = MODEL.predict_proba(filled[FEATURES])[:, 1]
    contrib = contributions(filled)
    records = filled[FEATURES].to_dict("records")
    out = pd.DataFrame({
        "Employee": work["_id"].values,
        "Chance of leaving": (probs * 100).round(1),
        "Job role": work["JobRole"].values,
    })
    if "Department" in work.columns:
        out["Department"] = work["Department"].astype(str).values
    out["Survey answers"] = [f"{k} of 5" for k in work["_survey_known"]]
    out["Key signals"] = [" · ".join(key_signals(rec, contrib.iloc[i])) for i, rec in enumerate(records)]
    out["Status"] = ["Updated by HR" if i in ss.overrides else "" for i in work["_id"]]
    out["Risk level"] = [risk_band(p) for p in probs]
    return out.sort_values("Chance of leaving", ascending=False).reset_index(drop=True)


def load_team_data(raw, source, file_id=None):
    ss.team_raw, ss.team_source, ss.team_file_id = raw, source, file_id
    ss.overrides, ss.results, ss.selected = {}, {}, None


def clear_team_data():
    load_team_data(None, None)
    ss.upload_version += 1


def team_page():
    hero("Team risk overview",
         "Upload an export from your HR system to score every employee at once. "
         "Then open anyone to enter their survey answers.")

    with st.container(border=True, key="card-upload"):
        uploaded = st.file_uploader(
            "Upload employee data (CSV)", type=["csv"], key=f"uploader_{ss.upload_version}",
            help="One row per employee with the columns in the template. EmployeeNumber and "
                 "Department are optional but make the table easier to use.",
        )
        if uploaded is not None and uploaded.file_id != ss.team_file_id:
            try:
                load_team_data(pd.read_csv(uploaded), uploaded.name, uploaded.file_id)
            except Exception:
                st.error("This file couldn't be read as a CSV. Save it as CSV (comma-separated) "
                         "and try again.", icon=":material/error:")
                return
        a, b, c = st.columns([1, 1, 1])
        a.download_button("Download template", template_csv(), "attrition_template.csv", "text/csv",
                          icon=":material/download:", width="stretch")
        if b.button("Try with sample data", icon=":material/dataset:", width="stretch",
                    help="Loads the IBM HR dataset the model was trained on."):
            load_team_data(pd.read_csv("HR-Employee-Attrition.csv"), "Sample data (IBM HR dataset)")
        if ss.team_raw is not None:
            c.button("Remove data", icon=":material/close:", width="stretch", on_click=clear_team_data)
        st.caption("Needs 11 columns from your HR records. The 5 engagement-survey columns are optional: "
                   "HR can add them per employee.")

    if ss.team_raw is None:
        steps = [
            ("1", "Download the template", "It lists the columns the model needs, with an example row."),
            ("2", "Fill it from your HR system", "Profile and career details for each employee. Add survey "
                                                 "answers if you have them, or leave them blank."),
            ("3", "Upload and review", "Everyone is scored and ranked. Open an employee to enter "
                                       "their survey answers and get an updated prediction."),
        ]
        items = "".join(
            f'<li class="ar-howto-step"><span class="ar-step">{n}</span>'
            f'<div><div style="font-weight:600">{title}</div><div class="ar-muted">{text}</div></div></li>'
            for n, title, text in steps
        )
        st.html(f'<div class="ar ar-card"><h3>How it works</h3><ol class="ar-howto">{items}</ol></div>')
        return

    df, bad_rows, error = prepare_upload(ss.team_raw)
    if error:
        st.error(error, icon=":material/error:")
        return
    if df.empty:
        st.error("No rows could be scored. Check the values against the template.", icon=":material/error:")
        return
    st.caption(f":material/description: Using **{escape(ss.team_source)}**")
    if bad_rows:
        st.warning(f"{count(bad_rows, 'row')} skipped because of missing or unexpected values "
                   "(for example a job role that isn't in the template).", icon=":material/warning:")

    work = apply_overrides(df)
    scored = score_team(work)
    n = len(scored)
    incomplete = int((work["_survey_known"] < 5).sum())
    if incomplete:
        st.info(f"{count(incomplete, 'employee')} {'is' if incomplete == 1 else 'are'} missing survey answers. "
                "Their scores use the most common answers until HR adds the real ones.", icon=":material/info:")

    bands = scored["Risk level"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees scored", f"{n:,}", "from the uploaded data", delta_color="off", delta_arrow="off",
              border=True)
    c2.metric(":orange[:material/warning: High risk · 50%+]", f"{bands.get('High', 0):,}",
              f"{bands.get('High', 0) / n:.1%} of employees", delta_color="off", delta_arrow="off", border=True)
    c3.metric(":material/visibility: Watch · 30–49%", f"{bands.get('Watch', 0):,}",
              f"{bands.get('Watch', 0) / n:.1%} of employees", delta_color="off", delta_arrow="off", border=True)
    c4.metric(":green[:material/check: Low risk · under 30%]", f"{bands.get('Low', 0):,}",
              f"{bands.get('Low', 0) / n:.1%} of employees", delta_color="off", delta_arrow="off", border=True)

    with st.container(border=True, key="card-table"):
        f1, f2, f3 = st.columns([2, 2, 3])
        search = f1.text_input("Search employee number", placeholder="Search employee number",
                               label_visibility="collapsed")
        department = "All departments"
        if "Department" in scored.columns:
            department = f2.selectbox("Department", ["All departments"] + sorted(scored["Department"].unique()),
                                      label_visibility="collapsed")
        # Start on the high-risk list when there is one; otherwise show everyone.
        band = f3.segmented_control("Risk level", ["All", "High", "Watch", "Low"],
                                    default="High" if bands.get("High", 0) else "All",
                                    required=True, label_visibility="collapsed", width="stretch")
        view = scored
        if band != "All":
            view = view[view["Risk level"] == band]
        if department != "All departments":
            view = view[view["Department"] == department]
        if search:
            view = view[view["Employee"].str.contains(search.strip(), regex=False)]
        view = view.reset_index(drop=True)

        st.caption(":material/touch_app: **Select an employee** to open their assessment and enter their "
                   "survey answers.")
        event = st.dataframe(
            view.drop(columns="Risk level"), hide_index=True, width="stretch",
            height="auto" if len(view) <= 10 else 420,
            placeholder="No employees match these filters. Choose All to see everyone.",
            key=f"team_table_{ss.table_version}", on_select="rerun", selection_mode="single-row",
            column_config={
                "Employee": st.column_config.TextColumn("Employee", width="small"),
                "Chance of leaving": st.column_config.ProgressColumn(
                    "Chance of leaving", format="%.1f%%", min_value=0, max_value=100),
                "Survey answers": st.column_config.TextColumn(
                    "Survey answers", help="How many of the 5 engagement answers are known"),
                "Key signals": st.column_config.TextColumn(
                    "Key signals", width="large", help="Answers that raise this employee's risk most"),
            },
        )
        a, b = st.columns([3, 1])
        a.caption(f"Showing {len(view):,} of {n:,} employees, highest risk first.")
        b.download_button("Download results", scored.to_csv(index=False).encode("utf-8"),
                          "attrition_scores.csv", "text/csv", icon=":material/download:",
                          type="primary", width="stretch")

    st.caption("Use these scores to decide who to check in with first, never to penalise anyone.")

    if event.selection.rows:
        emp_id = view.iloc[event.selection.rows[0]]["Employee"]
        row = work.loc[work["_id"] == emp_id].iloc[0]
        file_row = df.loc[df["_id"] == emp_id].iloc[0]
        record = {col: clean_value(row[col]) for col in FEATURES}
        file_record = {col: clean_value(file_row[col]) for col in FEATURES}
        previous, guessed = table_score(record)
        ss.selected = {
            "id": emp_id,
            "source": ss.team_source,
            "department": str(row["Department"]) if "Department" in work.columns else None,
            "record": record,                       # includes anything HR saved earlier
            "file_record": file_record,             # as uploaded, for "Clear answers"
            "file_survey": {c: file_record[c] for c in SURVEY},
            "missing_in_file": [c for c in SURVEY if file_record[c] is None],
            "previous": previous,                   # the Team table's score when opened
            "previous_guessed": guessed,
        }
        ss.table_version += 1  # so the table isn't still selected when HR comes back
        st.switch_page(EMPLOYEE_PAGE)


# --------------------------------------------------------------------------
# Individual employee page
# --------------------------------------------------------------------------
def survey_badge(value, unknown, file_value, missing):
    """The small status label next to each survey question."""
    if unknown:
        return "skip", "No answer"
    if value is None:
        return "needs", "Not in file" if missing else "Needed"
    if file_value is not None and value == file_value:
        return "file", "From file"
    return "done", "Entered"


def survey_rating(col, default, key, file_value, missing):
    label, question, scale = SURVEY_QUESTIONS[col]
    unknown_key, rating_key = f"unknown_{col}_{key}", f"rating_{col}_{key}"
    unknown = ss.get(unknown_key, False)
    value = ss.get(rating_key, default)
    kind, text = survey_badge(value, unknown, file_value, missing)
    source = " Or rated by their manager." if col == "JobInvolvement" else ""
    st.html(f'<div class="ar ar-q"><div class="ar-q-label"><span>{label}</span>'
            f'<span class="ar-qbadge {kind}">{text}</span></div>'
            f'<div class="ar-muted">Employee was asked: “{question}”{source}</div></div>')
    value = st.segmented_control(
        label, list(scale), default=default, required=default is not None, format_func=scale.get,
        label_visibility="collapsed", width="stretch", key=rating_key, disabled=unknown,
    )
    unknown = st.checkbox("No answer from the employee", key=unknown_key,
                          help="Uses the most common answer instead, and marks the result as less certain.")
    return value, unknown


def recommendations(answers, survey):
    """Retention suggestions. Survey-based ones only apply to known answers."""
    recs = []
    if answers["OverTime"] == "Yes":
        recs.append(("clock", "Consider reducing excessive overtime or redistributing workload."))
    if survey["JobSatisfaction"] is not None and survey["JobSatisfaction"] <= 2:
        recs.append(("chat", "Conduct a satisfaction survey or 1-on-1 session to understand dissatisfaction."))
    if survey["EnvironmentSatisfaction"] is not None and survey["EnvironmentSatisfaction"] <= 2:
        recs.append(("building", "Review the work environment: workspace, tools, and team culture."))
    if survey["WorkLifeBalance"] is not None and survey["WorkLifeBalance"] <= 2:
        recs.append(("balance", "Promote better work-life balance with flexible hours or remote options."))
    if answers["BusinessTravel"] == "Travel_Frequently":
        recs.append(("plane", "Look at reducing travel frequency or offering recovery time after trips."))
    if answers["YearsSinceLastPromotion"] >= 3:
        recs.append(("growth", "Evaluate if the employee is due for a promotion or new responsibilities."))
    if survey["RelationshipSatisfaction"] is not None and survey["RelationshipSatisfaction"] <= 2:
        recs.append(("people", "Encourage team bonding or assign a mentor to improve workplace relationships."))
    if answers["TrainingTimesLastYear"] < 2:
        recs.append(("learn", "Offer skill development or training opportunities to boost engagement."))
    return recs


def previous_text(prev, guessed):
    """How the Team table scored this employee before HR's answers."""
    return (f'<strong>{prev:.1%}</strong> in the Team table'
            + (", using the most common answers" if guessed else ""))


def render_result(res, stale):
    p, high = res["probability"], res["probability"] >= HIGH_RISK
    kind = "high" if high else "low"
    badge = "High attrition risk" if high else "Low attrition risk"
    verdict = "Likely to leave" if high else "Likely to stay"
    summary = ("This employee's answers closely match people who left." if high
               else "This employee's answers look like those of people who stayed.")

    was = ""
    if res.get("previous") is not None and round(res["previous"], 3) != round(p, 3):
        was = f'<p class="ar-was">Was {previous_text(res["previous"], res["previous_guessed"])}</p>'

    notes = ""
    if stale:
        notes += (f'<div class="ar-note warn" role="note">{icon("warn")}<span>Answers changed since this '
                  f'prediction. Press <strong>Predict attrition risk</strong> to update it.</span></div>')
    if res["unknown"]:
        names = ", ".join(n.lower() for n in res["unknown"])
        n = len(res["unknown"])
        notes += (f'<div class="ar-note" role="note">{icon("info")}<span>The employee gave no answer for '
                  f'{escape(names)}, so the most common {"answer was" if n == 1 else "answers were"} used. '
                  f'Treat this estimate with extra care.</span></div>')
    if res["survey_age_days"] is not None and res["survey_age_days"] > 365:
        notes += (f'<div class="ar-note warn" role="note">{icon("warn")}<span>The survey answers are '
                  f'over a year old. Talk with the employee before acting on this result.</span></div>')

    st.html(f"""
<div class="ar ar-card ar-result{' stale' if stale else ''}" aria-live="polite">
  <span class="ar-badge {kind}"><span class="ar-dot"></span>{badge}</span>
  <div class="ar-gauge" role="img" aria-label="{p:.1%} chance of leaving">
    <div class="ar-ring" style="--ar-p:{p * 100:.2f};--ring:var(--{kind})"></div>
    <div class="ar-gauge-label"><span class="ar-pct">{p:.1%}</span>
      <span class="ar-muted" style="margin-top:4px">chance of leaving</span></div>
  </div>
  <div><p class="ar-verdict">{verdict}</p><p class="ar-muted" style="margin:4px 0 0;font-size:14px">{summary}</p></div>
  {was}
  {notes}
</div>""")

    chips = "".join(f'<li class="ar-chip up">{icon("up", 14)}{escape(s)}</li>' for s in res["raises"])
    chips += "".join(f'<li class="ar-chip down">{icon("down", 14)}{escape(s)}</li>' for s in res["lowers"])
    if chips:
        st.html(f"""
<div class="ar ar-card" style="animation-delay:.2s">
  <h3>What drove this result</h3>
  <ul class="ar-chips">{chips}</ul>
  <span class="ar-muted">Orange arrows raise the risk, green arrows lower it, compared with an average employee.</span>
</div>""")

    if high and res["recommendations"]:
        items = "".join(
            f'<li class="ar-rec" style="animation-delay:{0.3 + i * 0.07:.2f}s">'
            f'<span class="ar-rec-icon">{icon(key)}</span><span>{escape(text)}</span></li>'
            for i, (key, text) in enumerate(res["recommendations"])
        )
        st.html(f"""
<div class="ar ar-card" style="animation-delay:.3s">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
    <h3>Retention recommendations</h3><span class="ar-count">{len(res["recommendations"])}</span></div>
  <ol class="ar-recs">{items}</ol>
</div>""")
    elif high:
        st.html(f'<div class="ar ar-card ar-tip" style="animation-delay:.3s">{icon("bulb", 22)}'
                '<div><strong>No single factor stands out.</strong><br>Schedule a career conversation to '
                'understand what would keep them.</div></div>')
    else:
        st.html(f'<div class="ar ar-card ar-tip" style="animation-delay:.3s">{icon("bulb", 22)}'
                '<div><strong>Keep it that way</strong><br>No major concerns detected. Regular feedback and '
                'career discussions still help.</div></div>')


def show_value(col, value):
    if col == "BusinessTravel":
        return TRAVEL[value]
    return str(value)


RECORD_LABELS = {
    "Age": "Age", "MaritalStatus": "Marital status", "JobRole": "Job role",
    "DistanceFromHome": "Distance from home (km)", "OverTime": "Works overtime",
    "BusinessTravel": "Business travel", **{c: NUMBERS[c][0] for c in NUMBERS},
}
CAREER = ["NumCompaniesWorked", "YearsInCurrentRole", "YearsWithCurrManager",
          "YearsSinceLastPromotion", "TrainingTimesLastYear"]


def record_inputs(record, key, cards):
    """The 11 HR-record inputs. In cards for the blank form, plain inside the HR records expander."""
    def section(card_key, step, title, subtitle):
        if cards:
            box = st.container(border=True, key=card_key)
            with box:
                section_header(step, title, subtitle, "HR records", "database")
            return box
        st.html(f'<div class="ar ar-subhead">{title}</div>')
        return st.container()

    with section("card-profile", 1, "Profile", "Who the employee is and how they work"):
        a, b = st.columns(2)
        age = a.slider("Age", 18, 70, int(np.clip(record["Age"], 18, 70)), key=f"age_{key}")
        marital = b.segmented_control("Marital status", MARITAL, default=record["MaritalStatus"],
                                      required=True, width="stretch", key=f"marital_{key}")
        a, b = st.columns(2)
        role = a.selectbox("Job role", JOB_ROLES, index=JOB_ROLES.index(record["JobRole"]), key=f"role_{key}")
        distance = b.slider("Distance from home (km)", 1, 50, int(np.clip(record["DistanceFromHome"], 1, 50)),
                            key=f"distance_{key}")
        a, b = st.columns(2)
        overtime = a.segmented_control("Works overtime", OVERTIME, default=record["OverTime"],
                                       required=True, width="stretch", key=f"overtime_{key}")
        travel = b.segmented_control("Business travel", list(TRAVEL), default=record["BusinessTravel"],
                                     format_func=TRAVEL.get, required=True, width="stretch", key=f"travel_{key}")

    numbers = {}
    with section("card-career", 2, "Career history", "Experience, tenure and growth"):
        for i in range(0, len(CAREER), 2):
            pair = st.columns(2)
            for col, slot in zip(CAREER[i:i + 2], pair):
                label, lo, hi = NUMBERS[col]
                numbers[col] = slot.number_input(label, lo, hi, int(np.clip(record[col], lo, hi)),
                                                 key=f"{col}_{key}")
    return {"OverTime": overtime, "JobRole": role, "MaritalStatus": marital, "BusinessTravel": travel,
            "Age": age, "DistanceFromHome": distance, **numbers}


def clear_answers(key):
    """Undo everything entered for this employee, here and in the Team table."""
    prefixes = ("rating_", "unknown_", "survey_date_", "edit_", "age_", "marital_", "role_", "distance_",
                "overtime_", "travel_", *(f"{c}_" for c in CAREER))
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith(prefixes) and k.endswith(f"_{key}")]:
        del ss[k]
    ss.results.pop(key, None)
    sel = ss.selected
    ss.overrides.pop(sel["id"], None)
    sel["record"] = dict(sel["file_record"])
    sel["previous"], sel["previous_guessed"] = table_score(sel["record"])


def employee_page():
    sel = ss.selected
    from_file = sel is not None
    key = f"emp_{sel['id']}" if from_file else "manual"
    record = sel["record"] if from_file else BLANK_FORM
    missing_in_file = sel["missing_in_file"] if from_file else []
    file_survey = sel["file_survey"] if from_file else {c: None for c in SURVEY}

    if from_file:
        with st.container(border=True, key="card-person"):
            banner = st.empty()
            a, b, _ = st.columns([1, 1, 2])
            if a.button("Back to team", icon=":material/arrow_back:", width="stretch"):
                st.switch_page(TEAM_PAGE)
            b.button("Clear answers", icon=":material/restart_alt:", width="stretch",
                     on_click=clear_answers, args=(key,),
                     help="Removes everything entered for this employee, here and in the Team table, "
                          "and goes back to the uploaded file.")
    else:
        hero("Assess an employee",
             "Fill in the details below to estimate how likely someone is to leave. To fill in HR records "
             "automatically, open an employee from the Team page.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        if from_file:
            # HR records come from the file: collapsed and read-only, so the page opens on the survey.
            with st.container(key="records"):
                with st.expander(f"**HR records** · 11 details from {sel['source']}", icon=":material/lock:"):
                    editing = st.toggle("Correct HR record details", key=f"edit_{key}",
                                        help="Applies to this assessment only, and is saved with it.")
                    if editing:
                        hr_answers = record_inputs(record, key, cards=False)
                    else:
                        hr_answers = {c: record[c] for c in HR_RECORDS}
                        tiles = "".join(
                            f'<div class="ar-field"><span>{RECORD_LABELS[c]}</span>'
                            f'<span>{escape(show_value(c, record[c]))}</span></div>'
                            for c in RECORD_LABELS)
                        st.html(f'<div class="ar ar-fields">{tiles}</div>')
        else:
            hr_answers = record_inputs(record, key, cards=True)

        with st.container(border=True, key="card-survey"):
            section_header(None if from_file else 3, "The employee's own answers",
                           "Satisfaction and engagement. Copy them from the engagement survey, or ask "
                           "in a check-in conversation.", "Entered by HR", "edit", accent=True)
            progress = st.empty()
            note = st.empty()
            survey_date = st.date_input(
                "Survey date", value=None, max_value=dt.date.today(), format="DD/MM/YYYY",
                key=f"survey_date_{key}",
                help="When the employee answered. Answers older than 12 months are flagged.",
            )
            survey = {}
            for i in range(0, len(SURVEY), 2):
                pair = st.columns(2)
                for col, slot in zip(SURVEY[i:i + 2], pair):
                    with slot:
                        default = None if record[col] is None else int(record[col])
                        survey[col] = survey_rating(col, default, key, file_survey[col], col in missing_in_file)
            st.caption(":material/lock: Only use surveys employees know are **confidential, not "
                       "anonymous**. Linking anonymous answers to a named person breaks the promise "
                       "made to them.")

        done = sum(v is not None or unknown for v, unknown in survey.values())
        progress.progress(done / 5, text=f"Answers entered · {done} of 5")
        if from_file and missing_in_file and done < 5:
            note.html(f'<div class="ar ar-note warn" role="note">{icon("warn")}<span>'
                      f'{count(len(missing_in_file), "answer")} {"was" if len(missing_in_file) == 1 else "were"} '
                      f'not in the uploaded file. Enter the employee\'s answer, or tick '
                      f'No answer from the employee.</span></div>')

        predict = st.button("Predict attrition risk", type="primary", icon=":material/arrow_forward:",
                            icon_position="right", width="stretch", key=f"predict_{key}")
        error = st.container(key="predict-error")
        st.caption("An estimate to help support employees, never a reason to penalise them.")

    if from_file:
        role_line = escape(record["JobRole"]) + (f" · {escape(sel['department'])}" if sel["department"] else "")
        status = ("All answers entered" if done == 5
                  else f"{5 - done} employee {'answer' if 5 - done == 1 else 'answers'} to enter")
        banner.html(f'<div class="ar ar-person"><span class="ar-avatar">{icon("person", 24)}</span>'
                    f'<div style="flex:1 1 200px;min-width:0"><div class="ar-person-name">Employee '
                    f'#{escape(sel["id"])}</div><div class="ar-muted">{role_line} · {status}</div></div></div>')

    known_survey = {c: (None if unknown else v) for c, (v, unknown) in survey.items()}
    answers = {**hr_answers}
    for col, value in known_survey.items():
        answers[col] = SURVEY_DEFAULTS[col] if value is None else value
    inputs = (answers, known_survey, survey_date)  # what the result was based on

    if predict:
        unanswered = [SURVEY_QUESTIONS[c][0] for c, (v, unknown) in survey.items() if v is None and not unknown]
        if unanswered:
            error.error("Choose an answer or tick **No answer from the employee** for: "
                        + ", ".join(unanswered) + ".", icon=":material/error:")
        else:
            row = pd.DataFrame([answers])[FEATURES]
            contrib = contributions(row).iloc[0]
            ss.results[key] = {
                "probability": float(MODEL.predict_proba(row)[0, 1]),
                "unknown": [SURVEY_QUESTIONS[c][0] for c, v in known_survey.items() if v is None],
                "survey_age_days": (dt.date.today() - survey_date).days if survey_date else None,
                "raises": key_signals(answers, contrib, direction=1, limit=4),
                "lowers": key_signals(answers, contrib, direction=-1, limit=3),
                "recommendations": recommendations(answers, known_survey),
                "previous": sel["previous"] if from_file else None,
                "previous_guessed": sel["previous_guessed"] if from_file else False,
                "inputs": inputs,
            }
            if from_file:
                # Store what HR entered so the Team page re-scores this employee.
                saved = {**{c: answers[c] for c in HR_RECORDS}, **known_survey}
                ss.overrides[sel["id"]] = saved
                ss.selected["record"] = saved
                st.toast(f"**Saved.** The Team page now uses these answers for employee #{sel['id']}.",
                         icon=":material/check_circle:")

    with right:
        result = ss.results.get(key)
        if result is None:
            hint = ("Enter the employee's answers and press <strong>Predict attrition risk</strong>. "
                    f"Until then the Team table shows <strong>{sel['previous']:.1%}</strong>"
                    + (", using the most common answers." if sel["previous_guessed"] else ".")
                    if from_file else "Fill in the details and press <strong>Predict attrition risk</strong>.")
            st.html(f"""
<div class="ar ar-card ar-empty">
  <span class="ar-empty-icon">{icon("result", 24)}</span>
  <h3>The result will appear here</h3>
  <span class="ar-muted">{hint}</span>
</div>""")
        else:
            render_result(result, stale=result["inputs"] != inputs)


# --------------------------------------------------------------------------
# App shell: header, then the two pages (Team first)
# --------------------------------------------------------------------------
st.html(f"""
<div class="ar ar-header">
  <div class="ar-logo">{icon("team", 22)}</div>
  <div><div class="ar-app-title">Employee Attrition Prediction</div><div class="ar-muted">HR retention toolkit</div></div>
  <span class="ar-model-chip">{icon("chart", 14)}Logistic Regression · {bundle["cv_accuracy"]:.1%} accuracy</span>
</div>""")

TEAM_PAGE = st.Page(team_page, title="Team", icon=":material/groups:", url_path="team", default=True)
EMPLOYEE_PAGE = st.Page(employee_page, title="Individual employee", icon=":material/person:",
                        url_path="employee")
st.navigation([TEAM_PAGE, EMPLOYEE_PAGE], position="top").run()
