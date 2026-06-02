import streamlit as st

STYLES = """
<style>
:root{
  --bg:#0b0d10;
  --card:#0f1114;
  --card-soft:rgba(255,255,255,0.035);
  --muted:#9aa2aa;
  --accent-blue:#2E86AB;
  --accent-red:#D64545;
  --accent-green:#91d94f;
  --border: rgba(255,255,255,0.10);
  --widget: rgba(255,255,255,0.06);
  --text: #e6eef6;
}

/* Reduce Streamlit chrome so the app feels more like a product */
#MainMenu,
footer,
[data-testid="stDeployButton"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{
  visibility: hidden !important;
}

/* App + main containers */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"]{
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--bg) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stButton > button{
  justify-content: flex-start !important;
  min-height: 38px !important;
  border-color: rgba(255,255,255,0.08) !important;
  background: rgba(255,255,255,0.025) !important;
  font-weight: 750 !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled{
  opacity: 1 !important;
  background: rgba(46,134,171,0.20) !important;
  border-color: rgba(46,134,171,0.60) !important;
  color: #e6eef6 !important;
  box-shadow: inset 3px 0 0 rgba(145,217,79,0.75) !important;
}
section[data-testid="stSidebar"] hr{
  border-color: rgba(255,255,255,0.10) !important;
  margin: 1.35rem 0 !important;
}

/* Generic “card” helpers */
.stCard{
  background: var(--card);
  border-radius:12px;
  padding:14px;
}
.preview-card{
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.02));
  padding:12px;
  border-radius:10px;
}
.small-muted{color:var(--muted);}

/* Force BaseWeb (Streamlit widgets) to dark */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div{
  background-color: var(--widget) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}

/* Dropdown menu */
div[role="listbox"]{
  background-color: #0f1114 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
div[role="option"]{
  background-color: transparent !important;
  color: var(--text) !important;
}

/* Buttons */
.stButton > button{
  background-color: rgba(255,255,255,0.04) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
.stButton > button:hover{
  background-color: rgba(255,255,255,0.08) !important;
}

/* Radio groups as segmented controls */
div[role="radiogroup"]{
  gap: 8px !important;
  flex-wrap: wrap !important;
}
div[role="radiogroup"] label{
  min-height: 38px !important;
  padding: 8px 12px !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  background: rgba(255,255,255,0.025) !important;
  transition: background 120ms ease, border-color 120ms ease !important;
}
div[role="radiogroup"] label:hover{
  background: rgba(255,255,255,0.06) !important;
  border-color: rgba(255,255,255,0.18) !important;
}
div[role="radiogroup"] label:has(input:checked){
  background: rgba(46,134,171,0.20) !important;
  border-color: rgba(46,134,171,0.62) !important;
  box-shadow: 0 0 0 1px rgba(46,134,171,0.20) inset !important;
}
div[role="radiogroup"] label > div:first-child{
  display: none !important;
}

/* Metric tiles */
div[data-testid="stMetric"]{
  padding: 12px 14px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,0.025) !important;
}
div[data-testid="stMetricLabel"]{
  color: var(--muted) !important;
}

/* Dataframes / tables */
div[data-testid="stDataFrame"],
div[data-testid="stTable"]{
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

/* Page density */
[data-testid="stMainBlockContainer"]{
  padding-top: 2rem !important;
}

/* Markdown links */
a { color: #6fb3d2 !important; }

/* Mobile polish */
@media (max-width: 760px){
  [data-testid="stMainBlockContainer"]{
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    padding-top: 0.8rem !important;
  }

  h1{font-size: 1.55rem !important; line-height: 1.18 !important;}
  h2{font-size: 1.30rem !important; line-height: 1.20 !important;}
  h3{font-size: 1.08rem !important; line-height: 1.22 !important;}

  .stButton > button,
  button[kind="primary"],
  button[kind="secondary"],
  div[data-testid="stFormSubmitButton"] button{
    min-height: 46px !important;
    border-radius: 12px !important;
    font-size: 0.98rem !important;
    white-space: normal !important;
  }

  .stCard{
    padding: 10px !important;
    border-radius: 10px !important;
  }

  div[data-testid="stHorizontalBlock"]{
    gap: 0.55rem !important;
  }

  div[data-testid="stHorizontalBlock"] > div{
    min-width: 0 !important;
  }

  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div,
  div[data-baseweb="textarea"] > div{
    min-height: 44px !important;
  }

  div[data-testid="stDataFrame"],
  div[data-testid="stTable"]{
    overflow-x: auto !important;
    border-radius: 10px !important;
  }
}
</style>
"""

def apply_base_style():
    # IMPORTANT: don't call set_page_config here if app.py already does it
    st.markdown(STYLES, unsafe_allow_html=True)
