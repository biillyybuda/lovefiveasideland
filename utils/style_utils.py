import streamlit as st

STYLES = """
<style>
:root{
  --bg:#0b0d10;
  --card:#0f1114;
  --muted:#9aa2aa;
  --accent-blue:#2E86AB;
  --accent-red:#D64545;
  --border: rgba(255,255,255,0.10);
  --widget: rgba(255,255,255,0.06);
  --text: #e6eef6;
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

/* Dataframes / tables */
div[data-testid="stDataFrame"],
div[data-testid="stTable"]{
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

/* Markdown links */
a { color: #6fb3d2 !important; }
</style>
"""

def apply_base_style():
    # IMPORTANT: don't call set_page_config here if app.py already does it
    st.markdown(STYLES, unsafe_allow_html=True)
