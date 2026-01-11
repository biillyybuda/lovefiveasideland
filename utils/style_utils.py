import streamlit as st
STYLES = '\n<style>\n:root{--bg:#0b0d10; --card:#0f1114; --muted:#9aa2aa; --accent-blue:#2E86AB; --accent-red:#D64545;}\nhtml, body, [data-testid="stAppViewContainer"]{background:var(--bg); color:#e6eef6;}\n.stCard{background:var(--card); border-radius:12px; padding:14px;}\n.preview-card{background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.02)); padding:12px; border-radius:10px;}\n.small-muted{color:var(--muted);}\n.badge.blue{background:linear-gradient(90deg,var(--accent-blue),#6fb3d2); color:white; padding:6px 10px; border-radius:999px;}\n.badge.red{background:linear-gradient(90deg,var(--accent-red),#f28b8b); color:white; padding:6px 10px; border-radius:999px;}\n</style>\n'

def apply_base_style():
    st.set_page_config(page_title='Love Five-A-Side App', layout='wide', initial_sidebar_state='expanded')
    st.markdown(STYLES, unsafe_allow_html=True)