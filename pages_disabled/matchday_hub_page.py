
import streamlit as st
from utils.ui_components import page_header
from pages_disabled.team_generator_page import render_team_generator_page

def render_matchday_hub_page():
    # Single source of truth: team generator now includes the MDK dropdown preview.
    render_team_generator_page()