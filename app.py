"""Streamlit entry point; cache saved resources and route the seven app sections."""
import streamlit as st
from src.model_loader import load_artifacts, ArtifactError
from src.project_views import landing
from src.ui import DISCLAIMER

st.set_page_config(page_title='NBFC Credit Risk Intelligence Platform',page_icon='📊',layout='wide',initial_sidebar_state='auto')
pages = {
    'Platform':[st.Page(landing,title='Home',default=True,url_path='home')],
    'Portfolio':[
        st.Page('pages/1_Executive_Overview.py',title='Executive Overview',url_path='executive'),
        st.Page('pages/2_Portfolio_Analytics.py',title='Portfolio Analytics',url_path='portfolio'),
        st.Page('pages/6_Portfolio_Explorer.py',title='Portfolio Explorer',url_path='explorer')],
    'Risk intelligence':[
        st.Page('pages/3_Borrower_Risk_Assessment.py',title='Borrower Risk Assessment',url_path='assessment'),
        st.Page('pages/4_Risk_Explainability.py',title='Risk Explainability',url_path='explainability'),
        st.Page('pages/5_Model_Performance.py',title='Model Performance',url_path='performance')],
    'Project':[st.Page('pages/7_About_Project.py',title='About Project',url_path='about')]
}
page=st.navigation(pages)
with st.sidebar:
    st.divider()
    try:
        artifacts=load_artifacts()
        st.markdown(f'**{artifacts.metadata["model_family"]} origination model**')
        st.caption(f'{len(artifacts.metadata["features"])} features · threshold {artifacts.metadata["selected_threshold"]:.0%}')
        st.caption('Amounts: source dollars ($); no currency conversion.')
    except ArtifactError as exc:
        st.error(str(exc))
    st.caption(DISCLAIMER)
page.run()
