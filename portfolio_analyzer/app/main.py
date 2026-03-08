import streamlit as st
from portfolio_analyzer.app.components.social import render_social_links

# Global config must be set first
st.set_page_config(
    page_title="Analytics",
    layout="wide"
)

# Global CSS: Rectangular buttons & Consistent Top Spacing
st.markdown("""
<style>
div.stButton > button:first-child, div.stDownloadButton > button:first-child {
    border-radius: 0px;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
h1 {
    font-weight: 500 !important;
    margin-top: -0.5rem !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] hr {
    margin-top: -0.25rem;
    margin-bottom: 0.5rem;
}
[data-testid="stSidebar"] h2 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# Define pages using existing files
# Note: st.Page takes the file path relative to the main script
pages = [
    st.Page("pages/portfolio_builder.py", 
            title="Builder", 
            icon=":material/widgets:"
    )
]

# Create navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()





