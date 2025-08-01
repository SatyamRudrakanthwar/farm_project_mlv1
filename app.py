import streamlit as st
from app_ui import render_ui
from pest_analysis import run_analysis  
from batch.zip_processor import zip_analysis_ui

# ✅ Streamlit Page Config
st.set_page_config(page_title="AgriSavant", layout="wide")

# ✅ App Title
st.title("🌻 AgriSavant")
st.markdown("Upload an image or a zip file to begin your pest detection and ETL analysis.")

# ✅ Step 1: Render UI and collect inputs (image / zip / abiotic CSV)
inputs = render_ui()

# ✅ Step 2: Decide what to run based on user upload
if inputs.get("folder_file"):   # ZIP uploaded
    zip_analysis_ui(inputs)

elif inputs.get("image_file"):  # Single image uploaded
    run_analysis(inputs)

else:
    st.info("📤 Please upload an image or zip file to start analysis.")
