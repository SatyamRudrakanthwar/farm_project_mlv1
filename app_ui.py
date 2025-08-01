import streamlit as st
import pandas as pd

def render_ui():
    # File upload section
    col1, col2 = st.columns([1, 1])
    
    image_file = col1.file_uploader("📸 Select an Image", type=["jpg", "jpeg", "png"], key="image")
    folder_file = col2.file_uploader("📁 Select a Folder (Zip)", type=["zip"], key="folder")

    st.markdown("---")
    st.subheader("📂 Upload CSV for Abiotic Values")
    rh_file = st.file_uploader("Upload abioticvalue.csv file", type=["csv"], key="rh_csv")

    weekly_rh = []

    if rh_file:
        weather_df = pd.read_csv(rh_file)

        if 'Time' in weather_df.columns:
            weather_df['Time'] = pd.to_datetime(weather_df['Time'])
            weather_df.set_index('Time', inplace=True)

            if 'RH' in weather_df.columns:
                weekly_rh = weather_df['RH'].resample('7D').mean().tolist()
            elif 'temperature_2m' in weather_df.columns:
                st.warning("⚠️ Using 'temperature_2m' as fallback.")
                weekly_rh = weather_df['temperature_2m'].resample('7D').max().tolist()
            else:
                st.error("❌ No valid RH or temperature_2m column found.")
        else:
            st.error("❌ 'Time' column not found in uploaded file.")

    return {
        "image_file": image_file,
        "folder_file": folder_file,
        "weekly_rh": weekly_rh
    }
