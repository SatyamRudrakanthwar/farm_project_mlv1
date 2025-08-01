import os
import shutil
import pandas as pd
import streamlit as st
import cv2
from batch.zip_processor import process_zip_folder
from core.etl_predictor import calculate_value_loss, predict_etl_days
import plotly.express as px

def handle_zip_upload(zip_file, selected_ops, weekly_rh=None, output_dir="zip outputs"):
    st.info("🔄 Processing ZIP file...")
    pest_counts, processed_data, failed_images, image_files = process_zip_folder(zip_file, selected_ops, output_dir)

    # Show pest count table
    if pest_counts:
        st.subheader("📈 Cumulative Pest Count")
        df_summary = pd.DataFrame(pest_counts.items(), columns=["Pest", "Total Count"])
        st.dataframe(df_summary, use_container_width=True)

    # Save to Excel
    if processed_data:
        df_proc = pd.DataFrame(processed_data, columns=["Image Name", "Pest Count"])
        excel_path = os.path.join(output_dir, "Processed_Data.xlsx")
        df_proc.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button("⬇️ Download Processed Data (Excel)", data=f, file_name="Processed_Data.xlsx")

    # Download failed images
    if failed_images:
        failed_dir = os.path.join(output_dir, "Unprocessed_Images")
        os.makedirs(failed_dir, exist_ok=True)
        for path in failed_images:
            shutil.move(path, os.path.join(failed_dir, os.path.basename(path)))
        failed_zip = shutil.make_archive(failed_dir, 'zip', failed_dir)
        with open(failed_zip, "rb") as f:
            st.download_button("⬇️ Download Failed Images", data=f, file_name="Unprocessed_Images.zip")

    # Hook into ETL block
    if pest_counts:
        run_etl_block(pest_counts, weekly_rh)

# -----------------------------
def run_etl_block(pest_counts, weekly_rh):
    st.subheader("📊 ETL Calculation")
    with st.form("etl_form"):
        pest_name = st.selectbox("Choose Pest", list(pest_counts.keys()))
        count = pest_counts.get(pest_name, 1)
        if isinstance(count, str):
            try:
                count = sum(eval(count).values())
            except:
                count = 1

        N_current = st.number_input("Current Pest Count (N)", min_value=1, value=count, step=1)
        I = st.number_input("Damage Index (I)", min_value=0.0, format="%.2f")
        C = st.number_input("Pesticide Cost", min_value=0.0)
        M = st.number_input("Market Cost per Kg", min_value=0.0)

        RH = weekly_rh[0] if weekly_rh else st.number_input("RH (Humidity)", min_value=0.0)

        submit = st.form_submit_button("Submit")

    if "results" not in st.session_state:
        st.session_state["results"] = []

    if submit:
        Y, V = calculate_value_loss(I, M)
        result = V / C if C != 0 else 0
        st.session_state["results"].append([pest_name, N_current, I, Y, C, M, V, result, RH])

    if st.session_state["results"]:
        df = pd.DataFrame(st.session_state["results"], columns=[
            "Pest Name", "N", "I", "Yield Loss", "Cost", "Market Price", "Value Loss", "Value Loss/Cost", "RH"
        ])
        st.dataframe(df, use_container_width=True)

        if st.button("📈 Predict ETL"):
            df_etl, df_progress = predict_etl_days(st.session_state["results"])
            if not df_etl.empty:
                st.subheader("✅ ETL Days")
                st.dataframe(df_etl[["Pest Name", "ETL Range (Days)"]])

            if not df_progress.empty:
                st.subheader("📉 ETL Progression")
                fig = px.line(df_progress, x="Day", y="Pest Severity (%)", color="Pest Name")
                st.plotly_chart(fig, use_container_width=True)
