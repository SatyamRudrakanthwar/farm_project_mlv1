import streamlit as st
import time
import os
import pandas as pd
import cv2
from PIL import Image
from core.image_processor import process_image
from core.leaf_extraction import run_inference_from_pil
from core.etl_predictor import calculate_value_loss, predict_etl_days
from core.color_analysis import (
                    leaf_vein_skeleton, extract_colors_around_mask,
                    leaf_boundary_dilation,bubble_plot, cluster_and_mark_palette
                )
from core.color_analysis import extract_leaf_colors_with_locations
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt 

def run_analysis(inputs):
    image_file = inputs.get("image_file")
    weekly_rh = inputs.get("weekly_rh")
    zip_file = inputs.get("folder_file")

    # ✅ ZIP analysis is handled through buttons in zip_processor.py, so skip here
    if zip_file:
        st.info("📁 ZIP file uploaded. Please use the buttons above to run the desired analysis.")
        return  # Skip further logic here if ZIP is used
    
    # ✅ Continue with image if no ZIP
    if image_file:
        image_changed = (
            "image_name" not in st.session_state or
            st.session_state["image_name"] != image_file.name
        )
        if image_changed:
            pil_image = Image.open(image_file).convert("RGB")
            timestamp = int(time.time())
            image_path = f"temp_{timestamp}_{image_file.name}"
            pil_image.save(image_path)

            with st.spinner("⏳ Processing Image..."):
                annotated_image, pest_counts, error = process_image(image_path)

                if not error:
                    st.session_state["annotated_image"] = annotated_image
                    st.session_state["pest_counts"] = pest_counts
                    st.session_state["image_name"] = image_file.name
                    st.session_state["image_path"] = image_path
        else:
            annotated_image = st.session_state.get("annotated_image")
            pest_counts = st.session_state.get("pest_counts", {})
    elif not zip_file:
        st.warning("Please upload an image or ZIP to proceed.")
        return

    pest_counts = st.session_state.get("pest_counts", {})

    st.markdown("---")
    st.markdown("#### 📌 Select an Analysis Type")
    cards = {
        "Pest Detection Name": "🐛Pest Detected: ",
        "Pest Counting for ETL analysis": "📊 Pest Count: ",
        "Leaf Extraction": "🍃Leaf Extraction: Extracted leaf details will appear here.",
        "Color Analysis": "🎨Color Analysis: Insights about color will be shown.",
        "Overall Colour Analysis": "Analysis of whole Image will be shown. ",
        "bubble plot": "🫧Visual representation of color distribution.",
        "Cluster and Mark Palette": "❄️Clustered Color Palette: Visualize dominant colors."
    }

    col1, _, col3 = st.columns([1, 0.05, 2])
    with col1:
        for card in cards:
            if st.button(card, key=card):
                st.session_state["selected_card"] = card

    with col3:
        selected_card = st.session_state.get("selected_card")
        if selected_card:
            output_text = cards[selected_card]
            if selected_card == "Pest Detection Name" and pest_counts:
                output_text += ", ".join(pest_counts.keys())
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image")
            elif selected_card == "Pest Counting for ETL analysis" and pest_counts:
                output_text += str(pest_counts)

            st.markdown(
                f"<div style='border:2px solid; border-radius:10px; padding:15px;"
                "background-color:white; box-shadow:2px 2px 10px rgba(0,0,0,0.1);"
                "font-size:16px; color:black;'>"
                f"🔍 {selected_card} <br><br>{output_text}</div>", 
                unsafe_allow_html=True
            )

            if selected_card == "Leaf Extraction":
                pil_image = Image.open(image_file).convert("RGB")
                with st.spinner("🍂 Extracting leaves..."):
                    extracted_paths, err = run_inference_from_pil(pil_image)

                    if err:
                        st.error(err)
                    else:
                        st.success("Leaf(s) extracted successfully:")

                        # Directly display the originally saved images
                        for path in extracted_paths:
                            st.image(path, caption=os.path.basename(path))


            elif selected_card == "Color Analysis":
                st.subheader("Leaf Color Analysis")

                input_dir = os.path.join("data", "outputs")
                color_output_dir = os.path.join("data", "colour_outputs")
                bar_plot_dir = os.path.join(color_output_dir, "bar_plots")
                os.makedirs(bar_plot_dir, exist_ok=True)

                output_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

                if not output_files:
                    st.warning("⚠️ No extracted leaf images found in 'outputs' folder.")
                else:
                    for file in output_files:
                        path = os.path.join(input_dir, file)
                        st.markdown(f"### 📄 Analyzing: `{file}`")

                        image_bgr = cv2.imread(path)
                        if image_bgr is None:
                            st.warning(f"⚠️ Could not read {file}")
                            continue

                        image_bgr = cv2.resize(image_bgr, (512, 512))
                        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Extracted Leaf Image")

                        # Run analysis
                        vein_mask = leaf_vein_skeleton(path)
                        boundary_mask = leaf_boundary_dilation(path)

                        st.image(vein_mask, caption="🩸 Vein Mask", channels="GRAY")
                        st.image(boundary_mask, caption="🧭 Boundary Mask", channels="GRAY")

                        # Save masks
                        base_name = os.path.splitext(file)[0]
                        vein_mask_path = os.path.join(color_output_dir, f"{base_name}_vein_mask.png")
                        boundary_mask_path = os.path.join(color_output_dir, f"{base_name}_boundary_mask.png")
                        cv2.imwrite(vein_mask_path, vein_mask)
                        cv2.imwrite(boundary_mask_path, boundary_mask)

                        # Color analysis
                        stats_veins, labels_veins, percs_veins, colors_veins = extract_colors_around_mask(
                            path, vein_mask, buffer_ratio=0.5
                        )
                        stats_boundary, labels_boundary, percs_boundary, colors_boundary = extract_colors_around_mask(
                            path, boundary_mask, buffer_ratio=0.1
                        )

                        # Plot bar charts
                        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                        axs[0].bar(labels_veins, percs_veins, color=colors_veins)
                        axs[0].set_title("Vein Color Distribution")
                        axs[0].tick_params(axis='x', rotation=45)

                        axs[1].bar(labels_boundary, percs_boundary, color=colors_boundary)
                        axs[1].set_title("Boundary Color Distribution")
                        axs[1].tick_params(axis='x', rotation=45)

                        st.pyplot(fig)

                        # ✅ Save bar plot in subfolder
                        bar_plot_path = os.path.join(bar_plot_dir, f"{base_name}_color_distribution.png")
                        fig.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)

                        st.success(f"✅ Bar plot saved to: `{os.path.basename(bar_plot_path)}`")




            elif selected_card == "Overall Colour Analysis":

                st.subheader("🔍 Running Overall Colour Analysis...")
                if "image_path" in st.session_state:
                    save_dir = os.path.join("data", "overall_colour_output")
                    with st.spinner("Extracting dominant color regions..."):
                        fig_main, region_figs = extract_leaf_colors_with_locations(
                            st.session_state["image_path"],
                            save_dir=save_dir
                        )

                    st.success("✅ Color analysis completed successfully!")

                    # Display main bar chart
                    st.subheader("🌈 Dominant Colors")
                    st.pyplot(fig_main)

                    # Display color region overlays
                    st.subheader("📍 Regions for Each Dominant Color")
                    for fig in region_figs:
                        st.pyplot(fig)
                else:
                    st.warning("⚠️ No image path found. Please upload and process an image first.")

            elif selected_card == "bubble plot":
                st.subheader("🫧 Bubble Plot of Color Distribution")
                input_dir = os.path.join("data", "outputs")
                bubble_output_dir = os.path.join("data", "Bubble_Plots")
                os.makedirs(bubble_output_dir, exist_ok=True)

                image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

                if not image_files:
                    st.warning("⚠️ No image files found in the extracted leaf folder.")
                else:
                    for img_file in image_files:
                        image_path = os.path.join(input_dir, img_file)
                        base_name = os.path.splitext(img_file)[0]

                        # Generate masks
                        vein_mask = leaf_vein_skeleton(image_path)
                        boundary_mask = leaf_boundary_dilation(image_path)

                        # Extract color stats
                        stats_veins, *_ = extract_colors_around_mask(image_path, vein_mask, buffer_ratio=0.5)
                        stats_boundary, *_ = extract_colors_around_mask(image_path, boundary_mask, buffer_ratio=0.1)

                        # Output paths
                        vein_plot_path = os.path.join(bubble_output_dir, f"{base_name}_vein_bubble.png")
                        boundary_plot_path = os.path.join(bubble_output_dir, f"{base_name}_boundary_bubble.png")

                        # Save bubble plots
                        bubble_plot(stats_veins, title=f"{base_name} - Vein Region", save_path=vein_plot_path)
                        bubble_plot(stats_boundary, title=f"{base_name} - Boundary Region", save_path=boundary_plot_path)

                        # Display in UI
                        st.markdown(f"### Vein Bubble Plot - `{base_name}`")
                        st.image(vein_plot_path, caption="Vein Region Bubble Plot")

                        st.markdown(f"### Boundary Bubble Plot - `{base_name}`")
                        st.image(boundary_plot_path, caption="Boundary Region Bubble Plot")

                    st.success("✅ All bubble plots generated and saved successfully!")


            elif selected_card == "Cluster and Mark Palette":
                st.subheader("🎨 Cluster and Mark Color Palette")
                input_dir = os.path.join("data", "outputs")
                palette_output_dir = os.path.join("data", "Palette_Outputs")
                os.makedirs(palette_output_dir, exist_ok=True)

                image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

                if not image_files:
                    st.warning("⚠️ No image files found in the extracted leaf folder.")
                else:
                    for img_file in image_files:
                        image_path = os.path.join(input_dir, img_file)
                        base_name = os.path.splitext(img_file)[0]

                        # Generate masks
                        vein_mask = leaf_vein_skeleton(image_path)
                        boundary_mask = leaf_boundary_dilation(image_path)

                        # Extract color stats
                        stats_veins, *_ = extract_colors_around_mask(image_path, vein_mask, buffer_ratio=0.5)
                        stats_boundary, *_ = extract_colors_around_mask(image_path, boundary_mask, buffer_ratio=0.1)

                        # Build color lists
                        vein_colors = list(stats_veins.keys())
                        boundary_colors = list(stats_boundary.keys())

                        # Output path for color palette
                        palette_path = os.path.join(palette_output_dir, f"{base_name}_color_palette.png")

                        # Generate and save clustered palette
                        cluster_and_mark_palette(
                            vein_colors=vein_colors,
                            boundary_colors=boundary_colors,
                            num_clusters=5,
                            output_path=palette_path
                        )

                        # Display in UI
                        st.markdown(f"### Clustered Palette - `{base_name}`")
                        st.image(palette_path, caption="Clustered Color Palette")

                    st.success("✅ All clustered color palettes generated and saved successfully!")


    if pest_counts:
        st.markdown("---")
        st.subheader("📈 Pest Economic Threshold Level (ETL) Prediction")
        st.markdown("Complete pest counting and enter crop and pest damage details for ETL analysis") 
        with st.form("pest_input_form"):
            st.subheader("Enter Pest and Crop Details")
            pest_name = st.selectbox("Pest Name (Detected)", list(pest_counts.keys()))
            raw_val = pest_counts[pest_name]
            if isinstance(raw_val, str):
                try:
                    val_dict = eval(raw_val)
                    pest_count = val_dict.get(pest_name, sum(val_dict.values()))
                except:
                    pest_count = 1
            else:
                pest_count = raw_val
            N_current = st.number_input("Number of Pests (Current)", min_value=1, value=pest_count, step=1)
            I = st.number_input("I (Damage index)", min_value=0.0, format="%.3f")
            pesticides_cost = st.number_input("Pesticide Cost per Acre", min_value=0.0, format="%.2f")
            market_cost_per_kg = st.number_input("Market Cost per Kg", min_value=0.0, format="%.2f")
            if weekly_rh:
                fev_con = weekly_rh[0]
                st.info(f"RH Value (Week 1) from CSV: {fev_con:.2f}")
            else:
                fev_con = st.number_input("RH (Relative Humidity)", min_value=0.0, format="%.2f")
            submitted = st.form_submit_button("Submit Pest Data")

        if "results" not in st.session_state:
            st.session_state["results"] = []

        if submitted:
            yield_lost, value_loss = calculate_value_loss(I, market_cost_per_kg)
            result = value_loss / pesticides_cost if pesticides_cost != 0 else 0
            st.success(f"Initial Result for {pest_name}: {result:.2f}")
            st.session_state["results"].append([
                pest_name, N_current, I, yield_lost, pesticides_cost,
                market_cost_per_kg, value_loss, result, fev_con
            ])

        if st.session_state["results"]:
            df = pd.DataFrame(st.session_state["results"], columns=[
                "Pest Name", "N (Current)", "I (Damage Index)", "Yield Loss (kg)",
                "Pesticide Cost", "Market Cost/kg", "Value Loss", "Value Loss / Cost", "RH"
            ])
            st.subheader("📊 Entered Pest Data")
            st.dataframe(df, use_container_width=True)

            if st.button("🗕 Predict ETL Days"):
                df_etl, df_progress = predict_etl_days(st.session_state["results"])

                if not df_etl.empty:
                    st.subheader("✅ Estimated ETL Days (±10% Range)")
                    st.dataframe(df_etl[["Pest Name", "ETL Range (Days)"]], use_container_width=True)

                if not df_progress.empty:
                    st.subheader("📉 Full ETL Progression Data")
                    st.dataframe(df_progress, use_container_width=True)

                    st.subheader("📈 Pest Severity Progression Over Time")
                    fig = px.line(
                        df_progress,
                        x="Day",
                        y="Pest Severity (%)",
                        color="Pest Name",
                        markers=True,
                        line_shape="spline",
                        labels={
                            "Day": "Days",
                            "Pest Severity (%)": "Pest Severity (%)",
                            "Pest Name": "Pest"
                        }
                    )
                    fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
                    st.plotly_chart(fig, use_container_width=True)
