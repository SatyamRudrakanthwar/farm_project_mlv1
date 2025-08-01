import os
import glob
import shutil  
import zipfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import plotly.express as px
import cv2
import uuid
from PIL import Image
from core.image_processor import process_image
from core.etl_predictor import calculate_value_loss, predict_etl_days
from core.leaf_extraction import run_inference_from_pil
from core.color_analysis import (
    leaf_vein_skeleton,
    leaf_boundary_dilation,
    extract_colors_around_mask,
    extract_leaf_colors_with_locations,
    bubble_plot,
    cluster_and_mark_palette
)

def zip_analysis_ui(inputs):
    image_file = inputs.get("image_file")
    weekly_rh = inputs.get("weekly_rh")
    zip_file = inputs.get("folder_file")

    if zip_file:
        st.success("📁 ZIP file uploaded successfully!")

        with st.expander("🛠️ Select What You Want to Analyze"):
            selected_ops = {
                "leaf_extraction": st.checkbox("🍃 Leaf Extraction", value=False),
                "ETL Analysis": st.checkbox("⏳ ETL Analysis", value=False),
                "region_color": st.checkbox("🎨 Region Color Analysis", value=False),
                "overall_color": st.checkbox("🌈 Overall Colour Analysis", value=False),
                "bubble_plot": st.checkbox("📊 Bubble Plot Analysis", value=False),
                "cluster_palette": st.checkbox("🔍 Clustered Color Palette", value=False)
            }

            if st.button("Done") or st.session_state.get("zip_done", False):
                st.session_state["zip_done"] = True  # ✅ Set flag
                cumulative_counts, processed_data, failed_images, image_files = process_zip_folder(
                    folder_file=zip_file,
                    selected_ops=selected_ops,
                    weekly_rh=weekly_rh,
                    output_dir="zip outputs"
                )
    else:
        st.info("📤 Please upload a ZIP file to begin analysis.")



def process_zip_folder(folder_file, selected_ops,weekly_rh, output_dir="outputs"):
    extract_path = "temp_extracted"
    unprocessed_path = "Unprocessed_Images"
    
    shutil.rmtree(extract_path, ignore_errors=True)
    shutil.rmtree(unprocessed_path, ignore_errors=True)
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(unprocessed_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(folder_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    image_files = glob.glob(os.path.join(extract_path, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(extract_path, "**", "*.jpeg"), recursive=True) + \
                  glob.glob(os.path.join(extract_path, "**", "*.png"), recursive=True)

    st.write(f"📦 Total images found in ZIP: {len(image_files)}")

    failed_images = []
    processed_data = []
    cumulative_counts = {}
    progress_bar = st.progress(0)

    for i, img_path in enumerate(image_files):
        annotated, pests, error = process_image(img_path)

        if error:
            failed_images.append(img_path)
            shutil.move(img_path, os.path.join(unprocessed_path, os.path.basename(img_path)))
        else:
            processed_data.append([os.path.basename(img_path), str(pests)])

            for pest, count in pests.items():
                cumulative_counts[pest] = cumulative_counts.get(pest, 0) + count

            pil_img = Image.open(img_path).convert("RGB")

            # Leaf Extraction
            if selected_ops.get("leaf_extraction"):
                try:
                    extracted_paths, err = run_inference_from_pil(pil_img)
                    if not err and extracted_paths:
                        leaf_dir = os.path.join(output_dir, "leaf_extractions")
                        os.makedirs(leaf_dir, exist_ok=True)
                        for idx, leaf_img_path in enumerate(extracted_paths, 1):
                            # Open, resize and convert to RGB
                            leaf_img = Image.open(leaf_img_path).convert("RGB")
                            leaf_img = leaf_img.resize((512, 512))

                            # Save as PNG with clear name
                            base_name = os.path.splitext(os.path.basename(img_path))[0]
                            save_name = f"leaf_{idx}_{base_name}.png"
                            save_path = os.path.join(leaf_dir, save_name)
                            leaf_img.save(save_path)
                except Exception as e:
                    st.warning(f"❌ Leaf extraction failed for {img_path}: {e}")
                    
            # ✅ Only run this if region_color is selected
            if selected_ops.get("region_color"):
                st.subheader("🎨 Running Region Color Analysis on Extracted Leaf Images...")

                # Define the path to extracted leaf images
                leaf_extract_dir = os.path.join("zip outputs", "leaf_extractions")
                leaf_paths = glob.glob(os.path.join(leaf_extract_dir, "*.png"))  # or "*.jpg" based on save format

                for leaf_path in leaf_paths:
                    try:
                        vein_mask = leaf_vein_skeleton(leaf_path)
                        boundary_mask = leaf_boundary_dilation(leaf_path)

                        stats_vein, labels_vein, percs_vein, colors_vein = extract_colors_around_mask(
                            leaf_path, vein_mask, buffer_ratio=0.5
                        )
                        stats_boundary, labels_boundary, percs_boundary, colors_boundary = extract_colors_around_mask(
                            leaf_path, boundary_mask, buffer_ratio=0.1
                        )

                        # Save output
                        reg_dir = os.path.join(output_dir, "region_color")
                        os.makedirs(reg_dir, exist_ok=True)

                        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                        axs[0].bar(labels_vein, percs_vein, color=colors_vein)
                        axs[0].set_title("Vein Color Distribution")
                        axs[0].tick_params(axis='x', rotation=45)

                        axs[1].bar(labels_boundary, percs_boundary, color=colors_boundary)
                        axs[1].set_title("Boundary Color Distribution")
                        axs[1].tick_params(axis='x', rotation=45)

                        base_name = os.path.splitext(os.path.basename(leaf_path))[0]
                        chart_path = os.path.join(reg_dir, f"{base_name}_region_colors.png")
                        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)

                        # Save masks
                        cv2.imwrite(os.path.join(reg_dir, f"{base_name}_vein_mask.png"), vein_mask)
                        cv2.imwrite(os.path.join(reg_dir, f"{base_name}_boundary_mask.png"), boundary_mask)

                    except Exception as e:
                        st.warning(f"❌ Failed region color analysis for {leaf_path}: {e}")


            # ✅ Run Overall Color Analysis on all leaf images if selected
            if selected_ops.get("overall_color"):
                st.subheader("🎨 Running Overall Color Analysis on Extracted Leaf Images...")

                leaf_extract_dir = os.path.join("zip outputs", "leaf_extractions")
                overall_dir = os.path.join("zip outputs", "overall_color")
                os.makedirs(overall_dir, exist_ok=True)

                leaf_paths = glob.glob(os.path.join(leaf_extract_dir, "*.png"))

                if not leaf_paths:
                    st.warning("⚠️ No leaf images found in leaf_extractions folder.")
                else:
                    for leaf_path in leaf_paths:
                        try:
                            fig_main, _ = extract_leaf_colors_with_locations(leaf_path)

                            base_name = os.path.splitext(os.path.basename(leaf_path))[0]
                            chart_path = os.path.join(overall_dir, f"{base_name}_overall_colors.png")
                            fig_main.savefig(chart_path, dpi=150, bbox_inches="tight")
                            plt.close(fig_main)

                        except Exception as e:
                            st.warning(f"❌ Failed overall color analysis for {leaf_path}: {e}")
                            
            # Initialize session state key
            if "bubble_done" not in st.session_state:
                st.session_state.bubble_done = False

            # ✅ Run Bubble Plot Analysis if selected
            if selected_ops.get("bubble_plot") and not st.session_state.bubble_done:
                st.subheader("📊 Running Bubble Plot Analysis on Extracted Leaf Images...")

                leaf_extract_dir = os.path.join("zip outputs", "leaf_extractions")
                bubble_dir = os.path.join("zip outputs", "bubble_plots")
                os.makedirs(bubble_dir, exist_ok=True)
                os.makedirs("zip outputs", exist_ok=True)

                leaf_paths = glob.glob(os.path.join(leaf_extract_dir, "*.png"))
                failed_images = []

                for leaf_path in leaf_paths:
                    try:
                        base_name = os.path.splitext(os.path.basename(leaf_path))[0]

                        # Generate masks
                        vein_mask = leaf_vein_skeleton(leaf_path)
                        boundary_mask = leaf_boundary_dilation(leaf_path)

                        # Extract color stats
                        stats_vein, _, _, _ = extract_colors_around_mask(
                            leaf_path, vein_mask, buffer_ratio=0.5
                        )
                        stats_boundary, _, _, _ = extract_colors_around_mask(
                            leaf_path, boundary_mask, buffer_ratio=0.1
                        )

                        # Save Vein Region Bubble Plot
                        fig_vein = bubble_plot(stats_vein, title=f'{base_name} - Vein Region Bubble Plot')
                        vein_output_path = os.path.join(bubble_dir, f"{base_name}_vein_bubble.png")
                        fig_vein.savefig(vein_output_path, dpi=150, bbox_inches="tight")
                        plt.close(fig_vein)

                        # Save Boundary Region Bubble Plot
                        fig_boundary = bubble_plot(stats_boundary, title=f'{base_name} - Boundary Region Bubble Plot')
                        boundary_output_path = os.path.join(bubble_dir, f"{base_name}_boundary_bubble.png")
                        fig_boundary.savefig(boundary_output_path, dpi=150, bbox_inches="tight")
                        plt.close(fig_boundary)

                    except Exception as e:
                        failed_images.append(leaf_path)
                        st.warning(f"❌ Failed bubble plot analysis for {leaf_path}: {e}")

                # Zip all generated plots once
                zip_path = os.path.join("zip outputs", "bubble_plots.zip")
                if os.path.exists(zip_path):
                    os.remove(zip_path)

                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file in os.listdir(bubble_dir):
                        file_path = os.path.join(bubble_dir, file)
                        zipf.write(file_path, arcname=file)

                st.success("✅ Vein and Boundary bubble plots generated, saved, and zipped successfully!")

                # ✅ Read zip once and show download button
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()

                import uuid
                st.download_button(
                    label="📥 Download Bubble Plots Zip",
                    data=zip_bytes,
                    file_name="bubble_plots.zip",
                    mime="application/zip",
                    key=f"download_bubble_zip_{uuid.uuid4()}"
                )

                # Mark bubble plot as completed
                st.session_state.bubble_done = True

            # Clustered Color Palette Analysis
            if selected_ops.get("cluster_palette") and not st.session_state.get("cluster_palette_done", False):
                st.subheader("🎨 Running Clustered Color Palette Analysis on Extracted Leaf Images...")
                leaf_extract_dir = os.path.join(output_dir, "leaf_extractions")
                cluster_dir = os.path.join(output_dir, "clustered_palette")
                os.makedirs(leaf_extract_dir, exist_ok=True)
                os.makedirs(cluster_dir, exist_ok=True)
                mask_output_dir = os.path.join(cluster_dir, "region_masks")
                os.makedirs(mask_output_dir, exist_ok=True)

                # Get extracted leaf image paths
                leaf_paths = glob.glob(os.path.join(leaf_extract_dir, "*.png"))

                # Fallback if no leaf images found
                if not leaf_paths:
                    st.info("⚠️ No extracted leaf images found. Running fallback extraction...")
                    original_image_paths = glob.glob(os.path.join(extract_path, "**", "*.jpg"), recursive=True) + \
                                        glob.glob(os.path.join(extract_path, "**", "*.jpeg"), recursive=True) + \
                                        glob.glob(os.path.join(extract_path, "**", "*.png"), recursive=True)
                    for img_path in original_image_paths:
                        try:
                            pil_img = Image.open(img_path).convert("RGB")
                            extracted_paths, err = run_inference_from_pil(pil_img)
                            if not err and extracted_paths:
                                for idx, leaf_img_path in enumerate(extracted_paths, 1):
                                    leaf_img = Image.open(leaf_img_path).convert("RGB").resize((512, 512))
                                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                                    save_name = f"leaf_{idx}_{base_name}.png"
                                    save_path = os.path.join(leaf_extract_dir, save_name)
                                    leaf_img.save(save_path, format="PNG")
                            else:
                                st.warning(f"⚠️ No leaves extracted from {img_path}")
                        except Exception as e:
                            st.warning(f"⚠️ Fallback extraction failed for {img_path}: {e}")

                    # Refresh leaf paths
                    leaf_paths = glob.glob(os.path.join(leaf_extract_dir, "*.png"))

                total = len(leaf_paths)
                if total == 0:
                    st.warning("❌ No leaf images available for clustered palette analysis.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i, leaf_path in enumerate(leaf_paths):
                        try:
                            base_name = os.path.splitext(os.path.basename(leaf_path))[0]
                            status_text.text(f"🔄 Processing: {base_name} ({i + 1}/{total})")

                            # Generate masks
                            vein_mask = leaf_vein_skeleton(leaf_path)
                            boundary_mask = leaf_boundary_dilation(leaf_path)

                            # Save masks for debugging
                            vein_mask_path = os.path.join(mask_output_dir, f"{base_name}_vein_mask.png")
                            boundary_mask_path = os.path.join(mask_output_dir, f"{base_name}_boundary_mask.png")
                            cv2.imwrite(vein_mask_path, (vein_mask * 255).astype("uint8"))
                            cv2.imwrite(boundary_mask_path, (boundary_mask * 255).astype("uint8"))

                            # Extract colors
                            stats_vein, _, _, colors_vein = extract_colors_around_mask(leaf_path, vein_mask, buffer_ratio=0.5)
                            stats_boundary, _, _, colors_boundary = extract_colors_around_mask(leaf_path, boundary_mask, buffer_ratio=0.1)

                            # Convert color arrays to RGB tuples
                            vein_colors = [tuple((color * 255).astype(int)) for color in colors_vein]
                            boundary_colors = [tuple((color * 255).astype(int)) for color in colors_boundary]

                            # Output for clustered palette
                            palette_path = os.path.join(cluster_dir, f"{base_name}_clustered_palette.png")
                            cluster_and_mark_palette(
                                vein_colors=vein_colors,
                                boundary_colors=boundary_colors,
                                num_clusters=5,
                                output_path=palette_path
                            )

                            # Display the clustered palette
                            st.image(palette_path, caption="Clustered Color Palette", use_container_width=True)

                        except Exception as e:
                            st.warning(f"⚠️ Clustered palette failed for {leaf_path}: {e}")
                        progress_bar.progress((i + 1) / total)

                    status_text.text("")
                    st.success(f"✅ Clustered color palettes and masks saved for {total} leaf images.")
                    st.session_state.cluster_palette_done = True

                    # Create ZIP containing clustered palettes and region masks
                    cluster_zip_path = os.path.join(output_dir, "Clustered_Palette_Outputs.zip")
                    with zipfile.ZipFile(cluster_zip_path, 'w') as zipf:
                        for folder in [cluster_dir, mask_output_dir]:
                            for root, _, files in os.walk(folder):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, start=cluster_dir)
                                    zipf.write(file_path, arcname=arcname)

                    # Download button
                    with open(cluster_zip_path, "rb") as f:
                        st.download_button(
                            label="📦 Download Clustered Palette & Mask Outputs",
                            data=f,
                            file_name="Clustered_Palette_Outputs.zip",
                            mime="application/zip"
                        )




    # ✅ Save cumulative pest counts persistently
    if cumulative_counts:
        st.session_state["cumulative_counts"] = cumulative_counts
        st.subheader("📈 Cumulative Pest Counts from ZIP")
        cumulative_df = pd.DataFrame(list(cumulative_counts.items()), columns=["Pest", "Total Count"])
        st.dataframe(cumulative_df, use_container_width=True)

    # ✅ Run ETL Analysis block only when selected
    if selected_ops.get("ETL Analysis") and st.session_state.get("cumulative_counts"):
        st.subheader("⏳ Running ETL From Calculated Pests...")
        st.markdown("---")
        st.subheader("📈 Pest Economic Threshold Level (ETL) Prediction")
        st.markdown("Complete pest counting and enter crop and pest damage details for ETL analysis")

        # ✅ Use session_state to retain RH across reruns
        if "weekly_rh" not in st.session_state:
            st.session_state["weekly_rh"] = weekly_rh if 'weekly_rh' in locals() else []

        with st.form("pest_input_form", clear_on_submit=False):
            st.subheader("Enter Pest and Crop Details")

            pest_keys = list(st.session_state["cumulative_counts"].keys())
            pest_name = st.selectbox("Pest Name (Detected)", pest_keys)
            pest_count = st.session_state["cumulative_counts"].get(pest_name, 1)

            N_current = st.number_input("Number of Pests (Current)", min_value=1, value=pest_count, step=1)
            I = st.number_input("I (Damage index)", min_value=0.0, format="%.3f")
            pesticides_cost = st.number_input("Pesticide Cost per Acre", min_value=0.0, format="%.2f")
            market_cost_per_kg = st.number_input("Market Cost per Kg", min_value=0.0, format="%.2f")

            if st.session_state["weekly_rh"]:
                fev_con = st.session_state["weekly_rh"][0]
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

    # ✅ Show results even after rerun or submission
    if st.session_state.get("results"):
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
                    line_shape="spline"
                )
                fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
                st.plotly_chart(fig, use_container_width=True)



    if failed_images:
        st.warning(f"⚠️ {len(failed_images)} images failed and moved to 'Unprocessed_Images'.")
        failed_zip = os.path.join(output_dir, "Unprocessed_Images.zip")
        shutil.make_archive(failed_zip.replace(".zip", ""), 'zip', unprocessed_path)
        with open(failed_zip, "rb") as f:
            st.download_button("⬇️ Download Unprocessed Images", data=f, file_name="Unprocessed_Images.zip", mime="application/zip")

 # ✅ If any images were processed successfully, create a ZIP of processed images
    if processed_data:
        st.success(f"✅ Successfully processed {len(image_files)} images.")
        
        df = pd.DataFrame(processed_data, columns=["Image Name", "Pest Count"])
        st.dataframe(df)

        # Define paths
        zip_base = os.path.join(output_dir, "Processed_Images")  # Without .zip
        success_zip = f"{zip_base}.zip"  # Final .zip file path

        # Create ZIP of extract_path folder (processed images)
        shutil.make_archive(zip_base, 'zip', extract_path)

        # Show download button only if file exists and is not empty
        if os.path.exists(success_zip) and os.path.getsize(success_zip) > 0:
            with open(success_zip, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Images",
                    data=f,
                    file_name="Processed_Images.zip",
                    mime="application/zip"
                )
        else:
            st.warning("❌ Processed ZIP not found or is empty.")

            
    # Excel of all processed images and their pests
    if processed_data:
        df = pd.DataFrame(processed_data, columns=["Image Name", "Pest Count"])
        excel_path = os.path.join(output_dir, "Processed_Data.xlsx")
        df.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button("⬇️ Download Processed Data (Excel)", data=f, file_name="Processed_Data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Generate and zip annotated images with highest pest counts
        df["Pest Count"] = df["Pest Count"].apply(eval)
        max_images = {}
        image_path_map = {os.path.basename(path): path for path in image_files}
        for _, row in df.iterrows():
            for pest, count in row["Pest Count"].items():
                if pest not in max_images or count > max_images[pest][1]:
                    max_images[pest] = (row["Image Name"], count)

        annotated_dir = os.path.join(output_dir, "highest_annotated")
        shutil.rmtree(annotated_dir, ignore_errors=True)
        os.makedirs(annotated_dir, exist_ok=True)

        for pest, (img_name, count) in max_images.items():
            full_img_path = image_path_map.get(img_name)
            if full_img_path and os.path.exists(full_img_path):
                annotated, _, error = process_image(full_img_path)
                if not error:
                    out_path = os.path.join(annotated_dir, f"{pest}_{img_name}")
                    cv2.imwrite(out_path, annotated)

        zip_path = os.path.join(output_dir, "Highest_Annotated_Images.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', annotated_dir)
        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download All Annotated Highest Count Images (ZIP)", data=f, file_name="Highest_Annotated_Images.zip", mime="application/zip")

    # Feature zip downloads
    for feature, folder in {
        "Leaf Extractions": "leaf_extractions",
        "Region Color Analysis": "region_color",
        "Overall Color Analysis": "overall_color"
    }.items():
        feat_dir = os.path.join(output_dir, folder)
        if os.path.exists(feat_dir) and os.listdir(feat_dir):
            zip_feat_path = os.path.join(output_dir, f"{folder}.zip")
            shutil.make_archive(zip_feat_path.replace(".zip", ""), 'zip', feat_dir)
            with open(zip_feat_path, "rb") as f:
                st.download_button(
                    f"⬇️ Download {feature} Results (ZIP)",
                    data=f,
                    file_name=os.path.basename(zip_feat_path),
                    mime="application/zip"
                )

    # ✅ Return both summary count (for ETL) and processed data (optional download)
    return cumulative_counts, processed_data, failed_images, image_files


