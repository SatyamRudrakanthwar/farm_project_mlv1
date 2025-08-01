import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
import webcolors
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from PIL import Image, ImageDraw
import os
import zipfile
import colorsys
from PIL import ImageFont, ImageDraw

# ✅ Match RGB color to nearest CSS3 color name
def closest_css3_color(requested_color: Tuple[int, int, int]) -> str:
    min_dist = float('inf')
    closest_name = None
    for name, hex_val in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
        dist = np.linalg.norm(np.array([r_c, g_c, b_c]) - np.array(requested_color))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


# ✅ Generate vein skeleton from grayscale image
def leaf_vein_skeleton(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Use adaptive threshold instead of fixed Canny thresholds
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    edges = cv2.Canny(enhanced, 30, 100) #reduced to 30 from 50 , 100 from 150(reduce thresholds for finer details)  
    
    # Morphological operations to clean up edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise components
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Filter out small noise
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    _, binary = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    return skeleton


# ✅ Generate leaf boundary mask using dilation
def leaf_boundary_dilation(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    boundary = cv2.subtract(dilated, binary)
    return boundary

# Color validation function to filter outliers
def is_valid_leaf_color(rgb_color: np.ndarray, color_type: str = "general") -> bool:
    r, g, b = rgb_color
    
    if r > 240 and g > 240 and b > 240: # filter out nearly white
        return False
    if r < 20 and g < 20 and b < 20: # Filter out near-black pixels
        return False
    
    total_intensity = r + g + b
    if total_intensity == 0:
        return False
    
    green_ratio = g / total_intensity
    
    if color_type == "vein":
        # Veins can be darker but should still have some green
        return green_ratio > 0.25 and g > 30
    elif color_type == "boundary":
        # Boundaries should have reasonable green content  
        return green_ratio > 0.28 and g > 40
    else:
        # General leaf color validation
        return green_ratio > 0.25 and g > 35


# outlier detection
def remove_color_outliers(colors: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    
    if len(colors) <= 3:
        return colors
    
    median = np.median(colors, axis=0)
    mad = np.median(np.abs(colors - median), axis=0)
    mad = np.where(mad == 0, 1, mad)

    modified_z_scores = 0.6745 * (colors - median) / mad
    
    # Keep colors where all channels are within threshold
    outlier_mask = np.all(np.abs(modified_z_scores) < threshold, axis=1)
    
    return colors[outlier_mask]

# ✅ Extract dominant colors around a binary mask
def extract_colors_around_mask(image_path: str, mask: np.ndarray, buffer_ratio=0.15, num_colors=8, color_type="general"):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = mask.shape[:2]
    diag = int(np.sqrt(h ** 2 + w ** 2))
    buffer_pixels = max(2, int(diag * buffer_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_pixels, buffer_pixels))
    region = cv2.dilate(mask, kernel, iterations=1)

    masked_pixels = image_rgb[region == 255].reshape(-1, 3)
    
    if len(masked_pixels) == 0:
        return {}, [], [], []
    
    # Filter valid pixels
    valid_pixels = []
    for pixel in masked_pixels:
        if is_valid_leaf_color(pixel, color_type):
            valid_pixels.append(pixel)
    
    valid_pixels = np.array(valid_pixels)
    
    if len(valid_pixels) == 0:
        print(f"Warning: No valid {color_type} colors found")
        return {}, [], [], []
    
    # Remove outliers from valid pixels
    filtered_pixels = remove_color_outliers(valid_pixels, threshold=2.5)
    
    if len(filtered_pixels) == 0:
        print(f"Warning: All {color_type} colors were filtered out as outliers")
        filtered_pixels = valid_pixels  # Fallback to valid pixels
    
    # Use filtered pixels for clustering instead of original masked_pixels
    kmeans = KMeans(n_clusters=min(num_colors, len(filtered_pixels)), random_state=42)
    labels = kmeans.fit_predict(filtered_pixels)

    color_stats: Dict[str, Dict] = {}
    total_pixels = len(filtered_pixels)  # Use filtered pixels count

    for i in range(kmeans.n_clusters):
        idx = np.where(labels == i)[0]  # These indices now match filtered_pixels
        if len(idx) == 0:
            continue
            
        cluster_colors = filtered_pixels[idx]  # Now this indexing works correctly
        mean_color = np.mean(cluster_colors, axis=0).astype(int)
        
        # Final validation of cluster center
        if not is_valid_leaf_color(mean_color, color_type):
            print(f"Skipping invalid cluster color: {mean_color}")
            continue
        
        r, g, b = mean_color
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        label = f"{hex_code}\n({r},{g},{b})"
        pixel_count = len(idx)
        color_stats[label] = {
            "count": pixel_count,
            "rgb": mean_color,
            "percentage": (pixel_count / total_pixels) * 100
        }

    sorted_labels = sorted(color_stats.items(), key=lambda x: x[1]["percentage"], reverse=True)
    labels = [label for label, _ in sorted_labels]
    percentages = [color_stats[label]["percentage"] for label in labels]
    colors = [np.array(color_stats[label]["rgb"]) / 255.0 for label in labels]

    return color_stats, labels, percentages, colors


# ✅ Show a pie chart of color distribution
def visualize_results(labels, percentages, colors):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title("Dominant Colors Around Region")
    plt.tight_layout()
    plt.show()


def extract_leaf_colors_with_locations(image_path, num_colors=5, save_dir=None):
    image_bgr = cv2.imread(image_path)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    image_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    resized_rgb = cv2.resize(image_rgb, (200, 200))
    resized_lab = cv2.resize(lab_eq, (200, 200))
    h, w, _ = resized_rgb.shape

    ab_pixels = resized_lab[:, :, 1:].reshape((-1, 2))
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    labels = kmeans.fit_predict(ab_pixels)
    label_map = labels.reshape((h, w))

    rgb_pixels = resized_rgb.reshape((-1, 3))
    colors, counts = [], []
    for i in range(num_colors):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        color = np.mean(rgb_pixels[idx], axis=0).astype(int)
        colors.append(color)
        counts.append(len(idx))

    sorted_idx = np.argsort(counts)[::-1]
    colors = np.array(colors)[sorted_idx]
    counts = np.array(counts)[sorted_idx]

    # Bar chart of dominant colors
    fig_main, ax = plt.subplots(figsize=(5, 2))
    for i, color in enumerate(colors):
        ax.bar(i, counts[i], color=np.array(color) / 255)
        ax.text(i, counts[i] + 100, f'{color}', ha='center', fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel("Pixel Count")
    ax.set_title("Dominant Colors in leaf image")
    plt.show()

    # Save bar chart
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        fig_main_path = os.path.join(save_dir, f"{base_name}_color_bar_chart.png")
        fig_main.savefig(fig_main_path, bbox_inches='tight', dpi=150)



    # Region overlays
    region_figs = []
    for idx, color in enumerate(colors):
        mask = (label_map == sorted_idx[idx]).astype(np.uint8) * 255
        overlay = np.zeros_like(resized_rgb)
        overlay[:, :] = color
        masked_color = cv2.bitwise_and(overlay, overlay, mask=mask)

        fig2, ax2 = plt.subplots()
        ax2.imshow(masked_color)
        ax2.set_title(f"Region for Color {tuple(color)}")
        ax2.axis('off')
        region_figs.append(fig2)

        if save_dir:
            region_path = os.path.join(save_dir, f"{base_name}region{idx + 1}.png")
            fig2.savefig(region_path, bbox_inches='tight', dpi=150)

    return fig_main, region_figs


# ✅ Bubble plot for color distribution
def bubble_plot(color_stats, title="Color Bubble Plot", save_path="bubble_plot.png", zip_path="bubble_plot.zip"):

    if not color_stats:
        print("Skipping bubble plot — color_stats is empty.")
        return

    rgb_values = [v['rgb'] for v in color_stats.values()]
    sizes = [v['percentage'] * 100 for v in color_stats.values()]
    labels = list(color_stats.keys())

    x = list(range(len(rgb_values)))
    y = [1] * len(rgb_values)
    normalized_colors = [np.array(rgb) / 255 for rgb in rgb_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=sizes, color=normalized_colors, edgecolors='black')

    for i, label in enumerate(labels):
        ax.text(x[i], y[i] + 0.03, label.split("\n")[1], ha='center', fontsize=8)

    for i, rgb in enumerate(rgb_values):
        r, g, b = rgb
        total_intensity = r + g + b
        green_percentage = (g / total_intensity) * 100 if total_intensity > 0 else 0
        ax.text(x[i], y[i] - 0.02, f"Green: {green_percentage:.1f}%",
                ha='center', fontsize=8, color='green', weight='bold')

    ax.set_xlim(-1, len(rgb_values))
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Create zip archive
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(save_path, os.path.basename(save_path))
    print(f"✅ Bubble plot zipped as {zip_path}")

    plt.show()
    plt.close(fig)

    return fig


# ✅ Cluster and mark colors on a palette, save as image and zip
# color palette
def hex_to_rgb(hex_color):
    if isinstance(hex_color, str) and hex_color.startswith('#'):
        hex_color = hex_color[1:]  # Remove '#' prefix
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return hex_color

def parse_colors(color_data):
    colors = []
    for color in color_data:
        if isinstance(color, str):
            if color.startswith('#'):
                hex_color = color[1:]
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                colors.append(rgb)
            else:
                try:
                    rgb = [int(x.strip()) for x in color.split(',')]
                    colors.append(rgb)
                except:
                    continue
        elif isinstance(color, (list, tuple)):
            colors.append([int(x) for x in color[:3]])
    return colors


def cluster_and_mark_palette(vein_colors, boundary_colors, num_clusters=5, output_path="clustered_palette.png", zip_path="clustered_palette.zip"):
    vein_rgb = parse_colors(vein_colors)
    boundary_rgb = parse_colors(boundary_colors)
    
    print(f"Parsed {len(vein_rgb)} vein colors and {len(boundary_rgb)} boundary colors")
    
    all_colors = vein_rgb + boundary_rgb
    types = ['vein'] * len(vein_rgb) + ['boundary'] * len(boundary_rgb)
    
    if len(all_colors) == 0:
        print("No colors provided for clustering.")
        return []
    
    # clustering
    colors_array = np.array(all_colors)
    kmeans = KMeans(n_clusters=min(num_clusters, len(all_colors)), random_state=42)
    labels = kmeans.fit_predict(colors_array)
    
    # create a color palette (HSV spectrum base)
    palette_width, palette_height = 512, 256
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

    for y in range(palette_height):
        for x in range(palette_width):
            hue = (x / palette_width) * 360
            saturation = 1.0
            value = y / palette_height
            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            palette[y, x] = [int(r * 255), int(g * 255), int(b * 255)]

    def find_closest_position(target_rgb):
        target = np.array(target_rgb[:3])
        min_distance = float('inf')
        best_pos = (0, 0)
        step = 4
        for y in range(0, palette_height, step):
            for x in range(0, palette_width, step):
                palette_color = palette[y, x]
                distance = np.linalg.norm(target - palette_color)
                if distance < min_distance:
                    min_distance = distance
                    best_pos = (x, y)
        return best_pos
    
    cluster_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    palette_pil = Image.fromarray(palette)
    draw = ImageDraw.Draw(palette_pil)
    font = ImageFont.load_default()
    
    marker_info = []

    for i, (color, label, color_type) in enumerate(zip(all_colors, labels, types)):
        pos = find_closest_position(color)
        if pos:
            x, y = pos
            marker_color = cluster_colors[label % len(cluster_colors)]

            if color_type == 'vein':
                draw.line([(x - 3, y - 3), (x + 3, y + 3)], fill=marker_color, width=1)
                draw.line([(x - 3, y + 3), (x + 3, y - 3)], fill=marker_color, width=1)
            else:
                draw.line([(x - 3, y), (x + 3, y)], fill=marker_color, width=1)
                draw.line([(x, y - 3), (x, y + 3)], fill=marker_color, width=1)

            r, g, b = color[:3]
            hex_code = f"#{r:02X}{g:02X}{b:02X}"
            marker_info.append((x, y, hex_code))

    for x, y, hex_code in marker_info:
        text_x = x + 10 if x <= palette_width - 50 else x - 50
        text_y = y - 20 if y >= 20 else y + 10
        draw.text((text_x, text_y), hex_code, fill=(0, 0, 0), font=font)

    palette = np.array(palette_pil)
    
    # Save palette image
    cv2.imwrite(output_path, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))
    print(f"✅ Palette saved at {output_path}")

    # Create ZIP file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(output_path, os.path.basename(output_path))
    print(f"✅ Zipped to {zip_path}")

    # Display
    plt.figure(figsize=(15, 8))
    plt.imshow(palette)
    plt.title('Color Palette with Clustered Points')
    plt.xlabel("Hue Spectrum")
    plt.ylabel("Brightness")
    plt.tight_layout()
    plt.show()

    return labels
