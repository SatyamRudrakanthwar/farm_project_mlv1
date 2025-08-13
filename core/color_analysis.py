import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
import webcolors
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from PIL import Image, ImageDraw
import os
import pandas as pd
from sklearn.metrics import silhouette_score


#  Match RGB color to nearest CSS3 color name
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


#  Generate vein skeleton from grayscale image
def leaf_vein_skeleton(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter to preserve edges while denoising
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    edges = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 15, 3)
  
    
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


#  Generate leaf boundary mask using dilation
def leaf_boundary_dilation(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    _, binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure leaf is foreground (invert if needed)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    #     binary = cv2.bitwise_not(binary)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    boundary = cv2.subtract(dilated, binary)
    return boundary

# Extract dominant colors around a binary mask
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
        
    kmeans = KMeans(n_clusters=min(num_colors, len(masked_pixels)), random_state=42)
    labels = kmeans.fit_predict(masked_pixels)

    color_stats: Dict[str, Dict] = {}
    total_pixels = len(masked_pixels)  # Use filtered pixels count

    for i in range(kmeans.n_clusters):
        idx = np.where(labels == i)[0]  
        if len(idx) == 0:
            continue
            
        cluster_colors = masked_pixels[idx]  
        mean_color = np.mean(cluster_colors, axis=0).astype(int)
        
        # # Final validation of cluster center
        # if not is_valid_leaf_color(mean_color, color_type):
        #     print(f"Skipping invalid cluster color: {mean_color}")
        #     continue
        
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
    # plt.show()


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
    # plt.show()

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

# Bubble plot of dominant colors
def bubble_plot(color_stats, title="Color Bubble Plot", save_path=None):

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
        
    # Calculate and display green percentage below each bubble
    for i, rgb in enumerate(rgb_values):
        r, g, b = rgb
        total_intensity = r + g + b
        if total_intensity > 0:
            green_percentage = (g / total_intensity) * 100
        else:
            green_percentage = 0
    
        # Display green percentage below the bubble
        ax.text(x[i], y[i] - 0.02, f"Green: {green_percentage:.1f}%", 
                ha='center', fontsize=8, color='green', weight='bold')

    ax.set_xlim(-1, len(rgb_values))
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    # save the bubble plot
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Bubble plot saved to {save_path}")
        # plt.show()
        plt.close(fig)
        return fig

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


def cluster_and_mark_palette(vein_colors, boundary_colors, num_clusters=5, output_path="clustered_palette.png"):
    vein_rgb = parse_colors(vein_colors)
    boundary_rgb = parse_colors(boundary_colors)
    
    print(f"Parsed {len(vein_rgb)} vein colors and {len(boundary_rgb)} boundary colors")
    
    all_colors = vein_rgb + boundary_rgb
    types = ['vein'] * len(vein_rgb) + ['boundary'] * len(boundary_rgb)
    
    # clustering
    colors_array = np.array(all_colors)
    kmeans = KMeans(n_clusters=min(num_clusters, len(all_colors)), random_state=42)
    labels = kmeans.fit_predict(colors_array)
    
    # create a color palette
    palette_width, palette_height = 512, 256
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

    for y in range(palette_height):
        for x in range(palette_width):
            # Mapping x to hue (0-360 degrees), y to saturation or brightness
            hue = (x / palette_width) * 360
            saturation = 1.0
            value = y / palette_height

            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
            palette[y, x] = [int(r*255), int(g*255), int(b*255)]

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
    
    # cluster colors for marking
    cluster_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
    
    palette_pil=Image.fromarray(palette)
    draw=ImageDraw.Draw(palette_pil) # to draw the text on palette
    
    from PIL import ImageFont
    font=ImageFont.load_default()
    
    marker_info=[]
    data_info=[]
    
    # mark points as clusters
    for i, (color, label, color_type) in enumerate(zip(all_colors, labels, types)):
        pos = find_closest_position(color)
        if pos:
            x, y = pos
            marker_color = cluster_colors[label % len(cluster_colors)]
            
            if color_type == 'vein':
                draw.line([(x-3, y-3), (x+3, y+3)], fill=marker_color, width=1)
                draw.line([(x-3, y+3), (x+3, y-3)], fill=marker_color, width=1) # vein point marked as (x)
            else:
                draw.line([(x-3, y), (x+3, y)], fill=marker_color, width=1)
                draw.line([(x, y-3), (x, y+3)], fill=marker_color, width=1)# boundary point marked as (+)
                
            r,g,b = color[:3]
            hex_code = f"#{r:02X}{g:02X}{b:02X}"
            marker_info.append((x,y,hex_code))
            data_info.append({x:x,y:y})
    df = pd.DataFrame(data_info)
    df.to_excel('pure diseased folder/excel sheet(analysis)/temp.xlsx', index=False)        
            
    for x,y , hex_code in marker_info:        
            text_x=x + 10
            text_y= y 
            
            if text_x>(palette_width-50):
                text_x = x-50
            if text_y < 10:
                text_y = y +10
            
            # Then draw the text
            draw.text((text_x, text_y), hex_code, fill=(0, 0, 0), font=font)
    
    # Convert back to save
    palette = np.array(palette_pil)
    
    # save palette
    cv2.imwrite(output_path, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))
    print(f"Saved clustered palette to {output_path}")
    
    # show results
    plt.figure(figsize=(15, 8))
    plt.imshow(palette)
    plt.title('Color Palette with Clustered Points')
    plt.xlabel("Hue Spectrum")
    plt.ylabel("Brightness")
    plt.tight_layout()
    
    return labels
