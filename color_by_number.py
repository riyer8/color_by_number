from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os

# load and convert image to RGB format
def get_image(filename: str) -> Image.Image:
    img = Image.open(filename)
    img = img.convert('RGB') # RGB 3-channel array
    return img

# crop image to largest square image and downgrade to target sizing for future use
def reshape_image(img: Image.Image, target_size: int = 800, img_dir: str = "images_generated") -> tuple[Image.Image, str]:
    width, height = img.size
    min_dim = min(width, height)

    # crop center to create square image
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img_cropped = img.crop((left, top, right, bottom))

    # LANCZOS resampling provides high-quality downscaling to preserve edge detection
    img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)
    img_resized.save(img_dir + "processed_input.jpg")
    return img_resized, img_dir

# converting PIL image to RGB 3-channel array
def get_rgb_data(img: Image.Image) -> np.ndarray:
    return np.array(img)

# detecting objects in the image for creating edges
def detect_object_boundaries(pixels: np.ndarray, edge_threshold: int = 50) -> np.ndarray:
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    
    # bilateral filter decreases noise while preserving edges, prevents similar colors from being treated as separate objects
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # adaptive threshold vs fixed threshold for varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # fine edge details
    edges_canny = cv2.Canny(filtered, edge_threshold, edge_threshold * 2)
    
    # combine adaptive threshold and canny edges, captures both strong object boundaries and subtle detail edges
    combined_edges = cv2.bitwise_or(edges_canny, 255 - adaptive_thresh)
    
    kernel = np.ones((3, 3), np.uint8)
    
    # fills gaps in object boundaries
    edges_dilated = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # fills small holes within edge regions
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return edges_closed > 0

def find_color_concentrations(pixels: np.ndarray, border_mask: np.ndarray, num_colors: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    New approach: Find color concentrations iteratively for more distinct palette
    WHY: Sequential clustering ensures colors are maximally distinct from each other
    WHY: Color distance threshold prevents similar colors from being separate clusters
    """
    height, width, channels = pixels.shape
    
    # Work only with non-border pixels to avoid edge color contamination
    non_border_mask = ~border_mask
    available_pixels = pixels[non_border_mask].astype(np.float32)
    
    if len(available_pixels) == 0:
        # Fallback to standard clustering if no non-border pixels
        flat_pixels = pixels.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flat_pixels)
        return labels.reshape(height, width), kmeans.cluster_centers_.astype(int)
    
    # Store the final cluster centers and labels
    final_centers = []
    pixel_assignments = np.full(len(available_pixels), -1, dtype=int)
    unassigned_mask = np.ones(len(available_pixels), dtype=bool)
    
    color_distance_threshold = 40  # Minimum distance between distinct colors
    
    for cluster_id in range(num_colors):
        if not np.any(unassigned_mask):
            break
            
        unassigned_pixels = available_pixels[unassigned_mask]
        
        if len(unassigned_pixels) < 10:  # Too few pixels remaining
            break
        
        # Find the most concentrated color in remaining pixels
        # WHY: KMeans with k=1 finds the centroid of the largest concentration
        kmeans_single = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans_single.fit(unassigned_pixels)
        new_center = kmeans_single.cluster_centers_[0]
        
        # Check if this color is distinct enough from existing centers
        is_distinct = True
        for existing_center in final_centers:
            # Calculate Euclidean distance in RGB space
            distance = np.linalg.norm(new_center - existing_center)
            if distance < color_distance_threshold:
                is_distinct = False
                break
        
        if not is_distinct and len(final_centers) > 0:
            # Skip this color concentration, try to find the next one
            # Remove the closest pixels to avoid infinite loop
            distances = np.linalg.norm(unassigned_pixels - new_center, axis=1)
            closest_pixels = distances < color_distance_threshold
            unassigned_indices = np.where(unassigned_mask)[0]
            pixels_to_remove = unassigned_indices[closest_pixels]
            unassigned_mask[pixels_to_remove] = False
            continue
        
        final_centers.append(new_center)
        
        # Assign pixels close to this center
        distances = np.linalg.norm(unassigned_pixels - new_center, axis=1)
        similarity_threshold = 35  # Pixels within this distance belong to this color
        close_pixels = distances < similarity_threshold
        
        # Update assignments
        unassigned_indices = np.where(unassigned_mask)[0]
        assigned_indices = unassigned_indices[close_pixels]
        pixel_assignments[assigned_indices] = cluster_id
        
        # Remove assigned pixels from unassigned set
        unassigned_mask[assigned_indices] = False
        
        print(f"Found color concentration {cluster_id + 1}: {len(assigned_indices)} pixels assigned")
    
    # Handle any remaining unassigned pixels by assigning to nearest center
    if np.any(unassigned_mask) and len(final_centers) > 0:
        remaining_pixels = available_pixels[unassigned_mask]
        for i, pixel in enumerate(remaining_pixels):
            distances = [np.linalg.norm(pixel - center) for center in final_centers]
            nearest_center = np.argmin(distances)
            remaining_idx = np.where(unassigned_mask)[0][i]
            pixel_assignments[remaining_idx] = nearest_center
    
    # Convert back to full image labels
    full_labels = np.zeros(height * width, dtype=int)
    non_border_indices = np.where(non_border_mask.flatten())[0]
    full_labels[non_border_indices] = pixel_assignments
    
    final_centers_array = np.array(final_centers) if final_centers else np.array([[128, 128, 128]])
    final_centers_array = np.clip(final_centers_array, 0, 255).astype(int)
    
    return full_labels.reshape(height, width), final_centers_array

# reduce image to the color palette
def reduce_colors(pixels: np.ndarray, border_mask: np.ndarray, num_colors: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    Improved K-means clustering that excludes border pixels for better color analysis
    WHY: Border pixels contain mixed/contaminated colors that skew the palette
    WHY: Standard K-means is reliable and well-tested, just needs clean input data
    """
    height, width, _ = pixels.shape
    
    # Work only with non-border pixels for cleaner color clustering
    non_border_mask = ~border_mask
    non_border_pixels = pixels[non_border_mask].astype(np.float32)
    
    # Fallback if all pixels are borders (shouldn't happen with improved edge detection)
    if len(non_border_pixels) == 0:
        flat_pixels = pixels.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flat_pixels)
        return labels.reshape(height, width), kmeans.cluster_centers_.astype(int)
    
    # Standard K-means clustering on clean non-border pixels
    actual_clusters = min(num_colors, len(non_border_pixels))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    kmeans.fit(non_border_pixels)
    
    # Apply the learned clusters to ALL pixels (including borders)
    # WHY: Border pixels still need color assignments, but don't influence the palette
    flat_pixels = pixels.reshape(-1, 3).astype(np.float32)
    all_labels = kmeans.predict(flat_pixels)
    
    centers_rgb = np.clip(kmeans.cluster_centers_, 0, 255).astype(int)
    
    return all_labels.reshape(height, width), centers_rgb

# connect regions with the same palette numbers
def create_region_map(labels: np.ndarray, border_mask: np.ndarray, min_region_size: int = 100) -> np.ndarray:
    height, width = labels.shape
    region_map = np.zeros_like(labels, dtype=np.int32)
    current_region_id = 1
    
    available_mask = ~border_mask
    
    for color_id in np.unique(labels):
        # ensure the color is not a border color (avoiding a lot of black variations due to borders)
        color_mask = (labels == color_id) & available_mask
        
        if not color_mask.any():
            continue
            
        color_mask_uint8 = color_mask.astype(np.uint8)
        num_components, components = cv2.connectedComponents(color_mask_uint8)
        
        for comp_id in range(1, num_components):
            component_mask = (components == comp_id)
            region_size = np.sum(component_mask)
            
            if region_size >= min_region_size:
                region_map[component_mask] = current_region_id
                current_region_id += 1
    
    region_map = merge_small_regions_improved(region_map, labels, border_mask, min_region_size * 2)
    
    return region_map

# merging regions so we don't have a ton of smaller regions
def merge_small_regions_improved(region_map: np.ndarray, labels: np.ndarray, border_mask: np.ndarray, min_size: int) -> np.ndarray:
    result_map = region_map.copy()
    
    for region_id in np.unique(region_map):
        if region_id == 0:
            continue
            
        region_mask = (region_map == region_id)
        region_size = np.sum(region_mask)
        
        if region_size < min_size:
            region_labels = labels[region_mask]
            if len(region_labels) == 0:
                continue
            region_color = np.bincount(region_labels).argmax()
            
            # find neighboring region's colors
            dilated = ndimage.binary_dilation(region_mask)
            boundary = dilated & ~region_mask & ~border_mask
            
            neighbor_regions = np.unique(region_map[boundary])
            neighbor_regions = neighbor_regions[neighbor_regions != 0]
            neighbor_regions = neighbor_regions[neighbor_regions != region_id]
            
            best_neighbor = None
            best_size = 0
            
            for neighbor_id in neighbor_regions:
                neighbor_mask = (region_map == neighbor_id)
                neighbor_labels = labels[neighbor_mask]
                if len(neighbor_labels) == 0:
                    continue
                neighbor_color = np.bincount(neighbor_labels).argmax()
                
                if neighbor_color == region_color:
                    neighbor_size = np.sum(neighbor_mask)
                    if neighbor_size > best_size:
                        best_neighbor = neighbor_id
                        best_size = neighbor_size
            
            # merge with best neighbor
            if best_neighbor is not None:
                result_map[region_mask] = best_neighbor
    
    return result_map

def create_final_borders(region_map: np.ndarray, original_borders: np.ndarray, border_thickness: int = 2) -> np.ndarray:
    height, width = region_map.shape
    border_map = np.zeros((height, width), dtype=np.uint8)
    
    border_map[original_borders] = 255
    
    for y in range(height - 1):
        for x in range(width - 1):
            current = region_map[y, x]
            right = region_map[y, x + 1]
            down = region_map[y + 1, x]
            
            if current != 0 and right != 0 and current != right:
                border_map[y, x] = 255
            if current != 0 and down != 0 and current != down:
                border_map[y, x] = 255
    
    if border_thickness > 1:
        kernel = np.ones((border_thickness, border_thickness), np.uint8)
        border_map = cv2.dilate(border_map, kernel, iterations=1)
    
    return border_map

def create_color_by_number(region_map: np.ndarray, labels: np.ndarray, palette: np.ndarray, border_map: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = region_map.shape
    
    # begin with an empty image of proper dimensions
    result = np.full((height, width, 3), 255, dtype=np.uint8)
    
    region_to_color = {}
    
    unique_regions = np.unique(region_map)
    unique_regions = unique_regions[unique_regions != 0]
    
    color_number = 1
    used_colors = set()
    
    for region_id in unique_regions:
        region_mask = (region_map == region_id)
        if np.sum(region_mask) == 0:
            continue
            
        region_labels = labels[region_mask]
        if len(region_labels) > 0:
            most_common_label = np.bincount(region_labels).argmax()
            
            if most_common_label not in used_colors:
                region_to_color[region_id] = color_number
                used_colors.add(most_common_label)
                color_number += 1
            else:
                for rid, cnum in region_to_color.items():
                    rid_mask = (region_map == rid)
                    if rid_mask.any():
                        rid_labels = labels[rid_mask]
                        if len(rid_labels) > 0:
                            rid_color = np.bincount(rid_labels).argmax()
                            if rid_color == most_common_label:
                                region_to_color[region_id] = cnum
                                break
                else:
                    region_to_color[region_id] = color_number
                    used_colors.add(most_common_label)
                    color_number += 1
    
    result[border_map > 0] = [0, 0, 0]
    
    for region_id in unique_regions:
        if region_id == 0 or region_id not in region_to_color:
            continue
            
        region_mask = (region_map == region_id)
        if np.sum(region_mask) < 100:  # skipping very small regions
            continue
            
        region_coords = np.where(region_mask & (border_map == 0))
        if len(region_coords[0]) == 0:
            continue
            
        center_y = int(np.mean(region_coords[0]))
        center_x = int(np.mean(region_coords[1]))
        
        number = region_to_color[region_id]
        
        pil_img = Image.fromarray(result)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font_size = max(12, min(24, int(np.sqrt(np.sum(region_mask)) / 15)))
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        text = str(number)
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 10, 10
        
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2
        
        for adj_x in [-1, 0, 1]:
            for adj_y in [-1, 0, 1]:
                if adj_x != 0 or adj_y != 0:
                    draw.text((text_x + adj_x, text_y + adj_y), text, 
                             fill=(255, 255, 255), font=font)
        
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        result = np.array(pil_img)
    
    return result, region_to_color

# creating filled image with color by number colors
def create_filled_image(region_map: np.ndarray, labels: np.ndarray, palette: np.ndarray, border_map: np.ndarray) -> np.ndarray:
    height, width = region_map.shape
    filled_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for region_id in np.unique(region_map):
        if region_id == 0:
            continue
            
        region_mask = (region_map == region_id)
        
        region_labels = labels[region_mask]
        if len(region_labels) > 0:
            most_common_label = np.bincount(region_labels).argmax()
            
            if most_common_label < len(palette):
                filled_image[region_mask] = palette[most_common_label]
    
    filled_image[border_map > 0] = [0, 0, 0]
    
    return filled_image

# create a color palette guide showing number -> color mapping
def create_color_palette_guide(palette: np.ndarray, color_mapping: dict, labels: np.ndarray, region_map: np.ndarray) -> np.ndarray:
    if len(color_mapping) == 0:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    number_to_palette_color = {}
    
    for region_id, color_num in color_mapping.items():
        if color_num not in number_to_palette_color:
            region_mask = (region_map == region_id)
            if region_mask.any():
                region_labels = labels[region_mask]
                if len(region_labels) > 0:
                    most_common_label = np.bincount(region_labels).argmax()
                    if most_common_label < len(palette):
                        number_to_palette_color[color_num] = palette[most_common_label]
    
    num_colors = len(number_to_palette_color)
    if num_colors == 0:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
        
    swatch_size = 60
    cols = min(6, num_colors)
    rows = (num_colors + cols - 1) // cols
    
    guide_width = cols * swatch_size
    guide_height = rows * swatch_size
    
    guide = np.ones((guide_height, guide_width, 3), dtype=np.uint8) * 255
    
    for i, (color_num, color) in enumerate(sorted(number_to_palette_color.items())):
        row = i // cols
        col = i % cols
        
        y1 = row * swatch_size
        y2 = y1 + swatch_size
        x1 = col * swatch_size
        x2 = x1 + swatch_size
        
        guide[y1:y2, x1:x2] = color
        
        cv2.rectangle(guide, (x1, y1), (x2-1, y2-1), (0, 0, 0), 2)
    
        pil_guide = Image.fromarray(guide)
        draw = ImageDraw.Draw(pil_guide)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = str(color_num)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x1 + (swatch_size - text_width) // 2
        text_y = y1 + (swatch_size - text_height) // 2
        
        for adj_x in [-1, 0, 1]:
            for adj_y in [-1, 0, 1]:
                if adj_x != 0 or adj_y != 0:
                    draw.text((text_x + adj_x, text_y + adj_y), text, 
                             fill=(255, 255, 255), font=font)
        
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        guide = np.array(pil_guide)
    
    return guide

def save_image(final_image: Image.Image, filled_image: Image.Image, color_guide: np.ndarray, new_img_folder: str) -> None:
    Image.fromarray(final_image).save(new_img_folder + "color_by_number_final.png")
    Image.fromarray(filled_image).save(new_img_folder + "color_by_number_filled.png")
    Image.fromarray(color_guide).save(new_img_folder + "color_palette_guide.png")

def generate_plot(reshaped_img: Image.Image, final_image: Image.Image, filled_image: Image.Image, color_guide: np.ndarray, new_img_folder: str) -> None:
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(reshaped_img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(final_image)
    plt.title("Color by Number")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(filled_image)
    plt.title("Filled Result")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(color_guide)
    plt.title("Color Guide")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(new_img_folder + "color_by_number_comparison.png")
    plt.show()

def main(input_filename: str, num_colors: int = 8, min_region_size: int = 300, edge_threshold: int = 50, img_folder: str = ""):
    
    new_img_dir = "images_generated/" + img_folder + "/"

    # Create directory if it doesn't exist
    os.makedirs(new_img_dir, exist_ok=True)

    print("🖼️ Loading and processing image...")
    img = get_image(input_filename)
    reshaped_img, _ = reshape_image(img, target_size=800, img_dir = new_img_dir)
    pixels = get_rgb_data(reshaped_img)
    
    print("🔍 Detecting object boundaries...")
    border_mask = detect_object_boundaries(pixels, edge_threshold)
    
    print(f"🎨 Reducing to {num_colors} colors with improved K-means...")
    labels, palette = reduce_colors(pixels, border_mask, num_colors=num_colors)
    
    print("🧩 Creating regions...")
    region_map = create_region_map(labels, border_mask, min_region_size=min_region_size)
    
    print("🖊️ Creating final borders...")
    final_border_map = create_final_borders(region_map, border_mask, border_thickness=1)
    
    print("🎯 Generating final color-by-number...")
    final_image, color_mapping = create_color_by_number(region_map, labels, palette, final_border_map)
    
    print("🪄 Creating filled version...")
    filled_image = create_filled_image(region_map, labels, palette, border_mask)
    
    print("🗂️ Creating color guide...")
    color_guide = create_color_palette_guide(palette, color_mapping, labels, region_map)
    
    print("💾 Saving images...")
    save_image(final_image, filled_image, color_guide, new_img_dir)
    
    print("🖥️ Displaying results...")
    generate_plot(reshaped_img, final_image, filled_image, color_guide, new_img_dir)
    
    print("✅ Color-by-number saved as 'color_by_number_final.png'")
    print("✅ Filled version saved as 'color_by_number_filled.png'")
    print("✅ Color guide saved as 'color_palette_guide.png'")
    print(f"🔢 Total regions: {len(color_mapping)}")
    print(f"🌈 Palette colors: {len(palette)}")
    
    # Display palette information
    print("\n🎨 Color Palette Analysis:")
    for i, color in enumerate(palette):
        print(f"  Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]})")
    
    return final_image, filled_image, color_guide, color_mapping

if __name__ == "__main__":
    original_img_dir = "sample.png"
    new_img_folder = "current_color_by_number"
    
    final_image_new, filled_image_new, color_guide_new, mapping_new = main(
        original_img_dir, 
        num_colors=12, 
        min_region_size=300, 
        edge_threshold=50,
        img_folder = new_img_folder
    )
    