from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import cv2

class ColorDetection:
    def __init__(self):
        pass

    def find_color_concentrations(self, pixels: np.ndarray, border_mask: np.ndarray, num_colors: int = 12) -> tuple[np.ndarray, np.ndarray]:
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
    def reduce_colors(self, pixels: np.ndarray, border_mask: np.ndarray, num_colors: int = 12) -> tuple[np.ndarray, np.ndarray]:
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

    def create_color_by_number(self, region_map: np.ndarray, labels: np.ndarray, palette: np.ndarray, border_map: np.ndarray) -> tuple[np.ndarray, dict]:
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
    def create_filled_image(self, region_map: np.ndarray, labels: np.ndarray, palette: np.ndarray, border_map: np.ndarray) -> np.ndarray:
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
    def create_color_palette_guide(self, palette: np.ndarray, color_mapping: dict, labels: np.ndarray, region_map: np.ndarray) -> np.ndarray:
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