import numpy as np
import cv2
from scipy import ndimage

class EdgeDetection:
    def __init__(self):
        pass

    # detecting objects in the image for creating edges
    def detect_object_boundaries(self, pixels: np.ndarray, edge_threshold: int = 50) -> np.ndarray:
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

    # connect regions with the same palette numbers
    def create_region_map(self, labels: np.ndarray, border_mask: np.ndarray, min_region_size: int = 100) -> np.ndarray:
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
        
        region_map = self.merge_small_regions_improved(region_map, labels, border_mask, min_region_size * 2)
        
        return region_map

    # merging regions so we don't have a ton of smaller regions
    def merge_small_regions_improved(self, region_map: np.ndarray, labels: np.ndarray, border_mask: np.ndarray, min_size: int) -> np.ndarray:
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

    def create_final_borders(self, region_map: np.ndarray, original_borders: np.ndarray, border_thickness: int = 2) -> np.ndarray:
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