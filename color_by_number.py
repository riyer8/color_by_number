from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from image_manipulation import ImageManipulation
from edge_detection import EdgeDetection
from color_detection import ColorDetection

class ColorByNumber:
    def __init__(self):
        pass

    def save_image(self, final_image: Image.Image, filled_image: Image.Image, color_guide: np.ndarray, new_img_folder: str) -> None:
        Image.fromarray(final_image).save(new_img_folder + "color_by_number_final.png")
        Image.fromarray(filled_image).save(new_img_folder + "color_by_number_filled.png")
        Image.fromarray(color_guide).save(new_img_folder + "color_palette_guide.png")

    def generate_plot(self, reshaped_img: Image.Image, final_image: Image.Image, filled_image: Image.Image, color_guide: np.ndarray, new_img_folder: str) -> None:
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(reshaped_img)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(final_image)
        plt.title("Edge Detection")
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

    def main(self, input_filename: str, num_colors: int = 8, min_region_size: int = 300, edge_threshold: int = 50, img_folder: str = ""):
        edge_detection = EdgeDetection()
        color_detection = ColorDetection()
        image_manipulation = ImageManipulation()

        new_img_dir = "images_generated/" + img_folder + "/"

        # Create directory if it doesn't exist
        os.makedirs(new_img_dir, exist_ok=True)

        print("🖼️ Loading and processing image...")
        img = image_manipulation.get_image(input_filename)
        reshaped_img, _ = image_manipulation.reshape_image(img, target_size=800, img_dir = new_img_dir)
        pixels = image_manipulation.get_rgb_data(reshaped_img)
        
        print("🔍 Detecting object boundaries...")
        border_mask = edge_detection.detect_object_boundaries(pixels, edge_threshold)
        
        print(f"🎨 Reducing to {num_colors} colors with improved K-means...")
        labels, palette = color_detection.reduce_colors(pixels, border_mask, num_colors=num_colors)
        
        print("🧩 Creating regions...")
        region_map = edge_detection.create_region_map(labels, border_mask, min_region_size=min_region_size)
        
        print("🖊️ Creating final borders...")
        final_border_map = edge_detection.create_final_borders(region_map, border_mask, border_thickness=1)
        
        print("🎯 Generating final color-by-number...")
        final_image, color_mapping = color_detection.create_color_by_number(region_map, labels, palette, final_border_map)
        
        print("🪄 Creating filled version...")
        filled_image = color_detection.create_filled_image(region_map, labels, palette, border_mask)
        
        print("🗂️ Creating color guide...")
        color_guide = color_detection.create_color_palette_guide(palette, color_mapping, labels, region_map)
        
        print("💾 Saving images...")
        self.save_image(final_image, filled_image, color_guide, new_img_dir)
        
        print("🖥️ Displaying results...")
        self.generate_plot(reshaped_img, final_image, filled_image, color_guide, new_img_dir)
        
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