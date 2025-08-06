from color_by_number import ColorByNumber

color_by_number_runner = ColorByNumber()

original_img_dir = "sample.png"
new_img_folder = "30_cbn"

final_image_new, filled_image_new, color_guide_new, mapping_new = color_by_number_runner.main(
    original_img_dir, 
    num_colors=30, 
    min_region_size=300, 
    edge_threshold=50,
    img_folder = new_img_folder
)
