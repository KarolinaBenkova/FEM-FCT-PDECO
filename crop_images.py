from PIL import Image
import os


def crop_image(input_image_path, output_image_path, crop_box):
    """
    Crop an image to the specified box and save the result.

    :param input_image_path: Path to the input image
    :param output_image_path: Path to save the cropped image
    :param crop_box: A tuple (left, upper, right, lower) defining the crop box
    """
    image = Image.open(input_image_path)
    cropped_image = image.crop(crop_box)
    cropped_image.save(output_image_path)
    print(f"Cropped image saved to {output_image_path}")

# Define your input and output image paths
input_folder = 'exact_sol_FCT_PGD/beta01_upper05'
# output_folder = 'solid_body_rotation_noFCT_cropped'
output_folder = 'exact_sol_FCT_PGD/presentation_beta01'
# input_image_path = "plot_000.png"
# output_image_path = "plo.t_000_cropped.png"

# Define the crop box (left, upper, right, lower)
# crop_box = (360, 360, 720, 0)  # Replace with your specific coordinates
# crop_box = (350, 30, 700, 330)  # Replace with your specific coordinates
crop_box = (30, 100, 700, 950)  # Replace with your specific coordinates
# Crop the image
# crop_image(input_image_path, output_image_path, crop_box)

start = 0
end = 1590
s = 0
for i in range(start, end + 1):
    if i%30 == 0:
        filename = f"plot_{i:03d}.png"
        filename_out = f'beta01_upper05_{s}.png'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename_out)
        crop_image(input_image_path, output_image_path, crop_box)
        s+=1
