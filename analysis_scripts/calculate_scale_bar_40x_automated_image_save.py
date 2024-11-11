
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
from scipy.signal import find_peaks, savgol_filter
import platform

def load_image_pillow(image_path):
    """
    Loads an image using Pillow.
    """
    try:
        img = Image.open(image_path)
        return img
    except IOError as e:
        print(f"Error loading image: {e}")
        return None

def process_image_pillow(image_path, physical_dist_in_um, num_bars=2):
    """
    Processes the image to automatically detect vertical bars and calculate the scaling factor.
    """
    # Load the image and convert to grayscale
    image = load_image_pillow(image_path)
    if image is None:
        raise ValueError("Failed to load the image.")

    img_gray = image.convert('L')  # Convert to grayscale
    img_array = np.array(img_gray)

    # Invert the image if bars are dark on light background
    img_inverted = ImageOps.invert(img_gray)
    img_inverted_array = np.array(img_inverted)

    # Projection profile along the x-axis (sum of pixel values along y-axis)
    projection = img_inverted_array.sum(axis=0)

    # Smooth the projection to reduce noise
    window_length = max(3, min(51, len(projection) // 2 * 2 - 1))  # Ensure window_length is valid
    projection_smooth = savgol_filter(projection, window_length=window_length, polyorder=3)

    # Find peaks in the projection profile (positions of vertical bars)
    peaks, _ = find_peaks(projection_smooth, distance=20, height=np.max(projection_smooth)*0.5)

    if len(peaks) < num_bars:
        raise ValueError(f"Could not detect {num_bars} bars in the image.")

    # Select two peaks for scaling calculation
    peak_positions = peaks[:num_bars]

    # Get the x-coordinates of the detected bars
    x_coords = peak_positions

    # Assume the y-coordinate is at the center of the image
    y_center = img_array.shape[0] // 2

    # Define points at the center of the detected bars
    point1 = (x_coords[0], y_center)
    point2 = (x_coords[1], y_center)

    # Calculate pixel distance between points (x-axis distance)
    physical_dist_in_pix = abs(point2[0] - point1[0])

    # Calculate scaling factor (micrometers per pixel)
    um_per_pixel = physical_dist_in_um / physical_dist_in_pix

    # Output the scaling factor
    print(f"Scaling factor: {um_per_pixel:.4f} micrometers per pixel")

    # Annotate the image
    annotated_image = image.convert('RGB')
    draw = ImageDraw.Draw(annotated_image)

    # Draw vertical lines at the positions of the bars
    for x in x_coords:
        draw.line([(x, 0), (x, img_array.shape[0])], fill="red", width=2)

    # Draw points
    for point, label_text in zip([point1, point2], ["Point 1", "Point 2"]):
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="yellow", outline="yellow")
        draw.text((x + 10, y - 10), label_text, fill="yellow")

    # Draw line between points
    draw.line([point1, point2], fill="yellow", width=2)

    return um_per_pixel, point1, point2, annotated_image

def add_scale_bar_with_scaling_factor(image, scale_length_um, um_per_pixel, bar_position=(50, 50), scale_bar_thickness=5, font_size=24):
    """
    Adds a white scale bar to a microscope image using Pillow and returns the image object.
    """
    # Draw on the image
    draw = ImageDraw.Draw(image)

    # Calculate scale bar length in pixels using the scaling factor
    scale_bar_length_px = int(scale_length_um / um_per_pixel)

    # Set position of the scale bar
    bar_x = bar_position[0]  # X position in pixels
    bar_y = image.height - bar_position[1]  # Y position in pixels (from the bottom)

    # Draw the scale bar (a white rectangle)
    draw.rectangle(
        [bar_x, bar_y - scale_bar_thickness, bar_x + scale_bar_length_px, bar_y],
        fill="white"
    )

    # Add scale text
    scale_text = f"{scale_length_um} μm"

    # Use a TrueType font if available
    try:
        # Determine the font path based on the operating system
        system = platform.system()
        if system == 'Windows':
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
        elif system == 'Darwin':  # macOS
            font_path = "/Library/Fonts/Arial.ttf"
        elif system == 'Linux':
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        else:
            font_path = None

        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            raise IOError
    except IOError:
        print("TrueType font not found. Using default bitmap font.")
        font = ImageFont.load_default()

    # Get the bounding box for the text
    try:
        # Use textbbox if available (Pillow >= 8.0)
        bbox = draw.textbbox((0, 0), scale_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback to textsize for older Pillow versions
        text_width, text_height = draw.textsize(scale_text, font=font)

    # Draw the text slightly above the scale bar
    draw.text(
        (bar_x, bar_y - scale_bar_thickness - text_height - 5),  # Position slightly above the scale bar
        scale_text,
        fill="white",
        font=font
    )

    return image

def measuring_points(image_with_annotations, um_per_pixel, bar_length_um, point1, point2, output_path, output_plot_path):
    """
    Overlays the scale bar on the annotated image, saves it, and displays it with the points and line.
    """
    # Overlay the scale bar
    image_with_scale_bar = add_scale_bar_with_scaling_factor(
        image_with_annotations,
        scale_length_um=bar_length_um,
        um_per_pixel=um_per_pixel,
        bar_position=(50, 50),  # Set the default position for the scale bar
        scale_bar_thickness=5,  # Thickness of the scale bar
        font_size=24  # Size of the font for the scale label
    )

    # Save the annotated image
    image_with_scale_bar.save(output_path, format='PNG')
    print(f"Annotated image saved to: {output_path}")

    # Convert image to array for plotting
    img_array = np.array(image_with_scale_bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_array)
    ax.axis('off')

    # Save the plot
    plt.savefig(output_plot_path, bbox_inches='tight')
    print(f"Plot saved to: {output_plot_path}")

    # Show the plot
    plt.show()

def main():
    # Main script to run both processes
    image_path = "/Users/anzalks/Documents/pattern_learning_paper/plotting_scripts/python_scripts_paper_ready/data/40X1mm_micrometerslide_01mm_div.tiff"
    physical_dist_in_um = 10  # Physical distance between the bars in micrometers
    bar_length_um = 50  # Desired scale bar length in micrometers

    # Step 1: Calculate the scaling factor and get the selected points
    um_per_pixel, point1, point2, annotated_image = process_image_pillow(image_path, physical_dist_in_um)

    # Create output file paths
    image_dir, image_filename = os.path.split(image_path)
    base_filename, ext = os.path.splitext(image_filename)
    output_filename = f"{base_filename} with points and scale bar marked.png"
    output_plot_filename = f"{base_filename} with points and scale bar marked (plot).png"
    output_path = os.path.join(image_dir, output_filename)
    output_plot_path = os.path.join(image_dir, output_plot_filename)

    # Step 2: Overlay the scale bar, save the image, and save the plot
    measuring_points(annotated_image, um_per_pixel, bar_length_um, point1, point2, output_path, output_plot_path)

if __name__ == "__main__":
    main()
