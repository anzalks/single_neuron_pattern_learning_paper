
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
from scipy.signal import find_peaks, savgol_filter
import platform
import cv2  # OpenCV is required for this implementation

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

def detect_rotation_angle(img_gray):
    """
    Detects the rotation angle of vertical bars in the grayscale image.
    """
    # Convert PIL image to NumPy array
    img_array = np.array(img_gray)

    # Use Canny edge detection
    edges = cv2.Canny(img_array, threshold1=50, threshold2=150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)

    if lines is None:
        print("No lines detected.")
        return 0  # Return zero rotation if no lines are detected

    # Calculate the angles of the lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angle = (theta * 180 / np.pi) - 90  # Convert to degrees and adjust range
            angles.append(angle)

    # Compute the median angle
    median_angle = np.median(angles)
    print(f"Detected rotation angle: {median_angle:.2f} degrees")
    return median_angle

def rotate_image(image, angle):
    """
    Rotates the image by the given angle.
    """
    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # Get image dimensions
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Perform the rotation
    rotated_array = cv2.warpAffine(img_array, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Convert back to PIL image
    rotated_image = Image.fromarray(rotated_array)
    return rotated_image

def process_image_pillow(image_path, physical_dist_in_um, num_bars=2):
    """
    Processes the image to automatically detect vertical bars and calculate the scaling factor.
    """
    # Load the image and convert to grayscale
    image = load_image_pillow(image_path)
    if image is None:
        raise ValueError("Failed to load the image.")

    img_gray = image.convert('L')  # Convert to grayscale

    # Detect rotation angle
    rotation_angle = detect_rotation_angle(img_gray)

    # Rotate the image to correct the orientation
    if abs(rotation_angle) > 0.1:  # Only rotate if angle is significant
        img_gray = rotate_image(img_gray, rotation_angle)
        image = rotate_image(image, rotation_angle)

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

# The rest of the script remains the same
# ... [Include the rest of the functions as previously provided]

# Ensure you have the latest versions of the other functions from the previous script

def main():
    # Main script to run both processes
    image_path = "/path/to/your/image.tiff"  # Update with your image path
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
