#!/usr/bin/env python3
"""
Add polar angle boundary circles to polar bin visualization images.

This script adds concentric circles at polar angle boundaries (30°, 60°, 90°, 120°)
to help visualize the relationship between image distance from center and polar angle.

For a fisheye camera with stereographic/division model:
- The polar angle θ relates to the image distance r via:
  r = 2 * f * tan(θ/2)  (stereographic)
  r = f * sin(θ)        (simple radial)

Given the DS model intrinsics, we compute r for each polar angle boundary.
"""

import os
import cv2
import numpy as np
import argparse

# Camera intrinsics for right camera (DS model)
# Format: [k, fu, fv, cu, cv]
DS_INTRINSICS = {
    'k': -0.19312564773657126,   # distortion coefficient
    'fu': 1164.4994369854726,    # focal length u
    'fv': 1164.470760605004,     # focal length v
    'cu': 2255.070357849204,     # principal point u
    'cv': 2269.6419392978223,    # principal point v
}

RESOLUTION = (4512, 4512)  # (width, height)


def compute_image_radius_for_polar_angle(theta_deg, fu, k):
    """
    Compute the image radius r for a given polar angle using the DS model.

    For the Division/Stereographic model ( Kannala-Brandt with k1=0, k2=k, k3=0, k4=0 ):
    r_d = f * θ + k * f * θ^3 + ...

    For stereographic projection:
    r = 2 * f * tan(θ/2)

    We use a simplified approximation:
    r = fu * θ_rad for small angles, but corrected for fisheye distortion.

    A more accurate approach: use the inverse of the distortion function.
    The DS model projects 3D rays to 2D points using:
    r = 2 * fu * tan(θ/2) / (1 - k * tan²(θ/2))

    For a given θ, we can compute r by solving this equation iteratively,
    or use a simpler approximation.
    """
    theta_rad = np.deg2rad(theta_deg)

    # Stereographic projection with distortion
    # r = 2 * f * tan(θ/2) / (1 - k * tan²(θ/2))
    tan_half = np.tan(theta_rad / 2.0)
    tan_sq = tan_half * tan_half
    r = 2.0 * fu * tan_half / (1.0 - k * tan_sq)

    return r


def compute_image_radius_linear(theta_deg, fu):
    """
    Simple linear approximation: r = fu * θ (in radians).
    This is valid for small angles or when distortion is negligible.
    """
    theta_rad = np.deg2rad(theta_deg)
    return fu * theta_rad


def compute_distorted_radius(theta_deg, fu, k):
    """
    Compute the distorted image radius for a given polar angle.

    For fisheye lenses, the relationship is typically:
    r = fu * θ + k1 * fu * θ^3 + k2 * fu * θ^5 + ...

    With the DS model, we use:
    r = 2 * fu * tan(θ/2) / (1 - k * tan²(θ/2))

    This is the forward projection from polar angle to image radius.
    """
    theta_rad = np.deg2rad(theta_deg)

    # For fisheye with division model (Kannala-Brandt type)
    # r = fu * sin(θ) / cos(θ/2) or similar approximations
    # Using stereographic with distortion correction:

    if abs(k) < 1e-10:
        # No distortion, pure stereographic
        r = 2.0 * fu * np.tan(theta_rad / 2.0)
    else:
        # Stereographic with distortion
        tan_half = np.tan(theta_rad / 2.0)
        tan_sq = tan_half * tan_half
        denominator = 1.0 - k * tan_sq
        if abs(denominator) < 1e-10:
            # Near singularity, use limit
            r = 2.0 * fu * tan_half / denominator
        else:
            r = 2.0 * fu * tan_half / denominator

    return r


def draw_polar_circles(image, cx, cy, fu, k, polar_thresholds=[30, 60, 90, 120],
                       thickness=3, circle_style='solid'):
    """
    Draw concentric circles at polar angle boundaries.

    Args:
        image: OpenCV image (will be modified in place)
        cx, cy: Principal point coordinates
        fu: Focal length in pixels
        k: Distortion coefficient
        polar_thresholds: List of polar angles in degrees
        thickness: Circle line thickness
        circle_style: 'solid' or 'dashed'
    """
    h, w = image.shape[:2]

    colors = [
        (0, 255, 0),      # Green: 30°
        (0, 165, 255),   # Orange: 60°
        (0, 0, 255),      # Red: 90°
        (255, 0, 255),    # Purple: 120°
    ]

    labels = ['30°', '60°', '90°', '120°']

    for i, theta in enumerate(polar_thresholds):
        # Compute image radius for this polar angle
        r = compute_distorted_radius(theta, fu, k)

        if r <= 0 or r > max(w, h):
            continue

        color = colors[i] if i < len(colors) else (255, 255, 255)
        label = labels[i] if i < len(labels) else f'{theta}°'

        if circle_style == 'solid':
            cv2.circle(image, (int(cx), int(cy)), int(round(r)), color, thickness)

            # Add label
            label_pos = (int(cx + r * 0.707), int(cy - r * 0.707))  # Upper-right
            cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 2, cv2.LINE_AA)
        else:
            # Dashed circle
            num_dashes = int(r / 5)
            for j in range(num_dashes):
                angle_start = 2 * np.pi * j / num_dashes
                angle_end = 2 * np.pi * (j + 0.4) / num_dashes

                x1 = int(cx + r * np.cos(angle_start))
                y1 = int(cy + r * np.sin(angle_start))
                x2 = int(cx + r * np.cos(angle_end))
                y2 = int(cy + r * np.sin(angle_end))

                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

            # Add label
            label_pos = (int(cx + r * 0.707), int(cy - r * 0.707))
            cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 2, cv2.LINE_AA)

    return image


def draw_reference_crosshair(image, cx, cy, size=50, color=(128, 128, 128)):
    """Draw a small crosshair at the principal point."""
    cv2.line(image, (int(cx - size), int(cy)), (int(cx + size), int(cy)), color, 2)
    cv2.line(image, (int(cx), int(cy - size)), (int(cx), int(cy + size)), color, 2)
    cv2.circle(image, (int(cx), int(cy)), 5, color, -1)


def add_legend(image, polar_thresholds=[30, 60, 90, 120], position='bottom-right'):
    """Add a legend showing circle colors and polar angles."""
    colors = [
        (0, 255, 0),      # Green
        (0, 165, 255),    # Orange
        (0, 0, 255),      # Red
        (255, 0, 255),    # Purple
    ]
    labels = [f'{t}°' for t in polar_thresholds]

    h, w = image.shape[:2]

    # Legend box position
    if position == 'bottom-right':
        x_start = w - 200
        y_start = h - 150
    else:
        x_start = 20
        y_start = 20

    # Draw legend background
    cv2.rectangle(image, (x_start - 10, y_start - 10),
                 (x_start + 150, y_start + len(labels) * 35 + 10),
                 (255, 255, 255), -1)
    cv2.rectangle(image, (x_start - 10, y_start - 10),
                 (x_start + 150, y_start + len(labels) * 35 + 10),
                 (0, 0, 0), 2)

    for i, (color, label) in enumerate(zip(colors[:len(labels)], labels)):
        y = y_start + i * 35
        cv2.circle(image, (x_start + 15, y + 10), 10, color, -1)
        cv2.putText(image, label, (x_start + 35, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return image


def process_single_image(input_path, output_path, fu, k, cu, cv,
                        polar_thresholds=[30, 60, 90, 120],
                        add_legend_flag=True):
    """Process a single image and add polar circles."""
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read {input_path}")
        return False

    # Draw polar circles
    image = draw_polar_circles(image, cu, cv, fu, k, polar_thresholds,
                              thickness=4, circle_style='solid')

    # Add crosshair at principal point
    draw_reference_crosshair(image, cu, cv)

    # Add legend if requested
    if add_legend_flag:
        image = add_legend(image, polar_thresholds)

    # Save result
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")
    return True


def create_montage_with_circles(input_dir, output_path, fu, k, cu, cv,
                               polar_thresholds=[30, 60, 90, 120],
                               cols=4):
    """Create a montage of images with polar circles overlaid."""
    import glob

    # Find all polar_bins30 PNG files
    pattern = os.path.join(input_dir, "*.png")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No PNG files found in {input_dir}")
        return False

    # Limit to first N images for the montage
    max_images = cols * 3  # 3 rows
    files = files[:max_images]

    images = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue

        # Resize for montage (make smaller)
        scale = 0.25
        img_small = cv2.resize(img, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

        # Draw circles on resized image
        cu_small = cu * scale
        cv_small = cv * scale
        draw_polar_circles(img_small, cu_small, cv_small, fu * scale, k,
                          polar_thresholds, thickness=2)
        draw_reference_crosshair(img_small, cu_small, cv_small, size=20)

        images.append(img_small)

    if not images:
        print("No valid images to create montage")
        return False

    # Create montage
    rows = []
    for i in range(0, len(images), cols):
        row = images[i:i+cols]
        # Pad with blank images if needed
        while len(row) < cols:
            row.append(np.zeros_like(images[0]))
        rows.append(np.hstack(row))

    montage = np.vstack(rows)

    # Add legend
    if len(montage.shape) == 2:
        montage = cv2.cvtColor(montage, cv2.COLOR_GRAY2BGR)
    montage = add_legend(montage, polar_thresholds)

    # Add title
    h, w = montage.shape[:2]
    title = "Polar Angle Boundaries (30°/60°/90°/120°)"
    cv2.putText(montage, title, (w//2 - 200, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, montage)
    print(f"Saved montage: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Add polar angle boundary circles to visualization images')
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Directory containing polar_bins30 PNG files')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory (default: input_dir + "_with_circles")')
    parser.add_argument('--fu', type=float, default=DS_INTRINSICS['fu'],
                       help='Focal length fu (default: from camera config)')
    parser.add_argument('--fv', type=float, default=DS_INTRINSICS['fv'],
                       help='Focal length fv (default: from camera config)')
    parser.add_argument('--cu', type=float, default=DS_INTRINSICS['cu'],
                       help='Principal point cu (default: from camera config)')
    parser.add_argument('--cv', type=float, default=DS_INTRINSICS['cv'],
                       help='Principal point cv (default: from camera config)')
    parser.add_argument('--k', type=float, default=DS_INTRINSICS['k'],
                       help='Distortion coefficient k (default: from camera config)')
    parser.add_argument('--polar-thresholds', '-t', type=int, nargs='+',
                       default=[30, 60, 90, 120],
                       help='Polar angle thresholds in degrees')
    parser.add_argument('--montage', action='store_true',
                       help='Create a montage instead of processing individual files')
    parser.add_argument('--montage-output', '-m',
                       help='Output path for montage')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base = os.path.basename(args.input_dir.rstrip('/'))
        parent = os.path.dirname(args.input_dir.rstrip('/'))
        output_dir = os.path.join(parent, base + '_with_circles')

    os.makedirs(output_dir, exist_ok=True)

    if args.montage:
        # Create montage
        montage_output = args.montage_output or os.path.join(output_dir, 'polar_circles_montage.png')
        create_montage_with_circles(args.input_dir, montage_output,
                                   args.fu, args.k, args.cu, args.cv,
                                   polar_thresholds=args.polar_thresholds)
    else:
        # Process individual files
        import glob
        pattern = os.path.join(args.input_dir, "*.png")
        files = sorted(glob.glob(pattern))

        for f in files:
            basename = os.path.basename(f)
            output_path = os.path.join(output_dir, basename)
            process_single_image(f, output_path, args.fu, args.k, args.cu, args.cv,
                                polar_thresholds=args.polar_thresholds)

        print(f"\nProcessed {len(files)} images")
        print(f"Output directory: {output_dir}")

        # Also create a montage
        montage_output = os.path.join(output_dir, 'polar_circles_montage.png')
        create_montage_with_circles(args.input_dir, montage_output,
                                   args.fu, args.k, args.cu, args.cv,
                                   polar_thresholds=args.polar_thresholds)


if __name__ == "__main__":
    main()
