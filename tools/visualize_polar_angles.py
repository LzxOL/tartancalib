#!/usr/bin/env python3
"""
Visualize polar angles of observations on calibration images.

This script creates comprehensive visualizations of polar angle data:
1. Each observation point colored by polar angle
2. Polar angle distribution histogram
3. Polar angle vs residual scatter plot
4. Grid visualization showing polar angle zones

Usage:
    python visualize_polar_angles.py --input-dir /path/to/data --output-dir /path/to/output
"""

import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import glob

# Default camera intrinsics (Double Sphere model)
DEFAULT_INTRINSICS = {
    'k': -0.193,
    'fu': 1164.5,
    'fv': 1164.5,
    'cu': 2255.0,
    'cv': 2269.6,
}


def compute_distorted_radius(theta_deg, fu, k):
    """Compute the distorted image radius for a given polar angle using DS model."""
    theta_rad = np.deg2rad(theta_deg)
    tan_half = np.tan(theta_rad / 2.0)
    tan_sq = tan_half * tan_half
    r = 2.0 * fu * tan_half / (1.0 - k * tan_sq)
    return r


def compute_polar_angle_from_pixel(u, v, fu, fv, cu, cv, k):
    """
    Compute polar angle (angle between ray and optical axis) from pixel coordinates.
    
    For Double Sphere model:
    1. Convert pixel to normalized coordinates
    2. Apply undistortion to get Euclidean coordinates
    3. Compute polar angle from the ray
    
    Returns angle in degrees.
    """
    # Normalized coordinates (before distortion)
    mx = (u - cu) / fu
    my = (v - cv) / fv
    
    # For Double Sphere model, need to undistort
    # The DS model: r_d = r / (1 + k * r^2)
    # Inverse: r = compute_distorted_radius_inv(mx, my)
    
    r_sq = mx * mx + my * my
    r = np.sqrt(r_sq)
    
    if r < 1e-10:
        return 0.0
    
    # Approximate undistortion for DS model
    # m = (2 * tan(theta/2)) / (1 - k * tan^2(theta/2))
    # This is iterative, but for small k, we can approximate
    # For the inverse, we use the relationship:
    # theta = 2 * atan(r / (2*fu)) for small angles
    
    # More accurate: solve for tan(theta/2) from the forward model
    # tan_half = r / (2*fu + k*r)
    # Actually let's use a simpler approximation:
    
    # For stereographic with DS correction:
    # tan_half = m / (2 + k * m^2) where m = r / fu
    m = r
    tan_half = m / (2.0 + k * m * m) if abs(2.0 + k * m * m) > 1e-10 else m / 2.0
    
    theta_rad = 2.0 * np.arctan(tan_half)
    return np.rad2deg(theta_rad)


def create_polar_colormap():
    """Create a colormap for polar angles (blue=0, red=90+)."""
    colors = [
        (0.0, '#0000FF'),   # Blue for 0°
        (0.33, '#00FFFF'), # Cyan for 30°
        (0.5, '#00FF00'),  # Green for 45°
        (0.67, '#FFFF00'), # Yellow for 60°
        (0.83, '#FF8000'), # Orange for 75°
        (1.0, '#FF0000'),  # Red for 90°+
    ]
    return LinearSegmentedColormap.from_list('polar', colors)


def draw_polar_circles(image, cx, cy, fu, k, polar_thresholds=[30, 60, 90],
                       thickness=3, alpha=1.0):
    """Draw concentric circles at polar angle boundaries."""
    overlay = image.copy()
    
    colors = [
        (0, 255, 0),       # Green: 30°
        (0, 165, 255),     # Orange: 60°
        (0, 0, 255),       # Red: 90°
        (255, 0, 255),     # Purple: 120°
    ]
    
    labels = ['30°', '60°', '90°', '120°']
    
    for i, theta in enumerate(polar_thresholds):
        r = compute_distorted_radius(theta, fu, k)
        if r <= 0 or r > max(image.shape[1], image.shape[0]):
            continue
        
        color = colors[i] if i < len(colors) else (255, 255, 255)
        cv2.circle(overlay, (int(cx), int(cy)), int(round(r)), color, thickness)
        
        # Add label at upper-right
        label_pos = (int(cx + r * 0.707 - 20), int(cy - r * 0.707 + 10))
        cv2.putText(overlay, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, color, 2, cv2.LINE_AA)
    
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    else:
        cv2.addWeighted(overlay, 1, image, 0, 0, image)
    
    return image


def draw_crosshair(image, cx, cy, size=30, color=(128, 128, 128)):
    """Draw crosshair at principal point."""
    cv2.line(image, (int(cx - size), int(cy)), (int(cx + size), int(cy)), color, 2)
    cv2.line(image, (int(cx), int(cy - size)), (int(cx), int(cy + size)), color, 2)
    cv2.circle(image, (int(cx), int(cy)), 5, color, -1)


def visualize_image_with_polar_colors(image, observations, fu, fv, cu, cv, k,
                                       point_size=3, alpha_overlay=0.7):
    """
    Draw an image with observation points colored by polar angle.
    
    Args:
        image: Input image (will be modified)
        observations: List of (u, v, residual) tuples
        fu, fv, cu, cv, k: Camera intrinsics
        point_size: Size of observation points
        alpha_overlay: Transparency of polar color overlay
    
    Returns:
        Image with colored observation points
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Create colormap
    cmap = create_polar_colormap()
    
    for u, v, residual in observations:
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        
        # Compute polar angle
        polar_angle = compute_polar_angle_from_pixel(u, v, fu, fv, cu, cv, k)
        
        # Map polar angle to [0, 1] for colormap (0-90 degrees -> 0-1)
        t = min(polar_angle / 90.0, 1.0)
        
        # Get color from colormap
        color = cmap(t)
        bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        
        cv2.circle(result, (int(u), int(v)), point_size, bgr, -1)
    
    # Draw polar angle circles
    result = draw_polar_circles(result, cu, cv, fu, k, 
                                polar_thresholds=[30, 60, 90, 120], 
                                thickness=2)
    
    # Draw crosshair at principal point
    draw_crosshair(result, cu, cv)
    
    return result


def visualize_residuals_by_polar_angle(image, observations, fu, fv, cu, cv, k,
                                       residual_range=(0, 20), point_size=4):
    """
    Draw an image with observation points colored by residual magnitude.
    
    Args:
        image: Input image
        observations: List of (u, v, residual) tuples
        residual_range: (min, max) residual for color scaling
        point_size: Size of observation points
    
    Returns:
        Image with residual-colored observation points
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    res_min, res_max = residual_range
    
    for u, v, residual in observations:
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        
        # Normalize residual to [0, 1]
        t = (residual - res_min) / (res_max - res_min + 1e-10)
        t = max(0, min(1, t))
        
        # Blue (low) -> Green -> Yellow -> Red (high)
        if t < 0.25:
            color = (int(255 * t * 4), 255, 0)  # Blue to Cyan
        elif t < 0.5:
            color = (0, int(255 * (1 - (t - 0.25) * 4)), 255)  # Cyan to Green
        elif t < 0.75:
            color = (0, 255, int(255 * (1 - (t - 0.5) * 4)))  # Green to Yellow
        else:
            color = (0, int(255 * (1 - (t - 0.75) * 4)), 255)  # Yellow to Red
            color = (0, int(255 * (1 - (t - 0.75) * 4)), int(255 * (1 - (t - 0.75) * 4)))
            color = (0, 0, 255)
        
        cv2.circle(result, (int(u), int(v)), point_size, color, -1)
    
    return result


def create_polar_angle_grid_visualization(images, observations_list, fu, fv, cu, cv, k,
                                          grid_rows=2, grid_cols=4):
    """
    Create a grid visualization of multiple frames with polar coloring.
    """
    rows = grid_rows
    cols = grid_cols
    num_slots = rows * cols
    
    # Assume square images
    if len(images) > 0:
        img_h, img_w = images[0].shape[:2]
    else:
        return None
    
    scale = 0.3
    small_w = int(img_w * scale)
    small_h = int(img_h * scale)
    
    grid = np.zeros((small_h * rows, small_w * cols, 3), dtype=np.uint8)
    
    for idx, (img, obs) in enumerate(zip(images[:num_slots], observations_list[:num_slots])):
        row = idx // cols
        col = idx % cols
        
        # Resize and colorize
        small_img = cv2.resize(img, (small_w, small_h))
        small_cu = cu * scale
        small_cv = cv * scale
        small_fu = fu * scale
        small_fv = fv * scale
        
        vis = visualize_image_with_polar_colors(small_img, obs, small_fu, small_fv, 
                                               small_cu, small_cv, k, point_size=2)
        
        grid[row * small_h:(row + 1) * small_h, 
             col * small_w:(col + 1) * small_w] = vis
    
    return grid


def plot_polar_angle_distribution(polar_angles, output_path, bins=20):
    """
    Create histogram of polar angle distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(polar_angles, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Polar Angle (degrees)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Observation Polar Angles', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines at bin boundaries
    bin_edges = np.linspace(0, 90, bins + 1)
    for edge in [30, 60]:
        ax.axvline(x=edge, color='red', linestyle='--', alpha=0.5, label=f'{edge}°')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved polar angle distribution: {output_path}")


def plot_polar_angle_vs_residual(polar_angles, residuals, output_path, 
                                  point_types=None, show_fitted_line=True):
    """
    Create scatter plot of polar angle vs residual.
    
    Args:
        polar_angles: List of polar angles in degrees
        residuals: List of residual magnitudes
        point_types: Optional list of point types ('outer' or 'internal')
        show_fitted_line: Whether to show a smoothed trend line
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if point_types is not None:
        outer_mask = np.array(point_types) == 'outer'
        internal_mask = np.array(point_types) == 'internal'
        
        ax.scatter(np.array(polar_angles)[outer_mask], np.array(residuals)[outer_mask],
                   c='blue', alpha=0.5, label='Outer', s=20)
        ax.scatter(np.array(polar_angles)[internal_mask], np.array(residuals)[internal_mask],
                   c='green', alpha=0.5, label='Internal', s=20)
    else:
        scatter = ax.scatter(polar_angles, residuals, 
                            c=polar_angles, cmap='coolwarm', 
                            alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Polar Angle (deg)')
    
    ax.set_xlabel('Polar Angle (degrees)', fontsize=12)
    ax.set_ylabel('Residual (pixels)', fontsize=12)
    ax.set_title('Polar Angle vs Reprojection Residual', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if point_types is not None:
        ax.legend()
    
    # Add smoothed trend line
    if show_fitted_line and len(polar_angles) > 10:
        # Bin the data by polar angle
        angle_bins = np.linspace(0, 90, 10)
        bin_centers = []
        bin_means = []
        bin_stds = []
        
        for i in range(len(angle_bins) - 1):
            mask = (np.array(polar_angles) >= angle_bins[i]) & \
                   (np.array(polar_angles) < angle_bins[i + 1])
            if np.sum(mask) > 0:
                bin_centers.append((angle_bins[i] + angle_bins[i + 1]) / 2)
                bin_means.append(np.mean(np.array(residuals)[mask]))
                bin_stds.append(np.std(np.array(residuals)[mask]))
        
        if len(bin_centers) > 0:
            ax.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Mean trend')
            ax.fill_between(bin_centers, 
                           np.array(bin_means) - np.array(bin_stds),
                           np.array(bin_means) + np.array(bin_stds),
                           alpha=0.2, color='red')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved polar angle vs residual: {output_path}")


def plot_polar_zones_summary(polar_angles, residuals, bin_edges=[0, 30, 60, 90, 120],
                             output_path='polar_zones_summary.png'):
    """
    Create a summary plot showing statistics per polar angle zone.
    
    Shows: point count, mean residual, std residual per zone.
    """
    labels = [f'{bin_edges[i]}-{bin_edges[i+1]}°' for i in range(len(bin_edges)-1)]
    
    # Compute stats per bin
    counts = []
    means = []
    stds = []
    medians = []
    
    for i in range(len(bin_edges) - 1):
        mask = (np.array(polar_angles) >= bin_edges[i]) & \
               (np.array(polar_angles) < bin_edges[i + 1])
        vals = np.array(residuals)[mask]
        
        counts.append(np.sum(mask))
        means.append(np.mean(vals) if len(vals) > 0 else 0)
        stds.append(np.std(vals) if len(vals) > 0 else 0)
        medians.append(np.median(vals) if len(vals) > 0 else 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Count per zone
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    ax.bar(labels, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Observation Count per Polar Angle Zone')
    ax.tick_params(axis='x', rotation=45)
    
    # Mean residual per zone
    ax = axes[0, 1]
    ax.bar(labels, means, yerr=stds, color=colors, edgecolor='black', capsize=5)
    ax.set_ylabel('Mean Residual (px)')
    ax.set_title('Mean Residual per Polar Angle Zone')
    ax.tick_params(axis='x', rotation=45)
    
    # Median residual per zone
    ax = axes[1, 0]
    ax.bar(labels, medians, color=colors, edgecolor='black')
    ax.set_ylabel('Median Residual (px)')
    ax.set_title('Median Residual per Polar Angle Zone')
    ax.tick_params(axis='x', rotation=45)
    
    # Box plot
    ax = axes[1, 1]
    data = []
    for i in range(len(bin_edges) - 1):
        mask = (np.array(polar_angles) >= bin_edges[i]) & \
               (np.array(polar_angles) < bin_edges[i + 1])
        data.append(np.array(residuals)[mask])
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Residual (px)')
    ax.set_title('Residual Distribution per Polar Angle Zone')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved polar zones summary: {output_path}")


def create_polar_heatmap(image_shape, observations, fu, fv, cu, cv, k, 
                        cell_size=50, polar_range=(0, 90)):
    """
    Create a 2D heatmap showing point density in polar angle space.
    """
    h, w = image_shape[:2]
    grid_h = h // cell_size
    grid_w = w // cell_size
    
    # Accumulate counts and residuals
    density = np.zeros((grid_h, grid_w))
    residual_sum = np.zeros((grid_h, grid_w))
    
    for u, v, residual in observations:
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        
        gx = int(u // cell_size)
        gy = int(v // cell_size)
        if gx >= grid_w or gy >= grid_h:
            continue
        
        density[gy, gx] += 1
        residual_sum[gy, gx] += residual
    
    # Compute mean residual
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_residual = np.where(density > 0, residual_sum / density, np.nan)
    
    return density, mean_residual


def load_observations_from_backend_result(result_path):
    """
    Load observations from a backend calibration result file.
    
    Expected format: JSON with observations containing image_xy and residual info.
    """
    if not os.path.exists(result_path):
        return None
    
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        observations = []
        for obs in data.get('observations', []):
            u = obs.get('image_x', 0)
            v = obs.get('image_y', 0)
            residual = obs.get('residual_norm', 0)
            observations.append((u, v, residual))
        
        return observations
    except Exception as e:
        print(f"Error loading {result_path}: {e}")
        return None


def load_observations_from_csv(csv_path):
    """Load observations from a CSV file with polar angle data."""
    observations = []
    polar_angles = []
    residuals = []
    point_types = []
    
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return None
    
    # Parse header
    header = lines[0].strip().split(',')
    
    try:
        # Find column indices
        u_idx = next((i for i, h in enumerate(header) if 'image_x' in h.lower() or 'u' in h.lower()), 0)
        v_idx = next((i for i, h in enumerate(header) if 'image_y' in h.lower() or 'v' in h.lower()), 1)
        res_idx = next((i for i, h in enumerate(header) if 'residual' in h.lower()), 2)
        polar_idx = next((i for i, h in enumerate(header) if 'polar' in h.lower()), -1)
        type_idx = next((i for i, h in enumerate(header) if 'type' in h.lower() or 'point_type' in h.lower()), -1)
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < max(u_idx, v_idx, res_idx) + 1:
                continue
            
            try:
                u = float(parts[u_idx])
                v = float(parts[v_idx])
                res = float(parts[res_idx])
                
                observations.append((u, v, res))
                residuals.append(res)
                
                if polar_idx >= 0 and polar_idx < len(parts):
                    polar_angles.append(float(parts[polar_idx]))
                
                if type_idx >= 0 and type_idx < len(parts):
                    point_types.append(parts[type_idx])
            except ValueError:
                continue
        
        return {
            'observations': observations,
            'polar_angles': polar_angles if polar_angles else None,
            'residuals': residuals,
            'point_types': point_types if point_types else None
        }
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize polar angles of calibration observations')
    
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Directory containing images and observation data')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory (default: input_dir/polar_visualization)')
    parser.add_argument('--images', nargs='+',
                       help='Specific image files to process')
    parser.add_argument('--observations', 
                       help='CSV or JSON file with observations data')
    parser.add_argument('--fu', type=float, default=DEFAULT_INTRINSICS['fu'],
                       help='Focal length fu')
    parser.add_argument('--fv', type=float, default=DEFAULT_INTRINSICS['fv'],
                       help='Focal length fv')
    parser.add_argument('--cu', type=float, default=DEFAULT_INTRINSICS['cu'],
                       help='Principal point cu')
    parser.add_argument('--cv', type=float, default=DEFAULT_INTRINSICS['cv'],
                       help='Principal point cv')
    parser.add_argument('--k', type=float, default=DEFAULT_INTRINSICS['k'],
                       help='DS distortion coefficient k')
    parser.add_argument('--polar-thresholds', type=int, nargs='+',
                       default=[30, 60, 90, 120],
                       help='Polar angle thresholds for circles')
    parser.add_argument('--colormap', choices=['polar', 'residual', 'both'], 
                       default='polar',
                       help='Type of colormap to use')
    parser.add_argument('--grid-rows', type=int, default=2,
                       help='Number of rows in grid visualization')
    parser.add_argument('--grid-cols', type=int, default=4,
                       help='Number of columns in grid visualization')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.input_dir, 'polar_visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load observations
    obs_data = None
    if args.observations:
        if args.observations.endswith('.csv'):
            obs_data = load_observations_from_csv(args.observations)
        else:
            obs_data = load_observations_from_backend_result(args.observations)
    
    if obs_data is None:
        print("No valid observation data found. Creating synthetic test data...")
        # Create synthetic test data for demonstration
        w, h = 4512, 4512
        observations = []
        for i in range(100):
            u = np.random.uniform(w * 0.2, w * 0.8)
            v = np.random.uniform(h * 0.2, h * 0.8)
            residual = np.random.exponential(2.0)
            observations.append((u, v, residual))
    else:
        observations = obs_data.get('observations', [])
        polar_angles = obs_data.get('polar_angles')
        residuals = obs_data.get('residuals')
        point_types = obs_data.get('point_types')
    
    # Compute polar angles if not provided
    if polar_angles is None:
        polar_angles = [
            compute_polar_angle_from_pixel(u, v, args.fu, args.fv, 
                                          args.cu, args.cv, args.k)
            for u, v, _ in observations
        ]
    
    # Create visualizations
    if args.colormap in ['polar', 'both']:
        # Plot polar angle distribution
        plot_polar_angle_distribution(polar_angles, 
                                      os.path.join(output_dir, 'polar_angle_distribution.png'))
        
        # Plot polar angle vs residual
        plot_polar_angle_vs_residual(polar_angles, residuals if residuals else [o[2] for o in observations],
                                     os.path.join(output_dir, 'polar_vs_residual.png'),
                                     point_types=point_types)
        
        # Plot zones summary
        plot_polar_zones_summary(polar_angles, 
                                residuals if residuals else [o[2] for o in observations],
                                bin_edges=[0, 30, 60, 90, 120],
                                output_path=os.path.join(output_dir, 'polar_zones_summary.png'))
    
    # Process images
    if args.images:
        image_files = args.images
    else:
        # Find images in input directory
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))
        image_files = sorted(image_files)
    
    if image_files:
        for img_path in image_files[:8]:  # Limit to 8 images
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Use subset of observations for each image (simulate frame association)
            # In real usage, you'd filter observations by frame
            vis = visualize_image_with_polar_colors(img, observations[:50], 
                                                   args.fu, args.fv, args.cu, args.cv, args.k)
            
            basename = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f'{basename}_polar_visualized.png')
            cv2.imwrite(out_path, vis)
            print(f"Saved: {out_path}")
        
        # Create grid visualization
        images = [cv2.imread(f) for f in image_files[:args.grid_rows * args.grid_cols] 
                  if cv2.imread(f) is not None]
        
        if images:
            grid = create_polar_angle_grid_visualization(
                images, [observations[:50]] * len(images),
                args.fu, args.fv, args.cu, args.cv, args.k,
                grid_rows=args.grid_rows, grid_cols=args.grid_cols)
            
            if grid is not None:
                grid_path = os.path.join(output_dir, 'polar_grid_visualization.png')
                cv2.imwrite(grid_path, grid)
                print(f"Saved: {grid_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("POLAR ANGLE SUMMARY")
    print("=" * 60)
    polar_arr = np.array(polar_angles)
    print(f"Total observations: {len(polar_angles)}")
    print(f"Polar angle range: {polar_arr.min():.1f}° - {polar_arr.max():.1f}°")
    print(f"Polar angle mean: {polar_arr.mean():.1f}°")
    print(f"Polar angle std: {polar_arr.std():.1f}°")
    
    # Count per zone
    for i in range(len(args.polar_thresholds)):
        lower = args.polar_thresholds[i-1] if i > 0 else 0
        upper = args.polar_thresholds[i]
        count = np.sum((polar_arr >= lower) & (polar_arr < upper))
        pct = 100 * count / len(polar_arr)
        print(f"  {lower:3d}° - {upper:3d}°: {count:4d} ({pct:5.1f}%)")
    
    print("=" * 60)
    print(f"\nVisualization complete! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
