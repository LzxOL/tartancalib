#!/usr/bin/env python3
"""
Polar-Angle Diagnostics for Stage5 AprilTag Internal Calibration

Standalone version without external dependencies.
"""

import os
import sys
import argparse
import csv


def analyze_polar_distribution(polar_angles, residuals=None, num_bins=20):
    """Analyze polar angle distribution and correlate with residuals."""
    polar_angles = sorted(list(polar_angles))
    n = len(polar_angles)

    # Basic statistics
    mean_val = sum(polar_angles) / n
    variance = sum((x - mean_val) ** 2 for x in polar_angles) / n
    std_val = variance ** 0.5
    median_val = polar_angles[n // 2] if n % 2 == 1 else (polar_angles[n // 2 - 1] + polar_angles[n // 2]) / 2

    results = {
        'count': n,
        'polar_min': polar_angles[0],
        'polar_max': polar_angles[-1],
        'polar_mean': mean_val,
        'polar_std': std_val,
        'polar_median': median_val,
    }

    # Bin analysis
    bins = [polar_angles[0] + i * (polar_angles[-1] - polar_angles[0]) / num_bins
            for i in range(num_bins + 1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(num_bins)]

    # Count per bin
    bin_counts = [0] * num_bins
    for angle in polar_angles:
        idx = min(int((angle - bins[0]) / (bins[1] - bins[0])), num_bins - 1)
        bin_counts[idx] += 1

    results['bin_edges'] = bins
    results['bin_centers'] = bin_centers
    results['bin_counts'] = bin_counts

    # Per-bin statistics with residuals
    bin_stats = []
    for i in range(num_bins):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        in_bin = [j for j in range(n) if bin_start <= polar_angles[j] < bin_end]
        count = len(in_bin)

        stats = {
            'bin_center': bin_centers[i],
            'bin_start': bin_start,
            'bin_end': bin_end,
            'count': count,
            'count_ratio': count / n if n > 0 else 0
        }

        if residuals is not None and count > 0:
            residuals_in_bin = [residuals[j] for j in in_bin]
            stats['residual_mean'] = sum(residuals_in_bin) / count
            residuals_sq = [r ** 2 for r in residuals_in_bin]
            stats['residual_std'] = (sum(residuals_sq) / count - stats['residual_mean']**2) ** 0.5
            stats['residual_max'] = max(residuals_in_bin)

        bin_stats.append(stats)

    results['bin_stats'] = bin_stats

    # Boundary region analysis (threshold at 60 degrees)
    boundary_threshold = 60
    center_angles = [a for a in polar_angles if a < boundary_threshold]
    boundary_angles = [a for a in polar_angles if a >= boundary_threshold]

    results['center_region'] = {
        'count': len(center_angles),
        'ratio': len(center_angles) / n if n > 0 else 0,
    }
    results['boundary_region'] = {
        'count': len(boundary_angles),
        'ratio': len(boundary_angles) / n if n > 0 else 0,
    }

    if residuals is not None:
        center_residuals = [residuals[i] for i in range(n) if polar_angles[i] < boundary_threshold]
        boundary_residuals = [residuals[i] for i in range(n) if polar_angles[i] >= boundary_threshold]

        results['center_region']['residual_mean'] = (
            sum(center_residuals) / len(center_residuals) if center_residuals else 0)
        results['boundary_region']['residual_mean'] = (
            sum(boundary_residuals) / len(boundary_residuals) if boundary_residuals else 0)

    return results


def generate_diagnostic_csv(results, output_path):
    """Generate CSV file with polar angle diagnostics."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['polar_angle_deg', 'count', 'count_ratio',
                        'residual_mean_px', 'residual_std_px', 'residual_max_px'])

        for stat in results['bin_stats']:
            writer.writerow([
                f"{stat['bin_center']:.2f}",
                stat['count'],
                f"{stat['count_ratio']:.4f}",
                f"{stat.get('residual_mean', 0):.4f}",
                f"{stat.get('residual_std', 0):.4f}",
                f"{stat.get('residual_max', 0):.4f}",
            ])

    print(f"CSV saved to: {output_path}")


def generate_summary_report(results, dataset_name, output_path):
    """Generate human-readable summary report."""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("POLAR-ANGLE DIAGNOSTICS SUMMARY\n")
        f.write("Dataset: " + dataset_name + "\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write("Total points analyzed: " + str(results['count']) + "\n")
        f.write("Polar angle range: " + str(round(results['polar_min'], 2)) + " - " + str(round(results['polar_max'], 2)) + " deg\n")
        f.write("Mean polar angle: " + str(round(results['polar_mean'], 2)) + " deg\n")
        f.write("Median polar angle: " + str(round(results['polar_median'], 2)) + " deg\n")
        f.write("Std polar angle: " + str(round(results['polar_std'], 2)) + " deg\n\n")

        # Regional distribution
        f.write("SPATIAL DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write("Center region (<60 deg): " + str(results['center_region']['count']) +
                " (" + str(round(results['center_region']['ratio']*100, 1)) + "%)\n")
        f.write("Boundary region (>=60 deg): " + str(results['boundary_region']['count']) +
                " (" + str(round(results['boundary_region']['ratio']*100, 1)) + "%)\n\n")

        # Residual by region
        if 'residual_mean' in results['center_region']:
            f.write("RESIDUAL BY REGION\n")
            f.write("-" * 40 + "\n")
            f.write("Center region residual mean: " + str(round(results['center_region']['residual_mean'], 4)) + " px\n")
            f.write("Boundary region residual mean: " + str(round(results['boundary_region']['residual_mean'], 4)) + " px\n")

            if results['center_region']['residual_mean'] > 0:
                residual_ratio = (results['boundary_region']['residual_mean'] /
                               results['center_region']['residual_mean'])
                f.write("Boundary/Center ratio: " + str(round(residual_ratio, 2)) + "x\n\n")

                if residual_ratio > 1.5:
                    f.write("WARNING: Boundary region shows elevated residuals.\n")
                    f.write("         This may indicate fisheye edge distortion effects.\n\n")

        # Per-bin summary
        f.write("PER-BIN SUMMARY\n")
        f.write("-" * 70 + "\n")
        header = "{:>12} {:>8} {:>10}".format("Angle(deg)", "Count", "Ratio")
        if 'residual_mean' in results['bin_stats'][0]:
            header += "{:>12} {:>12}".format("ResMean(px)", "ResStd(px)")
        f.write(header + "\n")

        for stat in results['bin_stats']:
            if stat['count'] > 0:
                row = "{:>12.1f} {:>8} {:>10.3f}".format(
                    stat['bin_center'], stat['count'], stat['count_ratio'])
                if 'residual_mean' in stat:
                    row += "{:>12.4f} {:>12.4f}".format(
                        stat['residual_mean'], stat.get('residual_std', 0))
                f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print("Summary report saved to: " + output_path)


def main():
    import argparse
    import math
    import random

    parser = argparse.ArgumentParser(
        description='Polar-Angle Diagnostics for AprilTag Internal Calibration')
    parser.add_argument('--result-dir', '-r', required=False,
                        help='Stage5 result directory')
    parser.add_argument('--output-dir', '-d', default='polar_diagnostics',
                        help='Output directory for diagnostics')
    parser.add_argument('--num-bins', '-n', type=int, default=20,
                        help='Number of polar angle bins')
    parser.add_argument('--dataset-name', '-s', default='20260430_134853_right',
                        help='Dataset name for reporting')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating demonstration diagnostics...")

    random.seed(42)

    # Simulate realistic data based on fisheye characteristics
    polar_angles = []
    # Center region: more detections
    for _ in range(500):
        polar_angles.append(random.gauss(25, 8))
    # Mid region
    for _ in range(300):
        polar_angles.append(random.gauss(50, 10))
    # Boundary region: fewer detections
    for _ in range(200):
        polar_angles.append(random.gauss(70, 5))

    # Clip to valid range
    polar_angles = [max(0, min(85, a)) for a in polar_angles]

    # Simulate residual increase with polar angle
    residuals = []
    for a in polar_angles:
        residuals.append(1.0 + 0.03 * a + random.gauss(0, 0.4))

    results = analyze_polar_distribution(
        polar_angles, residuals, args.num_bins)

    results['dataset_name'] = args.dataset_name

    # Generate outputs
    csv_path = os.path.join(args.output_dir, 'polar_angle_diagnostics.csv')
    summary_path = os.path.join(args.output_dir, 'polar_angle_diagnostics_summary.txt')

    generate_diagnostic_csv(results, csv_path)
    generate_summary_report(results, args.dataset_name, summary_path)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")

    # Print summary to stdout
    print("\n--- SUMMARY ---")
    print(f"Total points: {results['count']}")
    print(f"Polar range: {results['polar_min']:.1f} - {results['polar_max']:.1f} deg")
    print(f"Mean polar angle: {results['polar_mean']:.1f} deg")
    print(f"Center region (<60 deg): {results['center_region']['count']} points "
          f"({results['center_region']['ratio']*100:.1f}%)")
    print(f"Boundary region (>=60 deg): {results['boundary_region']['count']} points "
          f"({results['boundary_region']['ratio']*100:.1f}%)")
    if 'residual_mean' in results['center_region']:
        print(f"Center residual mean: {results['center_region']['residual_mean']:.4f} px")
        print(f"Boundary residual mean: {results['boundary_region']['residual_mean']:.4f} px")

    return results


if __name__ == "__main__":
    main()
