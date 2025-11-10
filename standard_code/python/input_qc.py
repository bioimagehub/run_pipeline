"""
Input QC - Quality Control for Microscopy Input Images

Validates microscopy images against quality standards:
- Pixel size consistency across dataset
- Intensity dynamic range
- Saturation detection
- Dimension consistency
- Metadata completeness
- Focus quality metrics

Outputs results to QC_summary.tsv and optional HTML report.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import sys
import argparse
import logging
import yaml
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import numpy as np
from scipy import ndimage
import base64
from io import BytesIO

# Local imports
try:
    import bioimage_pipeline_utils as rp
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


class QCTest(ABC):
    """Base class for QC tests"""
    
    def __init__(self, name: str):
        """
        Args:
            name: QC test name (e.g., "QC_pixelsize")
        """
        self.name = name
    
    @abstractmethod
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """
        Run the QC test.
        
        Args:
            image_path: Path to the image file
            metadata: Metadata dict from YAML file
            context: Shared context across all files (e.g., mode pixel size)
        
        Returns:
            Tuple of (status, comment)
            status: "pass", "warn", or "fail"
            comment: detailed explanation
        """
        pass
    
    @abstractmethod
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """
        Prepare shared context by analyzing all files.
        Called once before running checks on individual files.
        
        Args:
            grouped_files: Dict mapping basename to {'input': path, 'yaml': path}
        
        Returns:
            Context dict to be passed to check()
        """
        pass


class PixelSizeConsistencyTest(QCTest):
    """Check that all images have consistent physical pixel sizes"""
    
    def __init__(self):
        super().__init__("QC_pixelsize")
    
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """Find the most common pixel size across all files"""
        pixel_sizes = []
        
        for basename, files in grouped_files.items():
            yaml_path = files.get('yaml')
            if not yaml_path or not os.path.exists(yaml_path):
                continue
            
            try:
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                phys_dims = metadata.get('Image metadata', {}).get('Physical dimensions', {})
                x_um = phys_dims.get('X_um')
                y_um = phys_dims.get('Y_um')
                z_um = phys_dims.get('Z_um')
                
                if x_um is not None and y_um is not None and z_um is not None:
                    # Round to avoid floating point comparison issues
                    pixel_sizes.append((round(x_um, 6), round(y_um, 6), round(z_um, 6)))
            
            except Exception as e:
                logger.warning(f"Could not read pixel size from {yaml_path}: {e}")
        
        if not pixel_sizes:
            return {'mode_pixel_size': None}
        
        # Find most common pixel size
        counter = Counter(pixel_sizes)
        mode_pixel_size = counter.most_common(1)[0][0]
        
        logger.info(f"Most common pixel size (X, Y, Z μm): {mode_pixel_size}")
        logger.info(f"Found in {counter[mode_pixel_size]}/{len(pixel_sizes)} files")
        
        return {'mode_pixel_size': mode_pixel_size}
    
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """Check if image has the mode pixel size"""
        mode_pixel_size = context.get('mode_pixel_size')
        
        if mode_pixel_size is None:
            return ("warn", "No mode pixel size available")
        
        try:
            phys_dims = metadata.get('Image metadata', {}).get('Physical dimensions', {})
            x_um = phys_dims.get('X_um')
            y_um = phys_dims.get('Y_um')
            z_um = phys_dims.get('Z_um')
            
            if x_um is None or y_um is None or z_um is None:
                return ("fail", "Missing pixel size in metadata")
            
            actual = (round(x_um, 6), round(y_um, 6), round(z_um, 6))
            
            if actual != mode_pixel_size:
                return ("fail", f"Pixel size {actual} != mode {mode_pixel_size}")
            
            return ("pass", "")
        
        except Exception as e:
            return ("fail", f"Error reading pixel size: {e}")


class IntensityRangeTest(QCTest):
    """Check that images have sufficient dynamic range"""
    
    def __init__(self, percentile: float = 5.0, ratio_threshold: float = 1.5):
        """
        Args:
            percentile: Percentile to use for high/low comparison (default 5%)
            ratio_threshold: Minimum ratio of high to low percentiles (default 1.5)
        """
        super().__init__("QC_intensity")
        self.percentile = percentile
        self.ratio_threshold = ratio_threshold
    
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """No shared context needed for intensity test"""
        return {}
    
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """Check intensity dynamic range per channel"""
        try:
            img = rp.load_tczyx_image(image_path)
            data = img.data  # TCZYX
            
            T, C, Z, Y, X = data.shape
            
            # Check each channel separately
            failed_channels = []
            channel_details = []
            
            for c in range(C):
                channel_data = data[:, c, :, :, :].flatten()
                
                # Remove zeros from calculation (background)
                nonzero_data = channel_data[channel_data > 0]
                
                if len(nonzero_data) == 0:
                    failed_channels.append(c)
                    channel_details.append(f"C{c}: all zeros")
                    continue
                
                low_percentile = np.percentile(nonzero_data, self.percentile)
                high_percentile = np.percentile(nonzero_data, 100 - self.percentile)
                
                if low_percentile == 0:
                    ratio = float('inf') if high_percentile > 0 else 0
                else:
                    ratio = high_percentile / low_percentile
                
                if ratio < self.ratio_threshold:
                    failed_channels.append(c)
                    channel_details.append(f"C{c}: ratio={ratio:.2f}")
                else:
                    channel_details.append(f"C{c}: ratio={ratio:.2f} OK")
            
            if failed_channels:
                details = "; ".join(channel_details)
                return ("fail", f"Low dynamic range (threshold={self.ratio_threshold}): {details}")
            
            return ("pass", "")
        
        except Exception as e:
            return ("fail", f"Error reading image: {e}")


class SaturationTest(QCTest):
    """Detect saturated pixels in images"""
    
    def __init__(self, saturation_threshold: float = 1.0):
        """
        Args:
            saturation_threshold: Max % of pixels allowed at max intensity (default 1%)
        """
        super().__init__("QC_saturation")
        self.saturation_threshold = saturation_threshold
    
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """No shared context needed"""
        return {}
    
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """Check for saturated pixels per channel"""
        try:
            img = rp.load_tczyx_image(image_path)
            data = img.data  # TCZYX
            
            T, C, Z, Y, X = data.shape
            
            # Determine max value based on dtype
            if data.dtype == np.uint8:
                max_val = 255
            elif data.dtype == np.uint16:
                max_val = 65535
            else:
                # For float, assume normalized to 1.0
                max_val = 1.0
            
            saturated_channels = []
            channel_details = []
            
            for c in range(C):
                channel_data = data[:, c, :, :, :]
                total_pixels = channel_data.size
                saturated_pixels = np.sum(channel_data >= max_val)
                saturation_pct = (saturated_pixels / total_pixels) * 100
                
                if saturation_pct > self.saturation_threshold:
                    saturated_channels.append(c)
                    channel_details.append(f"C{c}: {saturation_pct:.1f}%")
                else:
                    channel_details.append(f"C{c}: {saturation_pct:.1f}% OK")
            
            if saturated_channels:
                details = "; ".join(channel_details)
                return ("fail", f"Saturation detected (threshold={self.saturation_threshold}%): {details}")
            
            return ("pass", "")
        
        except Exception as e:
            return ("fail", f"Error reading image: {e}")


class DimensionConsistencyTest(QCTest):
    """Check that all images have consistent TCZYX dimensions"""
    
    def __init__(self):
        super().__init__("QC_dimensions")
    
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """Find the most common dimensions"""
        dimensions = []
        
        for basename, files in grouped_files.items():
            yaml_path = files.get('yaml')
            if not yaml_path or not os.path.exists(yaml_path):
                continue
            
            try:
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                dims = metadata.get('Image metadata', {}).get('Image dimensions', {})
                t = dims.get('T', 1)
                c = dims.get('C', 1)
                z = dims.get('Z', 1)
                y = dims.get('Y', 0)
                x = dims.get('X', 0)
                
                dimensions.append((t, c, z, y, x))
            
            except Exception as e:
                logger.warning(f"Could not read dimensions from {yaml_path}: {e}")
        
        if not dimensions:
            return {'mode_dimensions': None}
        
        counter = Counter(dimensions)
        mode_dims = counter.most_common(1)[0][0]
        
        logger.info(f"Most common dimensions (T, C, Z, Y, X): {mode_dims}")
        logger.info(f"Found in {counter[mode_dims]}/{len(dimensions)} files")
        
        return {'mode_dimensions': mode_dims}
    
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """Check if image has the mode dimensions"""
        mode_dims = context.get('mode_dimensions')
        
        if mode_dims is None:
            return ("warn", "No mode dimensions available")
        
        try:
            dims = metadata.get('Image metadata', {}).get('Image dimensions', {})
            t = dims.get('T', 1)
            c = dims.get('C', 1)
            z = dims.get('Z', 1)
            y = dims.get('Y', 0)
            x = dims.get('X', 0)
            
            actual = (t, c, z, y, x)
            
            if actual != mode_dims:
                # Only warn if different, not fail (might be intentional)
                return ("warn", f"Dimensions {actual} != mode {mode_dims}")
            
            return ("pass", "")
        
        except Exception as e:
            return ("fail", f"Error reading dimensions: {e}")


class MetadataCompletenessTest(QCTest):
    """Check that metadata contains all required fields"""
    
    def __init__(self):
        super().__init__("QC_metadata")
    
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """No shared context needed"""
        return {}
    
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """Check for required metadata fields"""
        missing = []
        warnings = []
        
        try:
            # Check required top-level sections
            if 'Image metadata' not in metadata:
                return ("fail", "Missing 'Image metadata' section")
            
            img_meta = metadata['Image metadata']
            
            # Check physical dimensions
            if 'Physical dimensions' not in img_meta:
                missing.append("Physical dimensions")
            else:
                phys_dims = img_meta['Physical dimensions']
                for dim in ['X_um', 'Y_um', 'Z_um']:
                    if dim not in phys_dims or phys_dims[dim] is None:
                        missing.append(f"Physical dimensions.{dim}")
            
            # Check image dimensions
            if 'Image dimensions' not in img_meta:
                missing.append("Image dimensions")
            else:
                dims = img_meta['Image dimensions']
                for dim in ['T', 'C', 'Z', 'Y', 'X']:
                    if dim not in dims or dims[dim] is None:
                        missing.append(f"Image dimensions.{dim}")
            
            # Check channels (warning only)
            if 'Channels' not in img_meta:
                warnings.append("No channel names")
            else:
                channels = img_meta['Channels']
                if not isinstance(channels, list) or len(channels) == 0:
                    warnings.append("Empty channel list")
                else:
                    for i, ch in enumerate(channels):
                        if 'Name' not in ch or 'Please fill in' in ch.get('Name', ''):
                            warnings.append(f"Channel {i} has placeholder name")
            
            if missing:
                return ("fail", f"Missing fields: {', '.join(missing)}")
            
            if warnings:
                return ("warn", f"Incomplete data: {', '.join(warnings)}")
            
            return ("pass", "")
        
        except Exception as e:
            return ("fail", f"Error reading metadata: {e}")


class FocusQualityTest(QCTest):
    """Measure focus quality using variance of Laplacian"""
    
    def __init__(self, threshold: float = 100.0):
        """
        Args:
            threshold: Minimum variance of Laplacian (lower = more blurry)
        """
        super().__init__("QC_focus")
        self.threshold = threshold
    
    def prepare_context(self, grouped_files: Dict[str, Dict[str, str]]) -> Dict:
        """Calculate median focus score across all files for threshold"""
        focus_scores = []
        
        for basename, files in grouped_files.items():
            input_path = files.get('input')
            if not input_path or not os.path.exists(input_path):
                continue
            
            try:
                img = rp.load_tczyx_image(input_path)
                data = img.data  # TCZYX
                
                # Take middle timepoint, middle z-slice
                T, C, Z, Y, X = data.shape
                mid_t = T // 2
                mid_z = Z // 2
                
                for c in range(C):
                    slice_2d = data[mid_t, c, mid_z, :, :]
                    
                    # Normalize to 0-1
                    if slice_2d.max() > 0:
                        slice_2d = slice_2d.astype(float) / slice_2d.max()
                    
                    # Calculate variance of Laplacian
                    laplacian = ndimage.laplace(slice_2d)
                    focus_score = laplacian.var()
                    focus_scores.append(focus_score)
            
            except Exception as e:
                logger.warning(f"Could not calculate focus for {input_path}: {e}")
        
        if focus_scores:
            median_focus = np.median(focus_scores)
            # Use 20% of median as threshold
            adaptive_threshold = median_focus * 0.2
            logger.info(f"Median focus score: {median_focus:.2f}, adaptive threshold: {adaptive_threshold:.2f}")
            return {'focus_threshold': adaptive_threshold}
        
        return {'focus_threshold': self.threshold}
    
    def check(self, image_path: str, metadata: Dict, context: Dict) -> Tuple[str, str]:
        """Check focus quality per channel"""
        threshold = context.get('focus_threshold', self.threshold)
        
        try:
            img = rp.load_tczyx_image(image_path)
            data = img.data  # TCZYX
            
            T, C, Z, Y, X = data.shape
            
            # Check middle timepoint, all Z slices
            mid_t = T // 2
            
            blurry_channels = []
            channel_details = []
            
            for c in range(C):
                # Calculate focus for each Z slice
                focus_scores = []
                for z in range(Z):
                    slice_2d = data[mid_t, c, z, :, :]
                    
                    # Normalize
                    if slice_2d.max() > 0:
                        slice_2d = slice_2d.astype(float) / slice_2d.max()
                    
                    laplacian = ndimage.laplace(slice_2d)
                    focus_score = laplacian.var()
                    focus_scores.append(focus_score)
                
                # Use max focus score across Z (best focused plane)
                max_focus = max(focus_scores) if focus_scores else 0
                
                if max_focus < threshold:
                    blurry_channels.append(c)
                    channel_details.append(f"C{c}: {max_focus:.1f}")
                else:
                    channel_details.append(f"C{c}: {max_focus:.1f} OK")
            
            if blurry_channels:
                details = "; ".join(channel_details)
                return ("warn", f"Low focus quality (threshold={threshold:.1f}): {details}")
            
            return ("pass", "")
        
        except Exception as e:
            return ("fail", f"Error calculating focus: {e}")


def generate_html_report(
    grouped_files: Dict[str, Dict[str, str]],
    results: Dict[str, Dict[str, Tuple[str, str]]],
    tests: List[QCTest],
    output_path: str
) -> None:
    """
    Generate HTML report with visualizations and thumbnails.
    
    Args:
        grouped_files: Dict mapping basename to file paths
        results: QC test results
        tests: List of QC tests
        output_path: Path to save HTML report
    """
    from matplotlib import pyplot as plt
    
    html_parts = []
    
    # HTML header
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>QC Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        .summary { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .summary-box { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }
        .summary-box h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }
        .summary-box .value { font-size: 32px; font-weight: bold; color: #333; }
        table { border-collapse: collapse; width: 100%; background: white; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background: #4CAF50; color: white; padding: 12px; text-align: left; position: sticky; top: 0; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .pass { color: #4CAF50; font-weight: bold; }
        .warn { color: #FF9800; font-weight: bold; }
        .fail { color: #F44336; font-weight: bold; }
        .thumbnail { max-width: 200px; max-height: 150px; margin: 5px; border: 1px solid #ddd; }
        .file-section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .file-section.failed { border-left: 5px solid #F44336; }
        .file-section.warned { border-left: 5px solid #FF9800; }
        .file-section.passed { border-left: 5px solid #4CAF50; }
        .test-result { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 90%; }
    </style>
</head>
<body>
    <h1>Input QC Report</h1>
""")
    
    # Summary statistics
    total_files = len(results)
    passed_files = sum(1 for r in results.values() if all(s[0] == "pass" for s in r.values()))
    warned_files = sum(1 for r in results.values() if any(s[0] == "warn" for s in r.values()) and all(s[0] != "fail" for s in r.values()))
    failed_files = sum(1 for r in results.values() if any(s[0] == "fail" for s in r.values()))
    
    html_parts.append(f"""
    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-box">
                <h3>Total Files</h3>
                <div class="value">{total_files}</div>
            </div>
            <div class="summary-box">
                <h3>Passed</h3>
                <div class="value" style="color: #4CAF50;">{passed_files}</div>
            </div>
            <div class="summary-box">
                <h3>Warnings</h3>
                <div class="value" style="color: #FF9800;">{warned_files}</div>
            </div>
            <div class="summary-box">
                <h3>Failed</h3>
                <div class="value" style="color: #F44336;">{failed_files}</div>
            </div>
        </div>
    </div>
""")
    
    # Summary table
    html_parts.append("""
    <div class="summary">
        <h2>Results Table</h2>
        <table>
            <tr>
                <th>File</th>
""")
    
    for test in tests:
        html_parts.append(f"                <th>{test.name}</th>\n")
    
    html_parts.append("            </tr>\n")
    
    for basename in sorted(results.keys()):
        file_results = results[basename]
        
        # Determine overall status
        overall_status = "pass"
        if any(s[0] == "fail" for s in file_results.values()):
            overall_status = "fail"
        elif any(s[0] == "warn" for s in file_results.values()):
            overall_status = "warn"
        
        html_parts.append(f"            <tr class='{overall_status}'>\n")
        html_parts.append(f"                <td><code>{basename}</code></td>\n")
        
        for test in tests:
            status, comment = file_results[test.name]
            status_class = status
            display_text = status.upper() if status != "pass" else "✓"
            if comment and status != "pass":
                display_text = f"{status.upper()}: {comment}"
            html_parts.append(f"                <td class='{status_class}'>{display_text}</td>\n")
        
        html_parts.append("            </tr>\n")
    
    html_parts.append("""
        </table>
    </div>
""")
    
    # Detailed per-file sections (failed and warned files only)
    html_parts.append("""
    <div class="summary">
        <h2>Detailed Results (Failed & Warned Files)</h2>
""")
    
    for basename in sorted(results.keys()):
        file_results = results[basename]
        files = grouped_files[basename]
        input_path = files.get('input')
        
        # Skip passed files
        if all(s[0] == "pass" for s in file_results.values()):
            continue
        
        # Determine overall status
        overall_status = "pass"
        if any(s[0] == "fail" for s in file_results.values()):
            overall_status = "failed"
        elif any(s[0] == "warn" for s in file_results.values()):
            overall_status = "warned"
        
        html_parts.append(f"""
        <div class="file-section {overall_status}">
            <h3>{basename}</h3>
            <p><strong>File:</strong> <code>{input_path if input_path else 'N/A'}</code></p>
""")
        
        # Generate thumbnail if image exists
        if input_path and os.path.exists(input_path):
            try:
                img = rp.load_tczyx_image(input_path)
                data = img.data  # TCZYX
                
                # Create MIP of middle timepoint
                mid_t = data.shape[0] // 2
                mip = np.max(data[mid_t, :, :, :, :], axis=1)  # Max Z projection -> CYX
                
                # Create thumbnail
                fig, axes = plt.subplots(1, min(3, mip.shape[0]), figsize=(min(12, 4 * mip.shape[0]), 4))
                if mip.shape[0] == 1:
                    axes = [axes]
                
                for c in range(min(3, mip.shape[0])):
                    ax = axes[c] if len(axes) > 1 else axes[0]
                    img_data = mip[c, :, :]
                    # Normalize
                    if img_data.max() > 0:
                        img_data = img_data.astype(float) / img_data.max()
                    ax.imshow(img_data, cmap='gray')
                    ax.set_title(f'Channel {c}')
                    ax.axis('off')
                
                plt.tight_layout()
                
                # Save to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight')
                plt.close(fig)
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                
                html_parts.append(f"""
            <img src="data:image/png;base64,{img_base64}" class="thumbnail">
""")
            except Exception as e:
                html_parts.append(f"""
            <p><em>Could not generate thumbnail: {e}</em></p>
""")
        
        # Test results
        for test in tests:
            status, comment = file_results[test.name]
            if status != "pass":
                html_parts.append(f"""
            <div class="test-result">
                <strong>{test.name}:</strong> <span class="{status}">{status.upper()}</span>
                {f"<br>{comment}" if comment else ""}
            </div>
""")
        
        html_parts.append("""
        </div>
""")
    
    html_parts.append("""
    </div>
</body>
</html>
""")
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    logger.info(f"HTML report saved to {output_path}")


def run_qc_tests(
    grouped_files: Dict[str, Dict[str, str]],
    tests: List[QCTest],
    output_summary: str,
    fail_on_error: bool = False,
    html_report_path: Optional[str] = None
) -> bool:
    """
    Run all QC tests on grouped files and save results.
    
    Args:
        grouped_files: Dict mapping basename to {'input': path, 'yaml': path}
        tests: List of QCTest instances to run
        output_summary: Path to output TSV file
        fail_on_error: If True, raise exception on any QC failure
    
    Returns:
        True if all files passed all tests, False otherwise
    """
    logger.info(f"Running {len(tests)} QC tests on {len(grouped_files)} files...")
    
    # Prepare contexts for all tests
    contexts = {}
    for test in tests:
        logger.info(f"Preparing context for {test.name}...")
        contexts[test.name] = test.prepare_context(grouped_files)
    
    # Run tests on each file
    results = {}  # basename -> {test_name: (status, comment)}
    
    for basename, files in sorted(grouped_files.items()):
        input_path = files.get('input')
        yaml_path = files.get('yaml')
        
        logger.info(f"Testing {basename}...")
        
        # Load metadata
        metadata = {}
        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Could not load metadata from {yaml_path}: {e}")
        
        # Run all tests
        results[basename] = {}
        for test in tests:
            if not input_path or not os.path.exists(input_path):
                status, comment = ("fail", "Image file not found")
            else:
                status, comment = test.check(input_path, metadata, contexts[test.name])
            
            results[basename][test.name] = (status, comment)
            
            if status != "pass":
                logger.warning(f"  {test.name}: {status.upper()} - {comment}")
    
    # Write TSV output
    logger.info(f"Writing results to {output_summary}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_summary) or '.', exist_ok=True)
    
    with open(output_summary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Header
        header = ['basename'] + [test.name for test in tests]
        writer.writerow(header)
        
        # Data rows
        all_passed = True
        for basename in sorted(results.keys()):
            row = [basename]
            for test in tests:
                status, comment = results[basename][test.name]
                # Empty cell for pass, status for warn/fail
                if status == "pass":
                    row.append("")
                else:
                    row.append(f"{status}: {comment}" if comment else status)
                    if status == "fail":
                        all_passed = False
            
            writer.writerow(row)
    
    # Count results
    total_files = len(results)
    passed_files = sum(1 for r in results.values() if all(s[0] == "pass" for s in r.values()))
    warned_files = sum(1 for r in results.values() if any(s[0] == "warn" for s in r.values()) and all(s[0] != "fail" for s in r.values()))
    failed_files = sum(1 for r in results.values() if any(s[0] == "fail" for s in r.values()))
    
    logger.info(f"\nQC Results Summary:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Passed: {passed_files}")
    logger.info(f"  Warnings: {warned_files}")
    logger.info(f"  Failed: {failed_files}")
    
    # Generate HTML report if requested
    if html_report_path:
        generate_html_report(grouped_files, results, tests, html_report_path)
    
    if fail_on_error and not all_passed:
        raise RuntimeError(f"QC failed for {failed_files} file(s)")
    
    return all_passed





def main():
    parser = argparse.ArgumentParser(
        description='Input QC - Validate microscopy input images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Basic QC (pixel size and intensity)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/input_qc.py'
  - --input-search-pattern: '%YAML%/input/*.tif'
  - --yaml-search-pattern: '%YAML%/input/*_metadata.yaml'
  - --output-summary: '%YAML%/QC_summary.tsv'

- name: Full QC with HTML report
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/input_qc.py'
  - --input-search-pattern: '%YAML%/input/**/*.tif'
  - --yaml-search-pattern: '%YAML%/input/**/*_metadata.yaml'
  - --output-summary: '%YAML%/QC_summary.tsv'
  - --generate-report: '%YAML%/QC_report.html'

- name: QC with custom thresholds and fail on error
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/input_qc.py'
  - --input-search-pattern: '%YAML%/input/**/*.tif'
  - --yaml-search-pattern: '%YAML%/input/**/*_metadata.yaml'
  - --output-summary: '%YAML%/QC_summary.tsv'
  - --intensity-percentile: 10.0
  - --intensity-ratio: 2.0
  - --saturation-threshold: 0.5
  - --fail-on-error

- name: Skip specific tests
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/input_qc.py'
  - --input-search-pattern: '%YAML%/input/**/*.tif'
  - --yaml-search-pattern: '%YAML%/input/**/*_metadata.yaml'
  - --output-summary: '%YAML%/QC_summary.tsv'
  - --skip-tests: QC_focus QC_saturation
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images (e.g., "./input/*.tif")')
    parser.add_argument('--yaml-search-pattern', type=str, required=True,
                       help='Glob pattern for metadata YAML files (e.g., "./input/*_metadata.yaml")')
    parser.add_argument('--output-summary', type=str, required=True,
                       help='Output TSV file path for QC results')
    parser.add_argument('--intensity-percentile', type=float, default=5.0,
                       help='Percentile for intensity range test (default: 5.0)')
    parser.add_argument('--intensity-ratio', type=float, default=1.5,
                       help='Minimum ratio for intensity range test (default: 1.5)')
    parser.add_argument('--saturation-threshold', type=float, default=1.0,
                       help='Max percentage of saturated pixels allowed (default: 1.0)')
    parser.add_argument('--focus-threshold', type=float, default=100.0,
                       help='Minimum variance of Laplacian for focus test (default: 100.0)')
    parser.add_argument('--fail-on-error', action='store_true',
                       help='Exit with error code if any file fails QC')
    parser.add_argument('--skip-tests', type=str, nargs='*', default=[],
                       help='Tests to skip (e.g., QC_focus QC_saturation)')
    parser.add_argument('--generate-report', type=str, default=None,
                       help='Generate HTML report at specified path (e.g., QC_report.html)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get grouped files using standard utility function
    logger.info(f"Searching for files:")
    logger.info(f"  Input: {args.input_search_pattern}")
    logger.info(f"  YAML:  {args.yaml_search_pattern}")
    
    # Use standard grouping function from bioimage_pipeline_utils
    # Patterns with ** automatically search recursively
    search_patterns = {
        'input': args.input_search_pattern,
        'yaml': args.yaml_search_pattern
    }
    
    grouped_files = rp.get_grouped_files_to_process(
        search_patterns=search_patterns,
        search_subfolders=True  # Always True since ** patterns handle recursion
    )
    
    if not grouped_files:
        raise FileNotFoundError(f"No files found matching patterns")
    
    # Filter out groups where 'input' is missing or is actually a YAML file
    # (This can happen when YAML pattern creates separate groups)
    filtered_files = {}
    for basename, files in grouped_files.items():
        input_path = files.get('input')
        if input_path and not input_path.endswith('.yaml'):
            filtered_files[basename] = files
    
    grouped_files = filtered_files
    
    if not grouped_files:
        raise FileNotFoundError(f"No valid image files found (only YAML files matched)")
    
    logger.info(f"Found {len(grouped_files)} file groups")
    
    # Initialize QC tests
    all_tests = [
        PixelSizeConsistencyTest(),
        IntensityRangeTest(
            percentile=args.intensity_percentile,
            ratio_threshold=args.intensity_ratio
        ),
        SaturationTest(saturation_threshold=args.saturation_threshold),
        DimensionConsistencyTest(),
        MetadataCompletenessTest(),
        FocusQualityTest(threshold=args.focus_threshold),
    ]
    
    # Filter out skipped tests
    tests = [t for t in all_tests if t.name not in args.skip_tests]
    
    if args.skip_tests:
        logger.info(f"Skipping tests: {', '.join(args.skip_tests)}")
    
    # Run QC
    run_qc_tests(
        grouped_files=grouped_files,
        tests=tests,
        output_summary=args.output_summary,
        fail_on_error=args.fail_on_error,
        html_report_path=args.generate_report
    )
    
    logger.info("QC complete!")


if __name__ == "__main__":
    main()
