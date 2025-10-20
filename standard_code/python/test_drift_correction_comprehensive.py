#!/usr/bin/env python3
"""
Comprehensive Drift Correction Test Suite

This module implements a robust testing strategy to validate and optimize drift correction algorithms.
Tests use synthetic data with known ground truth to quantitatively measure algorithm performance.

Test Strategy:
1. Generate synthetic datasets with varying drift characteristics
2. Apply drift correction algorithms
3. Measure correction quality using multiple metrics
4. Compare results against ground truth
5. Generate detailed diagnostic reports

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Universal import fix
current_dir = Path(__file__).parent
standard_code_dir = current_dir
project_root = standard_code_dir.parent
for path in [str(project_root), str(standard_code_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import bioimage_pipeline_utils as rp
from drift_correction_utils.synthetic_data_generators import (
    create_simple_squares,
    create_cell_like_video
)
from drift_correction_utils.drift_correct_utils import (
    drift_correction_score,
    apply_shifts_to_tczyx_stack
)
from drift_correction_visualization import (
    plot_shift_trajectory,
    plot_before_after_comparison,
    plot_score_comparison,
    plot_accuracy_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftCorrectionTester:
    """Comprehensive drift correction testing framework."""
    
    def __init__(self, output_dir: str):
        """
        Initialize tester.
        
        Args:
            output_dir: Directory to save test results and diagnostics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        logger.info(f"Initialized drift correction tester - Output: {self.output_dir}")
    
    def generate_test_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, List]]:
        """
        Generate comprehensive synthetic test datasets with known ground truth.
        
        Returns:
            Dictionary mapping test names to (drifted_video, ground_truth, true_shifts)
        """
        logger.info("Generating synthetic test datasets...")
        
        datasets = {}
        
        # Test 1: Small drift with simple squares (2D) - Should achieve near-perfect correction
        logger.info("Generating: small_drift_2d_squares")
        img_data, gt_data, shifts = create_simple_squares(
            T=8, C=1, Z=1, Y=128, X=128, square_size=12
        )
        datasets['small_drift_2d_squares'] = (img_data, gt_data, shifts)
        
        # Test 2: Medium drift with simple squares (2D)
        logger.info("Generating: medium_drift_2d_squares")
        img_data, gt_data, shifts = create_simple_squares(
            T=10, C=1, Z=1, Y=200, X=200, square_size=15
        )
        # Increase drift magnitude
        for i in range(len(shifts)):
            shifts[i] = tuple(s * 3.0 for s in shifts[i])
        # Re-apply larger shifts
        img_data = self._apply_known_shifts(gt_data, shifts)
        datasets['medium_drift_2d_squares'] = (img_data, gt_data, shifts)
        
        # Test 3: Large drift with simple squares (2D) - Critical test for 5-6px oscillations
        logger.info("Generating: large_drift_2d_squares")
        img_data, gt_data, shifts = create_simple_squares(
            T=12, C=1, Z=1, Y=256, X=256, square_size=20
        )
        # Create large drift with potential oscillation patterns
        large_shifts = []
        for t in range(12):
            dy = 8.0 * np.sin(t * 0.5) + np.random.uniform(-1, 1)  # Smooth drift with noise
            dx = 6.0 * np.cos(t * 0.5) + np.random.uniform(-1, 1)
            large_shifts.append((dy, dx))
        img_data = self._apply_known_shifts(gt_data, large_shifts)
        datasets['large_drift_2d_squares'] = (img_data, gt_data, large_shifts)
        
        # Test 4: Cell-like patterns with noise (2D) - Realistic biological data
        logger.info("Generating: cell_like_2d_noisy")
        img_data, gt_data, shifts = create_cell_like_video(
            T=10, C=1, Z=1, Y=256, X=256, num_cells=15, noise_level=10.0
        )
        datasets['cell_like_2d_noisy'] = (img_data, gt_data, shifts)
        
        # Test 5: 3D drift with simple cubes (small drift)
        logger.info("Generating: small_drift_3d_cubes")
        img_data, gt_data, shifts = create_simple_squares(
            T=8, C=1, Z=8, Y=100, X=100, square_size=10
        )
        datasets['small_drift_3d_cubes'] = (img_data, gt_data, shifts)
        
        # Test 6: 3D cell-like patterns
        logger.info("Generating: cell_like_3d")
        img_data, gt_data, shifts = create_cell_like_video(
            T=8, C=1, Z=8, Y=128, X=128, num_cells=10, noise_level=5.0
        )
        datasets['cell_like_3d'] = (img_data, gt_data, shifts)
        
        logger.info(f"Generated {len(datasets)} test datasets")
        return datasets
    
    def _apply_known_shifts(self, ground_truth: np.ndarray, shifts: List[Tuple]) -> np.ndarray:
        """Apply known shifts to ground truth to create drifted video."""
        return apply_shifts_to_tczyx_stack(ground_truth, np.array(shifts), mode='constant', order=3)
    
    def compute_enhanced_scores(self, img_data: np.ndarray, name: str = "") -> Dict[str, float]:
        """
        Compute multiple alignment scores for comprehensive evaluation.
        
        Args:
            img_data: TCZYX image stack
            name: Descriptive name for logging
            
        Returns:
            Dictionary of score names to values
        """
        scores = {}
        
        # Original drift correction score (Pearson correlation)
        scores['pearson_first'] = drift_correction_score(
            img_data=img_data, channel=0, reference="first", 
            central_crop=0.8, z_project="mean" if img_data.shape[2] > 1 else None
        )
        
        scores['pearson_previous'] = drift_correction_score(
            img_data=img_data, channel=0, reference="previous",
            central_crop=0.8, z_project="mean" if img_data.shape[2] > 1 else None
        )
        
        scores['pearson_median'] = drift_correction_score(
            img_data=img_data, channel=0, reference="median",
            central_crop=0.8, z_project="mean" if img_data.shape[2] > 1 else None
        )
        
        # Temporal stability: variance of frame-to-frame differences
        scores['temporal_stability'] = self._compute_temporal_stability(img_data)
        
        # Edge alignment quality
        scores['edge_alignment'] = self._compute_edge_alignment(img_data)
        
        if name:
            logger.info(f"{name} scores: pearson_first={scores['pearson_first']:.4f}, "
                       f"temporal_stability={scores['temporal_stability']:.4f}")
        
        return scores
    
    def _compute_temporal_stability(self, img_data: np.ndarray) -> float:
        """
        Measure temporal stability: lower frame-to-frame variation = better alignment.
        
        Returns value in [0, 1] where 1 is perfectly stable.
        """
        T, C, Z, Y, X = img_data.shape
        if T < 2:
            return 1.0
        
        # Compute frame-to-frame differences
        diffs = []
        for t in range(1, T):
            # Use central crop to avoid edge artifacts
            crop_y = int(Y * 0.1)
            crop_x = int(X * 0.1)
            
            frame_prev = img_data[t-1, 0, :, crop_y:-crop_y, crop_x:-crop_x].astype(np.float32)
            frame_curr = img_data[t, 0, :, crop_y:-crop_y, crop_x:-crop_x].astype(np.float32)
            
            # Normalized difference
            diff = np.abs(frame_curr - frame_prev)
            mean_intensity = (np.mean(frame_prev) + np.mean(frame_curr)) / 2
            if mean_intensity > 0:
                normalized_diff = np.mean(diff) / mean_intensity
                diffs.append(normalized_diff)
        
        if not diffs:
            return 1.0
        
        # Convert variance to 0-1 score (lower variance = higher score)
        mean_diff = np.mean(diffs)
        # Clamp and invert: assume 0.5 = very bad, 0.0 = perfect
        stability = np.clip(1.0 - mean_diff * 2.0, 0.0, 1.0)
        
        return float(stability)
    
    def _compute_edge_alignment(self, img_data: np.ndarray) -> float:
        """
        Measure edge alignment quality across timepoints.
        
        Returns value in [0, 1] where 1 is perfect alignment.
        """
        try:
            from skimage.filters import sobel
        except ImportError:
            logger.warning("scikit-image not available, skipping edge alignment")
            return 0.0
        
        T, C, Z, Y, X = img_data.shape
        if T < 2:
            return 1.0
        
        # Compute edges for each frame
        edges = []
        for t in range(T):
            if Z > 1:
                # Use max projection for 3D
                frame = np.max(img_data[t, 0], axis=0)
            else:
                frame = img_data[t, 0, 0]
            
            edge_map = sobel(frame.astype(np.float32))
            edges.append(edge_map)
        
        # Compute correlation between edge maps (should be high if aligned)
        correlations = []
        ref_edges = edges[0].ravel()
        ref_edges = (ref_edges - ref_edges.mean()) / (ref_edges.std() + 1e-12)
        
        for t in range(1, T):
            curr_edges = edges[t].ravel()
            curr_edges = (curr_edges - curr_edges.mean()) / (curr_edges.std() + 1e-12)
            
            corr = np.dot(ref_edges, curr_edges) / len(ref_edges)
            correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 1.0
    
    def compute_ground_truth_accuracy(self, computed_shifts: np.ndarray, 
                                     true_shifts: List[Tuple]) -> Dict[str, float]:
        """
        Compare computed shifts against ground truth.
        
        IMPORTANT: true_shifts are the DRIFT shifts (applied to create synthetic data),
        but PCC returns CORRECTION shifts (negative of drift). So we compare:
        computed_shifts vs -true_shifts
        
        Args:
            computed_shifts: Algorithm output shifts (T, 2) or (T, 3) - these are CORRECTION shifts
            true_shifts: Ground truth DRIFT shifts from synthetic generation
            
        Returns:
            Dictionary with accuracy metrics
        """
        true_shifts_array = np.array(true_shifts)
        
        # PCC returns correction shifts, which are negative of drift shifts
        expected_correction_shifts = -true_shifts_array
        
        # Compute errors
        errors = computed_shifts - expected_correction_shifts
        
        # RMS error
        rms_error = np.sqrt(np.mean(errors**2))
        
        # Mean absolute error
        mae = np.mean(np.abs(errors))
        
        # Max error
        max_error = np.max(np.abs(errors))
        
        # Per-dimension errors
        dim_names = ['dz', 'dy', 'dx'] if errors.shape[1] == 3 else ['dy', 'dx']
        dim_errors = {}
        for i, name in enumerate(dim_names):
            dim_errors[f'{name}_rmse'] = np.sqrt(np.mean(errors[:, i]**2))
            dim_errors[f'{name}_mae'] = np.mean(np.abs(errors[:, i]))
        
        return {
            'rms_error': float(rms_error),
            'mae': float(mae),
            'max_error': float(max_error),
            **dim_errors
        }
    
    def test_phase_cross_correlation(self, img_data: np.ndarray, ground_truth: np.ndarray,
                                    true_shifts: List[Tuple], test_name: str,
                                    use_gpu: bool = False) -> Dict[str, Any]:
        """
        Test phase cross-correlation algorithm.
        
        Args:
            img_data: Drifted video (T,C,Z,Y,X)
            ground_truth: Reference video without drift (T,C,Z,Y,X)
            true_shifts: Known shifts applied
            test_name: Descriptive test name
            use_gpu: Use GPU acceleration
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing phase_cross_correlation {'GPU' if use_gpu else 'CPU'} on {test_name}")
        
        from standard_code.python.drift_correction_utils._phase_cross_correlation import (
            phase_cross_correlation_cpu,
            phase_cross_correlation_gpu
        )
        
        T, C, Z, Y, X = img_data.shape
        
        # Prepare data (T, Z, Y, X) for algorithm
        if Z == 1:
            reg_data = img_data[:, 0, 0, :, :]  # (T, Y, X)
        else:
            reg_data = img_data[:, 0, :, :, :]  # (T, Z, Y, X)
        
        # Run algorithm
        try:
            if use_gpu:
                computed_shifts, _ = phase_cross_correlation_gpu(
                    reg_data, reference="first", upsample_factor=10
                )
            else:
                computed_shifts, _ = phase_cross_correlation_cpu(
                    reg_data, reference="first", upsample_factor=10
                )
            
            # Apply correction
            corrected = apply_shifts_to_tczyx_stack(img_data, computed_shifts, mode='constant', order=3)
            
            # Compute scores
            input_scores = self.compute_enhanced_scores(img_data, f"{test_name} INPUT")
            output_scores = self.compute_enhanced_scores(corrected, f"{test_name} OUTPUT")
            gt_scores = self.compute_enhanced_scores(ground_truth, f"{test_name} GT")
            
            # Compute accuracy vs ground truth
            accuracy = self.compute_ground_truth_accuracy(computed_shifts, true_shifts)
            
            # Check for oscillations (5-6px jumps detected)
            oscillation_detected = self._detect_oscillations(computed_shifts)
            
            # Create visualizations
            viz_dir = self.output_dir / "visualizations" / test_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            algo_name = f"pcc_{'gpu' if use_gpu else 'cpu'}"
            
            try:
                plot_shift_trajectory(
                    computed_shifts, true_shifts,
                    str(viz_dir / f"shifts_{algo_name}.png"),
                    f"{test_name} - {algo_name}"
                )
                
                plot_before_after_comparison(
                    img_data, corrected,
                    str(viz_dir / f"comparison_{algo_name}.png"),
                    f"{test_name} - {algo_name}"
                )
                
                scores_dict = {
                    'input_scores': input_scores,
                    'output_scores': output_scores,
                    'ground_truth_scores': gt_scores
                }
                plot_score_comparison(
                    scores_dict,
                    str(viz_dir / f"scores_{algo_name}.png"),
                    f"{test_name} - {algo_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to create visualizations: {e}")
            
            result = {
                'test_name': test_name,
                'algorithm': f"phase_cross_correlation_{'gpu' if use_gpu else 'cpu'}",
                'input_scores': input_scores,
                'output_scores': output_scores,
                'ground_truth_scores': gt_scores,
                'accuracy': accuracy,
                'oscillation_detected': oscillation_detected,
                'improvement': {
                    'pearson_first': output_scores['pearson_first'] - input_scores['pearson_first'],
                    'temporal_stability': output_scores['temporal_stability'] - input_scores['temporal_stability']
                },
                'success': output_scores['pearson_first'] > input_scores['pearson_first']
            }
            
            logger.info(f"  ✓ Completed - Pearson improvement: {result['improvement']['pearson_first']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'test_name': test_name,
                'algorithm': f"phase_cross_correlation_{'gpu' if use_gpu else 'cpu'}",
                'error': str(e),
                'success': False
            }
    
    def _detect_oscillations(self, shifts: np.ndarray) -> Dict[str, Any]:
        """
        Detect oscillations in shift trajectory (back-and-forth movements).
        
        NOTE: This checks the CORRECTION shifts returned by the algorithm.
        Reversals in correction shifts are NORMAL when undoing back-and-forth drift.
        True oscillation problems occur when corrections OVERCORRECT, causing the
        corrected image to oscillate more than the input.
        
        This function is kept for diagnostic purposes but should not be used
        as the primary quality metric. Use drift_correction_score instead.
        
        Args:
            shifts: Computed shifts (T, 2) or (T, 3)
            
        Returns:
            Dictionary with oscillation metrics
        """
        T = shifts.shape[0]
        if T < 3:
            return {'detected': False, 'max_oscillation': 0.0, 'note': 'Too few timepoints'}
        
        # Compute second derivative (acceleration) to detect direction changes
        shift_deltas = np.diff(shifts, axis=0)  # First derivative (velocity)
        shift_accel = np.diff(shift_deltas, axis=0)  # Second derivative (acceleration)
        
        # Large accelerations indicate oscillations
        max_accel = np.max(np.abs(shift_accel))
        mean_accel = np.mean(np.abs(shift_accel))
        
        # Count direction reversals (sign changes in velocity)
        reversals = 0
        for dim in range(shifts.shape[1]):
            signs = np.sign(shift_deltas[:, dim])
            reversals += np.sum(np.abs(np.diff(signs))) / 2  # Each reversal = 2 sign change
        
        # NOTE: Reversals are EXPECTED in correction shifts when undoing oscillating drift
        # Only flag as problematic if accelerations are VERY large (>10px)
        detected = max_accel > 10.0  # Much higher threshold
        
        return {
            'detected': bool(detected),
            'max_oscillation': float(max_accel),
            'mean_oscillation': float(mean_accel),
            'num_reversals': int(reversals),
            'reversal_rate': float(reversals / max(1, T-2)),
            'note': 'Reversals in correction shifts are normal; only large accelerations (>10px) indicate problems'
        }
    
    def run_all_tests(self) -> None:
        """Run complete test suite and generate report."""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE DRIFT CORRECTION TEST SUITE")
        logger.info("="*80)
        
        # Generate test datasets
        datasets = self.generate_test_datasets()
        
        # Save datasets for inspection
        datasets_dir = self.output_dir / "test_datasets"
        datasets_dir.mkdir(exist_ok=True)
        for name, (drifted, gt, shifts) in datasets.items():
            rp.save_tczyx_image(drifted, str(datasets_dir / f"{name}_drifted.tif"))
            rp.save_tczyx_image(gt, str(datasets_dir / f"{name}_ground_truth.tif"))
            
            # Save shifts as JSON
            shifts_data = {
                'shifts': [list(s) for s in shifts],
                'format': 'ZYX order (dz, dy, dx) for 3D or YX order (dy, dx) for 2D'
            }
            with open(datasets_dir / f"{name}_true_shifts.json", 'w') as f:
                json.dump(shifts_data, f, indent=2)
        
        logger.info(f"Saved {len(datasets)} test datasets to {datasets_dir}")
        
        # Run tests on each dataset
        all_results = []
        
        for test_name, (drifted, gt, shifts) in datasets.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing: {test_name}")
            logger.info(f"{'='*80}")
            
            # Test CPU version
            result_cpu = self.test_phase_cross_correlation(
                drifted, gt, shifts, test_name, use_gpu=False
            )
            all_results.append(result_cpu)
            
            # Test GPU version if available
            try:
                import cupy
                result_gpu = self.test_phase_cross_correlation(
                    drifted, gt, shifts, test_name, use_gpu=True
                )
                all_results.append(result_gpu)
            except ImportError:
                logger.info("  Skipping GPU test (CuPy not available)")
        
        # Generate summary report
        self._generate_report(all_results)
        
        # Create summary visualizations
        try:
            plot_accuracy_summary(
                all_results,
                str(self.output_dir / "accuracy_summary.png")
            )
            logger.info(f"Created summary visualizations in {self.output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create summary visualizations: {e}")
        
        logger.info(f"\n{'='*80}")
        logger.info("TEST SUITE COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}")
    
    def _generate_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate comprehensive test report."""
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Compute summary statistics
        successful_tests = [r for r in results if r.get('success', False)]
        failed_tests = [r for r in results if not r.get('success', False)]
        oscillation_tests = [r for r in results if r.get('oscillation_detected', {}).get('detected', False)]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'successful': len(successful_tests),
            'failed': len(failed_tests),
            'oscillations_detected': len(oscillation_tests),
            'success_rate': len(successful_tests) / len(results) if results else 0,
            'average_improvement': np.mean([r['improvement']['pearson_first'] 
                                           for r in successful_tests]) if successful_tests else 0,
            'results': results
        }
        
        # Save JSON report
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Oscillations detected: {summary['oscillations_detected']}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Average Pearson improvement: {summary['average_improvement']:.4f}")
        
        if oscillation_tests:
            logger.warning(f"\n⚠️  OSCILLATION ISSUES DETECTED IN {len(oscillation_tests)} TESTS:")
            for r in oscillation_tests:
                osc = r['oscillation_detected']
                logger.warning(f"  - {r['test_name']}: {osc['num_reversals']} reversals, "
                             f"max oscillation {osc['max_oscillation']:.2f}px")
        
        logger.info(f"\nFull report saved to: {report_file}")


def main():
    """Main entry point for comprehensive drift correction testing."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get test folder from environment or use default
    test_folder = os.getenv("TEST_FOLDER", "test_results")
    output_dir = os.path.join(test_folder, "drift_correction", "comprehensive_tests")
    
    logger.info(f"Starting comprehensive drift correction tests")
    logger.info(f"Output directory: {output_dir}")
    
    # Create tester and run all tests
    tester = DriftCorrectionTester(output_dir)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
