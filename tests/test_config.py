#!/usr/bin/env python3
"""
Unit tests for configuration module.

Tests configuration management, environment variable handling,
and default value validation.
"""

import unittest
import os
from pathlib import Path
from unittest.mock import patch

from config import FloodAnalysisConfig, EnvironmentConfig, LoggingConfig


class TestFloodAnalysisConfig(unittest.TestCase):
    """Test FloodAnalysisConfig class methods."""
    
    def test_get_default_dem_path(self):
        """Test getting default DEM path."""
        expected = FloodAnalysisConfig.DEFAULT_DATA_DIR / FloodAnalysisConfig.DEFAULT_DEM_FILENAME
        result = FloodAnalysisConfig.get_default_dem_path()
        self.assertEqual(result, expected)
    
    def test_get_default_roads_path(self):
        """Test getting default roads path."""
        expected = FloodAnalysisConfig.DEFAULT_DATA_DIR / FloodAnalysisConfig.DEFAULT_ROADS_FILENAME
        result = FloodAnalysisConfig.get_default_roads_path()
        self.assertEqual(result, expected)
    
    def test_get_output_path_default_dir(self):
        """Test getting output path with default directory."""
        filename = "test_output.tif"
        expected = FloodAnalysisConfig.DEFAULT_OUTPUT_DIR / filename
        result = FloodAnalysisConfig.get_output_path(filename)
        self.assertEqual(result, expected)
    
    def test_get_output_path_custom_dir(self):
        """Test getting output path with custom directory."""
        filename = "test_output.tif"
        custom_dir = Path("/custom/output")
        expected = custom_dir / filename
        result = FloodAnalysisConfig.get_output_path(filename, custom_dir)
        self.assertEqual(result, expected)
    
    def test_validate_water_level_valid(self):
        """Test water level validation with valid values."""
        valid_levels = [0, 10, 100, -10]
        for level in valid_levels:
            with self.subTest(level=level):
                self.assertTrue(FloodAnalysisConfig.validate_water_level(level))
    
    def test_validate_water_level_invalid(self):
        """Test water level validation with invalid values."""
        invalid_levels = [-2000, 20000]  # Extreme values
        for level in invalid_levels:
            with self.subTest(level=level):
                self.assertFalse(FloodAnalysisConfig.validate_water_level(level))
    
    def test_get_risk_category(self):
        """Test risk category classification."""
        test_cases = [
            (0.9, 'High'),
            (0.7, 'Medium'),  # Boundary case
            (0.5, 'Medium'),
            (0.4, 'Low'),     # Boundary case
            (0.1, 'Low'),
            (0.0, 'Low')
        ]
        
        for score, expected_category in test_cases:
            with self.subTest(score=score):
                result = FloodAnalysisConfig.get_risk_category(score)
                self.assertEqual(result, expected_category)
    
    def test_get_visualization_sample_size_small_dataset(self):
        """Test visualization sample size for small datasets."""
        small_pixels = 1000
        result = FloodAnalysisConfig.get_visualization_sample_size(small_pixels)
        self.assertEqual(result, small_pixels)
    
    def test_get_visualization_sample_size_large_dataset(self):
        """Test visualization sample size for large datasets."""
        large_pixels = 10_000_000  # 10 million pixels
        result = FloodAnalysisConfig.get_visualization_sample_size(large_pixels)
        max_pixels = FloodAnalysisConfig.PERFORMANCE_CONFIG['max_visualization_pixels']
        self.assertLess(result, max_pixels * max_pixels)
    
    def test_supported_extensions_not_empty(self):
        """Test that supported file extensions lists are not empty."""
        self.assertGreater(len(FloodAnalysisConfig.SUPPORTED_DEM_EXTENSIONS), 0)
        self.assertGreater(len(FloodAnalysisConfig.SUPPORTED_VECTOR_EXTENSIONS), 0)
    
    def test_risk_thresholds_ordered(self):
        """Test that risk thresholds are properly ordered."""
        thresholds = FloodAnalysisConfig.RISK_THRESHOLDS
        self.assertGreater(thresholds['high'], thresholds['medium'])
        self.assertGreaterEqual(thresholds['medium'], thresholds['low'])
    
    def test_d8_flow_directions_complete(self):
        """Test that D8 flow directions cover all 8 directions."""
        directions = FloodAnalysisConfig.D8_FLOW_DIRECTIONS
        self.assertEqual(len(directions), 8)
        
        # Check that direction codes are 1-8
        codes = [code for code, _ in directions]
        self.assertEqual(set(codes), set(range(1, 9)))


class TestEnvironmentConfig(unittest.TestCase):
    """Test EnvironmentConfig class methods."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_data_directory_default(self):
        """Test getting data directory with no environment variable."""
        result = EnvironmentConfig.get_data_directory()
        expected = FloodAnalysisConfig.DEFAULT_DATA_DIR
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_DATA_DIR': '/custom/data'})
    def test_get_data_directory_from_env(self):
        """Test getting data directory from environment variable."""
        result = EnvironmentConfig.get_data_directory()
        expected = Path('/custom/data')
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_output_directory_default(self):
        """Test getting output directory with no environment variable."""
        result = EnvironmentConfig.get_output_directory()
        expected = FloodAnalysisConfig.DEFAULT_OUTPUT_DIR
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_OUTPUT_DIR': '/custom/output'})
    def test_get_output_directory_from_env(self):
        """Test getting output directory from environment variable."""
        result = EnvironmentConfig.get_output_directory()
        expected = Path('/custom/output')
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_max_memory_mb_default(self):
        """Test getting max memory with no environment variable."""
        result = EnvironmentConfig.get_max_memory_mb()
        expected = FloodAnalysisConfig.PERFORMANCE_CONFIG['max_memory_usage_mb']
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_MAX_MEMORY_MB': '1024'})
    def test_get_max_memory_mb_from_env(self):
        """Test getting max memory from environment variable."""
        result = EnvironmentConfig.get_max_memory_mb()
        self.assertEqual(result, 1024)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_MAX_MEMORY_MB': 'invalid'})
    def test_get_max_memory_mb_invalid_env(self):
        """Test getting max memory with invalid environment variable."""
        result = EnvironmentConfig.get_max_memory_mb()
        expected = FloodAnalysisConfig.PERFORMANCE_CONFIG['max_memory_usage_mb']
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_is_debug_mode_default(self):
        """Test debug mode detection with no environment variable."""
        result = EnvironmentConfig.is_debug_mode()
        self.assertFalse(result)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_DEBUG': '1'})
    def test_is_debug_mode_true_numeric(self):
        """Test debug mode detection with numeric true value."""
        result = EnvironmentConfig.is_debug_mode()
        self.assertTrue(result)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_DEBUG': 'true'})
    def test_is_debug_mode_true_text(self):
        """Test debug mode detection with text true value."""
        result = EnvironmentConfig.is_debug_mode()
        self.assertTrue(result)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_DEBUG': 'false'})
    def test_is_debug_mode_false(self):
        """Test debug mode detection with false value."""
        result = EnvironmentConfig.is_debug_mode()
        self.assertFalse(result)


class TestLoggingConfig(unittest.TestCase):
    """Test LoggingConfig class methods."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_log_level_default(self):
        """Test getting log level with no environment variable."""
        result = LoggingConfig.get_log_level()
        expected = LoggingConfig.DEFAULT_LOG_LEVEL
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_LOG_LEVEL': 'DEBUG'})
    def test_get_log_level_from_env(self):
        """Test getting log level from environment variable."""
        result = LoggingConfig.get_log_level()
        self.assertEqual(result, 'DEBUG')
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_log_file_default(self):
        """Test getting log file with no environment variable."""
        result = LoggingConfig.get_log_file()
        expected = EnvironmentConfig.get_output_directory() / 'flood_analysis.log'
        self.assertEqual(result, expected)
    
    @patch.dict(os.environ, {'FLOOD_ANALYSIS_LOG_FILE': '/custom/app.log'})
    def test_get_log_file_from_env(self):
        """Test getting log file from environment variable."""
        result = LoggingConfig.get_log_file()
        expected = Path('/custom/app.log')
        self.assertEqual(result, expected)
    
    def test_default_values_exist(self):
        """Test that default configuration values exist."""
        self.assertIsNotNone(LoggingConfig.DEFAULT_LOG_LEVEL)
        self.assertIsNotNone(LoggingConfig.DEFAULT_LOG_FORMAT)
        self.assertIsNotNone(LoggingConfig.DEFAULT_DATE_FORMAT)


class TestConfigConstants(unittest.TestCase):
    """Test configuration constants and values."""
    
    def test_nyc_bounds_structure(self):
        """Test NYC bounds dictionary structure."""
        bounds = FloodAnalysisConfig.NYC_BOUNDS
        
        # Check required keys exist
        self.assertIn('wgs84', bounds)
        self.assertIn('state_plane', bounds)
        
        # Check WGS84 bounds structure
        wgs84 = bounds['wgs84']
        self.assertIn('lat_range', wgs84)
        self.assertIn('lon_range', wgs84)
        
        # Check that ranges are tuples with 2 values
        self.assertEqual(len(wgs84['lat_range']), 2)
        self.assertEqual(len(wgs84['lon_range']), 2)
        
        # Check that ranges are ordered correctly
        lat_min, lat_max = wgs84['lat_range']
        lon_min, lon_max = wgs84['lon_range']
        self.assertLess(lat_min, lat_max)
        self.assertLess(lon_min, lon_max)
    
    def test_visualization_config_structure(self):
        """Test visualization configuration structure."""
        viz_config = FloodAnalysisConfig.VISUALIZATION_CONFIG
        
        required_keys = ['elevation_colormap', 'flood_risk_colormap', 
                        'default_figure_size', 'road_colors', 'road_linewidths']
        
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, viz_config)
    
    def test_performance_config_structure(self):
        """Test performance configuration structure."""
        perf_config = FloodAnalysisConfig.PERFORMANCE_CONFIG
        
        required_keys = ['max_visualization_pixels', 'chunk_size_for_large_rasters',
                        'progress_update_interval', 'max_memory_usage_mb']
        
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, perf_config)
                self.assertIsInstance(perf_config[key], int)
                self.assertGreater(perf_config[key], 0)


if __name__ == '__main__':
    unittest.main()