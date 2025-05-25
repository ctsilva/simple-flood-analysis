#!/usr/bin/env python3
"""
Unit tests for validation module.

Tests input validation functions, error handling, and edge cases.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we're testing
from validation import (
    validate_file_path, validate_dem_file, validate_shapefile,
    validate_numeric_parameter, validate_output_directory,
    validate_bounds, FileValidationError, DataValidationError,
    ParameterValidationError
)


class TestFileValidation(unittest.TestCase):
    """Test file path validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "test.txt"
        self.test_file.write_text("test content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        if self.temp_dir.exists():
            self.temp_dir.rmdir()
    
    def test_validate_file_path_success(self):
        """Test successful file path validation."""
        result = validate_file_path(self.test_file)
        self.assertEqual(result, self.test_file)
    
    def test_validate_file_path_nonexistent(self):
        """Test validation with non-existent file."""
        nonexistent = self.temp_dir / "nonexistent.txt"
        with self.assertRaises(FileValidationError):
            validate_file_path(nonexistent)
    
    def test_validate_file_path_none(self):
        """Test validation with None path."""
        with self.assertRaises(FileValidationError):
            validate_file_path(None)
    
    def test_validate_file_path_directory(self):
        """Test validation with directory instead of file."""
        with self.assertRaises(FileValidationError):
            validate_file_path(self.temp_dir)
    
    def test_validate_file_path_wrong_extension(self):
        """Test validation with wrong file extension."""
        with self.assertRaises(FileValidationError):
            validate_file_path(self.test_file, required_extensions=['.tif', '.img'])
    
    def test_validate_file_path_correct_extension(self):
        """Test validation with correct file extension."""
        result = validate_file_path(self.test_file, required_extensions=['.txt', '.doc'])
        self.assertEqual(result, self.test_file)


class TestNumericValidation(unittest.TestCase):
    """Test numeric parameter validation functions."""
    
    def test_validate_numeric_parameter_valid_int(self):
        """Test validation with valid integer."""
        result = validate_numeric_parameter(42, "test_param")
        self.assertEqual(result, 42.0)
    
    def test_validate_numeric_parameter_valid_float(self):
        """Test validation with valid float."""
        result = validate_numeric_parameter(3.14, "test_param")
        self.assertEqual(result, 3.14)
    
    def test_validate_numeric_parameter_string_number(self):
        """Test validation with string that can be converted to number."""
        result = validate_numeric_parameter("42.5", "test_param")
        self.assertEqual(result, 42.5)
    
    def test_validate_numeric_parameter_none_not_allowed(self):
        """Test validation with None when not allowed."""
        with self.assertRaises(ParameterValidationError):
            validate_numeric_parameter(None, "test_param")
    
    def test_validate_numeric_parameter_none_allowed(self):
        """Test validation with None when allowed."""
        result = validate_numeric_parameter(None, "test_param", allow_none=True)
        self.assertIsNone(result)
    
    def test_validate_numeric_parameter_invalid_string(self):
        """Test validation with non-numeric string."""
        with self.assertRaises(ParameterValidationError):
            validate_numeric_parameter("not a number", "test_param")
    
    def test_validate_numeric_parameter_nan(self):
        """Test validation with NaN."""
        with self.assertRaises(ParameterValidationError):
            validate_numeric_parameter(float('nan'), "test_param")
    
    def test_validate_numeric_parameter_infinity(self):
        """Test validation with infinity."""
        with self.assertRaises(ParameterValidationError):
            validate_numeric_parameter(float('inf'), "test_param")
    
    def test_validate_numeric_parameter_below_minimum(self):
        """Test validation with value below minimum."""
        with self.assertRaises(ParameterValidationError):
            validate_numeric_parameter(5, "test_param", min_value=10)
    
    def test_validate_numeric_parameter_above_maximum(self):
        """Test validation with value above maximum."""
        with self.assertRaises(ParameterValidationError):
            validate_numeric_parameter(15, "test_param", max_value=10)
    
    def test_validate_numeric_parameter_within_range(self):
        """Test validation with value within range."""
        result = validate_numeric_parameter(7, "test_param", min_value=5, max_value=10)
        self.assertEqual(result, 7.0)


class TestBoundsValidation(unittest.TestCase):
    """Test bounds validation functions."""
    
    def test_validate_bounds_valid_tuple(self):
        """Test validation with valid bounds tuple."""
        bounds = (-74.0, 40.7, -73.9, 40.8)
        result = validate_bounds(bounds)
        self.assertEqual(result, bounds)
    
    def test_validate_bounds_valid_list(self):
        """Test validation with valid bounds list."""
        bounds = [-74.0, 40.7, -73.9, 40.8]
        result = validate_bounds(bounds)
        self.assertEqual(result, tuple(bounds))
    
    def test_validate_bounds_wrong_length(self):
        """Test validation with wrong number of bounds values."""
        with self.assertRaises(ParameterValidationError):
            validate_bounds((1, 2, 3))  # Only 3 values
    
    def test_validate_bounds_non_numeric(self):
        """Test validation with non-numeric bounds."""
        with self.assertRaises(ParameterValidationError):
            validate_bounds(("a", 2, 3, 4))
    
    def test_validate_bounds_invalid_x_range(self):
        """Test validation with min_x >= max_x."""
        with self.assertRaises(ParameterValidationError):
            validate_bounds((5, 1, 3, 4))  # min_x > max_x
    
    def test_validate_bounds_invalid_y_range(self):
        """Test validation with min_y >= max_y."""
        with self.assertRaises(ParameterValidationError):
            validate_bounds((1, 5, 3, 4))  # min_y > max_y
    
    def test_validate_bounds_extreme_longitude(self):
        """Test validation with longitude out of range."""
        with self.assertRaises(ParameterValidationError):
            validate_bounds((-200, 40, -73, 41))  # longitude < -180
    
    def test_validate_bounds_extreme_latitude(self):
        """Test validation with latitude out of range."""
        with self.assertRaises(ParameterValidationError):
            validate_bounds((-74, 100, -73, 101))  # latitude > 90


class TestOutputDirectoryValidation(unittest.TestCase):
    """Test output directory validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            self.temp_dir.rmdir()
    
    def test_validate_output_directory_existing(self):
        """Test validation with existing directory."""
        result = validate_output_directory(self.temp_dir)
        self.assertEqual(result, self.temp_dir)
    
    def test_validate_output_directory_create_missing(self):
        """Test validation with non-existing directory (create=True)."""
        new_dir = self.temp_dir / "new_subdir"
        result = validate_output_directory(new_dir, create_if_missing=True)
        self.assertEqual(result, new_dir)
        self.assertTrue(new_dir.exists())
        new_dir.rmdir()  # Clean up
    
    def test_validate_output_directory_dont_create_missing(self):
        """Test validation with non-existing directory (create=False)."""
        new_dir = self.temp_dir / "new_subdir"
        with self.assertRaises(FileValidationError):
            validate_output_directory(new_dir, create_if_missing=False)
    
    def test_validate_output_directory_none(self):
        """Test validation with None directory."""
        with self.assertRaises(FileValidationError):
            validate_output_directory(None)


class TestDEMValidation(unittest.TestCase):
    """Test DEM file validation functions."""
    
    @patch('rasterio.open')
    def test_validate_dem_file_success(self, mock_rasterio_open):
        """Test successful DEM validation."""
        # Mock rasterio dataset
        mock_dataset = MagicMock()
        mock_dataset.count = 1
        mock_dataset.width = 100
        mock_dataset.height = 100
        mock_dataset.read.return_value = np.ones((10, 10))
        mock_dataset.nodata = None
        
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        # Create temporary file with correct extension
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            result = validate_dem_file(temp_path)
            self.assertEqual(result, temp_path)
        finally:
            temp_path.unlink()
    
    def test_validate_dem_file_wrong_extension(self):
        """Test DEM validation with wrong file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            with self.assertRaises(FileValidationError):
                validate_dem_file(temp_path)
        finally:
            temp_path.unlink()
    
    @patch('rasterio.open')
    def test_validate_dem_file_no_bands(self, mock_rasterio_open):
        """Test DEM validation with no bands."""
        mock_dataset = MagicMock()
        mock_dataset.count = 0
        
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            with self.assertRaises(DataValidationError):
                validate_dem_file(temp_path)
        finally:
            temp_path.unlink()
    
    @patch('rasterio.open')
    def test_validate_dem_file_extreme_values(self, mock_rasterio_open):
        """Test DEM validation with extreme elevation values."""
        mock_dataset = MagicMock()
        mock_dataset.count = 1
        mock_dataset.width = 100
        mock_dataset.height = 100
        mock_dataset.nodata = None
        # Create array with unrealistic elevation
        mock_dataset.read.return_value = np.full((10, 10), 20000)  # 20km elevation
        
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            with self.assertRaises(DataValidationError):
                validate_dem_file(temp_path)
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    unittest.main()