#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for flood analysis tests.

This module provides common test fixtures and configuration
for the entire test suite.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_dem_data():
    """Create sample DEM data for testing."""
    # Create a simple 100x100 elevation grid
    height, width = 100, 100
    
    # Create realistic elevation data
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # Create elevation surface (Gaussian-like with some variation)
    elevation = 50 + 30 * np.exp(-(X**2 + Y**2)) + 5 * np.random.randn(height, width)
    
    # Add some "water" areas (low elevation)
    water_mask = (X**2 + Y**2) > 3
    elevation[water_mask] = np.random.uniform(-2, 5, np.sum(water_mask))
    
    return elevation.astype(np.float32)


@pytest.fixture
def mock_rasterio_dataset(sample_dem_data):
    """Create a mock rasterio dataset for testing."""
    mock_dataset = MagicMock()
    
    # Set up basic properties
    mock_dataset.width = sample_dem_data.shape[1]
    mock_dataset.height = sample_dem_data.shape[0]
    mock_dataset.count = 1
    mock_dataset.dtypes = [sample_dem_data.dtype]
    mock_dataset.nodata = -9999
    
    # Set up coordinate system
    mock_dataset.crs = MagicMock()
    mock_dataset.crs.to_epsg.return_value = 2263  # NY State Plane
    
    # Set up transform (simple identity-like transform)
    mock_transform = MagicMock()
    mock_transform.__getitem__ = lambda self, key: [1.0, 0.0, 0.0, 0.0, -1.0, 100.0][key]
    mock_dataset.transform = mock_transform
    
    # Set up bounds
    mock_bounds = MagicMock()
    mock_bounds.left = 0
    mock_bounds.bottom = 0
    mock_bounds.right = sample_dem_data.shape[1]
    mock_bounds.top = sample_dem_data.shape[0]
    mock_dataset.bounds = mock_bounds
    
    # Set up resolution
    mock_dataset.res = (1.0, 1.0)
    
    # Set up profile
    mock_dataset.profile = {
        'driver': 'GTiff',
        'width': mock_dataset.width,
        'height': mock_dataset.height,
        'count': 1,
        'dtype': sample_dem_data.dtype,
        'crs': mock_dataset.crs,
        'transform': mock_dataset.transform,
        'nodata': mock_dataset.nodata
    }
    
    # Mock read method
    def mock_read(band=1, window=None):
        if window is None:
            return sample_dem_data
        else:
            # Simple window reading simulation
            row_start = getattr(window, 'row_off', 0)
            col_start = getattr(window, 'col_off', 0)
            row_end = row_start + getattr(window, 'height', 10)
            col_end = col_start + getattr(window, 'width', 10)
            return sample_dem_data[row_start:row_end, col_start:col_end]
    
    mock_dataset.read = mock_read
    
    return mock_dataset


@pytest.fixture
def sample_roads_geojson(temp_directory):
    """Create a sample roads GeoJSON file for testing."""
    geojson_content = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "EPSG:2263"}
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Main Street", "type": "primary"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[10, 10], [20, 20], [30, 30]]
                }
            },
            {
                "type": "Feature", 
                "properties": {"name": "Side Street", "type": "secondary"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[5, 15], [15, 25], [25, 35]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Circle Road", "type": "local"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[40, 40], [45, 42], [50, 40], [45, 38], [40, 40]]
                }
            }
        ]
    }
    
    import json
    geojson_file = temp_directory / "sample_roads.geojson"
    with open(geojson_file, 'w') as f:
        json.dump(geojson_content, f)
    
    return geojson_file


@pytest.fixture
def sample_elevation_values():
    """Provide sample elevation values for testing."""
    return {
        'valid_elevations': np.array([1.5, 5.2, 12.8, 25.1, 8.7, 15.3, 2.1]),
        'water_levels': [5, 10, 15, 20],
        'expected_flood_counts': {
            5: 2,   # elevations <= 5: [1.5, 2.1]
            10: 4,  # elevations <= 10: [1.5, 5.2, 8.7, 2.1]
            15: 6,  # elevations <= 15: [1.5, 5.2, 12.8, 8.7, 15.3, 2.1]
            20: 7   # all elevations <= 20
        }
    }


@pytest.fixture
def flood_risk_test_data():
    """Provide test data for flood risk calculations."""
    return {
        'road_elevations': [2.0, 8.5, 15.2, 22.1, 5.8],
        'flood_level': 10.0,
        'expected_high_risk': 2,  # roads with elevation <= 10: [2.0, 8.5, 5.8]
        'expected_medium_risk': 1,  # roads with 10 < elevation <= 15: [15.2]
        'expected_low_risk': 1   # roads with elevation > 15: [22.1]
    }


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require external data files"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark certain tests based on their names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "large" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or item.fspath.basename.startswith("test_integration"):
            item.add_marker(pytest.mark.integration)


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up clean environment for each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    os.environ['FLOOD_ANALYSIS_LOG_LEVEL'] = 'DEBUG'
    os.environ.pop('FLOOD_ANALYSIS_DATA_DIR', None)
    os.environ.pop('FLOOD_ANALYSIS_OUTPUT_DIR', None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)