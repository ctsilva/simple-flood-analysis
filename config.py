#!/usr/bin/env python3
"""
Configuration management for flood analysis toolkit.

This module centralizes all configuration parameters, default values,
and settings used throughout the flood analysis toolkit.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from pathlib import Path
import os


class FloodAnalysisConfig:
    """
    Central configuration class for flood analysis parameters.
    
    This class contains all configurable parameters used across the toolkit,
    providing a single source of truth for default values and settings.
    """
    
    # File and directory settings
    DEFAULT_DATA_DIR = Path("nyc_data")
    DEFAULT_OUTPUT_DIR = Path("flood_analysis_results")
    DEFAULT_DEM_FILENAME = "DEM_LiDAR_1ft_2010_Improved_NYC.img"
    DEFAULT_ROADS_FILENAME = "geo_export_streets.shp"
    
    # Supported file formats
    SUPPORTED_DEM_EXTENSIONS = ['.tif', '.tiff', '.img', '.hgt', '.bil', '.asc']
    SUPPORTED_VECTOR_EXTENSIONS = ['.shp', '.geojson', '.gpkg']
    
    # Default analysis parameters
    DEFAULT_WATER_LEVELS = [5, 10, 15, 20]  # feet above sea level
    DEFAULT_FLOOD_LEVEL = 10  # feet
    DEFAULT_SAMPLE_SIZE = 50000  # for visualization performance
    DEFAULT_ROAD_SAMPLE_DISTANCE = 10  # map units between road elevation samples
    
    # Elevation validation ranges
    MIN_REASONABLE_ELEVATION = -1000  # feet (below sea level)
    MAX_REASONABLE_ELEVATION = 15000  # feet (mountain peaks)
    
    # Risk classification thresholds
    RISK_THRESHOLDS = {
        'high': 0.7,      # > 0.7 = high risk
        'medium': 0.4,    # 0.4-0.7 = medium risk
        'low': 0.4        # <= 0.4 = low risk
    }
    
    # Flow accumulation parameters
    FLOW_ACCUMULATION_THRESHOLD = 1000  # minimum accumulation for significant flow
    D8_FLOW_DIRECTIONS = [
        (1, (0, 1)),   # East
        (2, (1, 1)),   # Southeast  
        (3, (1, 0)),   # South
        (4, (1, -1)),  # Southwest
        (5, (0, -1)),  # West
        (6, (-1, -1)), # Northwest
        (7, (-1, 0)),  # North
        (8, (-1, 1))   # Northeast
    ]
    
    # Coordinate system settings
    NYC_STATE_PLANE_EPSG = 2263  # NY State Plane Long Island (feet)
    WGS84_EPSG = 4326           # WGS84 Geographic
    UTM_ZONE_18N_EPSG = 32618   # UTM Zone 18N (meters)
    
    # Geographic bounds for validation (NYC area)
    NYC_BOUNDS = {
        'wgs84': {
            'lat_range': (40.4, 41.0),   # Latitude range for NYC
            'lon_range': (-74.3, -73.5)  # Longitude range for NYC
        },
        'state_plane': {
            'x_range': (900000, 1100000),  # X range in NY State Plane feet
            'y_range': (120000, 280000)    # Y range in NY State Plane feet
        }
    }
    
    # Visualization settings
    VISUALIZATION_CONFIG = {
        'elevation_colormap': 'terrain',
        'flood_risk_colormap': 'Reds',
        'default_figure_size': (12, 8),
        'default_dpi': 100,
        'road_colors': {
            'High': 'red',
            'Medium': 'orange', 
            'Low': 'green',
            'Unknown': 'gray'
        },
        'road_linewidths': {
            'High': 2.5,
            'Medium': 2.0,
            'Low': 1.5,
            'Unknown': 1.0
        }
    }
    
    # Performance settings
    PERFORMANCE_CONFIG = {
        'max_visualization_pixels': 2000,  # Maximum pixels per dimension for visualization
        'chunk_size_for_large_rasters': 1000,  # Chunk size for memory-efficient processing
        'progress_update_interval': 1000,  # Update progress every N iterations
        'max_memory_usage_mb': 500,  # Warning threshold for memory usage
    }
    
    # Greenwich Village bounds (for example/testing)
    GREENWICH_VILLAGE_BOUNDS = {
        'wgs84': {
            'lat_range': (40.728, 40.738),
            'lon_range': (-74.008, -73.995)
        },
        'state_plane': (983000, 199000, 990000, 206000),  # (min_x, min_y, max_x, max_y)
        'utm18n': (583000, 4511000, 585000, 4513000)
    }
    
    @classmethod
    def get_default_dem_path(cls) -> Path:
        """Get the default DEM file path."""
        return cls.DEFAULT_DATA_DIR / cls.DEFAULT_DEM_FILENAME
    
    @classmethod
    def get_default_roads_path(cls) -> Path:
        """Get the default roads shapefile path."""
        return cls.DEFAULT_DATA_DIR / cls.DEFAULT_ROADS_FILENAME
    
    @classmethod
    def get_output_path(cls, filename: str, output_dir: Path = None) -> Path:
        """Get full output file path."""
        if output_dir is None:
            output_dir = cls.DEFAULT_OUTPUT_DIR
        return output_dir / filename
    
    @classmethod
    def validate_water_level(cls, water_level: float) -> bool:
        """Check if water level is within reasonable range."""
        return cls.MIN_REASONABLE_ELEVATION <= water_level <= cls.MAX_REASONABLE_ELEVATION
    
    @classmethod
    def get_risk_category(cls, risk_score: float) -> str:
        """Classify risk score into category."""
        if risk_score > cls.RISK_THRESHOLDS['high']:
            return 'High'
        elif risk_score > cls.RISK_THRESHOLDS['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    @classmethod
    def get_visualization_sample_size(cls, total_pixels: int) -> int:
        """Calculate appropriate sample size for visualization."""
        max_pixels = cls.PERFORMANCE_CONFIG['max_visualization_pixels']
        if total_pixels <= max_pixels * max_pixels:
            return total_pixels
        else:
            # Calculate step size to reduce to manageable size
            step_size = int((total_pixels / (max_pixels * max_pixels)) ** 0.5) + 1
            return total_pixels // (step_size * step_size)


class EnvironmentConfig:
    """
    Environment-specific configuration settings.
    
    Handles environment variables and system-specific settings.
    """
    
    @staticmethod
    def get_data_directory() -> Path:
        """Get data directory from environment or use default."""
        data_dir = os.getenv('FLOOD_ANALYSIS_DATA_DIR')
        if data_dir:
            return Path(data_dir)
        return FloodAnalysisConfig.DEFAULT_DATA_DIR
    
    @staticmethod
    def get_output_directory() -> Path:
        """Get output directory from environment or use default."""
        output_dir = os.getenv('FLOOD_ANALYSIS_OUTPUT_DIR')
        if output_dir:
            return Path(output_dir)
        return FloodAnalysisConfig.DEFAULT_OUTPUT_DIR
    
    @staticmethod
    def get_max_memory_mb() -> int:
        """Get maximum memory usage from environment or use default."""
        max_memory = os.getenv('FLOOD_ANALYSIS_MAX_MEMORY_MB')
        if max_memory:
            try:
                return int(max_memory)
            except ValueError:
                pass
        return FloodAnalysisConfig.PERFORMANCE_CONFIG['max_memory_usage_mb']
    
    @staticmethod
    def is_debug_mode() -> bool:
        """Check if debug mode is enabled."""
        return os.getenv('FLOOD_ANALYSIS_DEBUG', '').lower() in ('1', 'true', 'yes')


class LoggingConfig:
    """
    Logging configuration settings.
    """
    
    DEFAULT_LOG_LEVEL = 'INFO'
    DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    @staticmethod
    def get_log_level() -> str:
        """Get log level from environment or use default."""
        return os.getenv('FLOOD_ANALYSIS_LOG_LEVEL', LoggingConfig.DEFAULT_LOG_LEVEL)
    
    @staticmethod
    def get_log_file() -> Path:
        """Get log file path from environment or use default."""
        log_file = os.getenv('FLOOD_ANALYSIS_LOG_FILE')
        if log_file:
            return Path(log_file)
        return EnvironmentConfig.get_output_directory() / 'flood_analysis.log'


# Global configuration instance
config = FloodAnalysisConfig()
env_config = EnvironmentConfig()
log_config = LoggingConfig()