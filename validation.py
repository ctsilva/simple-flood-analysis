#!/usr/bin/env python3
"""
Input validation and error handling utilities for flood analysis toolkit.

This module provides comprehensive validation functions for file paths,
data types, parameter ranges, and geospatial data integrity.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class FileValidationError(ValidationError):
    """Exception raised for file-related validation errors."""
    pass


class DataValidationError(ValidationError):
    """Exception raised for data integrity validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Exception raised for parameter validation errors."""
    pass


def validate_file_path(file_path: Union[str, Path], 
                      file_type: str = "file",
                      required_extensions: Optional[list[str]] = None) -> Path:
    """
    Validate that a file path exists and has the correct extension.
    
    Args:
        file_path: Path to validate
        file_type: Type description for error messages
        required_extensions: List of allowed file extensions (e.g., ['.tif', '.img'])
        
    Returns:
        Validated Path object
        
    Raises:
        FileValidationError: If file doesn't exist or has wrong extension
    """
    if file_path is None:
        raise FileValidationError(f"{file_type} path cannot be None")
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileValidationError(f"{file_type} not found: {path}")
    
    if not path.is_file():
        raise FileValidationError(f"Path is not a file: {path}")
    
    if required_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in required_extensions]:
            raise FileValidationError(
                f"{file_type} must have one of these extensions: {required_extensions}. "
                f"Got: {path.suffix}"
            )
    
    return path


def validate_dem_file(dem_path: Union[str, Path]) -> Path:
    """
    Validate DEM raster file.
    
    Args:
        dem_path: Path to DEM file
        
    Returns:
        Validated Path object
        
    Raises:
        FileValidationError: If DEM file is invalid
        DataValidationError: If DEM data is corrupted
    """
    # Validate file existence and extension
    path = validate_file_path(
        dem_path, 
        "DEM file", 
        ['.tif', '.tiff', '.img', '.hgt', '.bil']
    )
    
    # Validate DEM data integrity
    try:
        with rasterio.open(path) as src:
            # Check basic properties
            if src.count == 0:
                raise DataValidationError(f"DEM file has no bands: {path}")
            
            if src.width == 0 or src.height == 0:
                raise DataValidationError(f"DEM file has zero dimensions: {path}")
            
            # Check if we can read a sample of the data
            sample = src.read(1, window=rasterio.windows.Window(0, 0, 10, 10))
            if sample is None:
                raise DataValidationError(f"Cannot read DEM data: {path}")
            
            # Check for reasonable elevation values (basic sanity check)
            if src.nodata is not None:
                valid_data = sample[sample != src.nodata]
            else:
                valid_data = sample.flatten()
            
            if len(valid_data) > 0:
                min_elev, max_elev = float(np.min(valid_data)), float(np.max(valid_data))
                # Very basic sanity check for Earth elevations
                if min_elev < -15000 or max_elev > 15000:
                    raise DataValidationError(
                        f"DEM contains unrealistic elevation values: "
                        f"min={min_elev:.2f}, max={max_elev:.2f}"
                    )
            
    except rasterio.RasterioIOError as e:
        raise DataValidationError(f"Cannot open DEM file {path}: {e}")
    
    return path


def validate_shapefile(shapefile_path: Union[str, Path],
                      geometry_types: Optional[list[str]] = None) -> Path:
    """
    Validate shapefile.
    
    Args:
        shapefile_path: Path to shapefile
        geometry_types: Allowed geometry types (e.g., ['LineString', 'MultiLineString'])
        
    Returns:
        Validated Path object
        
    Raises:
        FileValidationError: If shapefile is invalid
        DataValidationError: If shapefile data is corrupted
    """
    # Validate file existence and extension
    path = validate_file_path(shapefile_path, "Shapefile", ['.shp'])
    
    # Check for required shapefile components
    required_files = ['.shp', '.shx', '.dbf']
    missing_files = []
    
    for ext in required_files:
        component_file = path.with_suffix(ext)
        if not component_file.exists():
            missing_files.append(str(component_file))
    
    if missing_files:
        raise FileValidationError(
            f"Missing required shapefile components: {missing_files}"
        )
    
    # Validate shapefile data integrity
    try:
        gdf = gpd.read_file(path)
        
        if len(gdf) == 0:
            raise DataValidationError(f"Shapefile contains no features: {path}")
        
        if geometry_types:
            actual_types = set(gdf.geometry.geom_type.unique())
            allowed_types = set(geometry_types)
            
            if not actual_types.issubset(allowed_types):
                invalid_types = actual_types - allowed_types
                raise DataValidationError(
                    f"Shapefile contains invalid geometry types: {invalid_types}. "
                    f"Allowed types: {allowed_types}"
                )
        
        # Check for valid CRS
        if gdf.crs is None:
            raise DataValidationError(f"Shapefile has no coordinate reference system: {path}")
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Cannot read shapefile {path}: {e}")
    
    return path


def validate_coordinate_system_compatibility(dem_crs: CRS, 
                                           shapefile_crs: CRS) -> bool:
    """
    Check if coordinate systems are compatible (same or can be reprojected).
    
    Args:
        dem_crs: DEM coordinate reference system
        shapefile_crs: Shapefile coordinate reference system
        
    Returns:
        True if compatible
        
    Raises:
        DataValidationError: If CRS are incompatible
    """
    if dem_crs is None:
        raise DataValidationError("DEM has no coordinate reference system")
    
    if shapefile_crs is None:
        raise DataValidationError("Shapefile has no coordinate reference system")
    
    # If they're the same, no problem
    if dem_crs == shapefile_crs:
        return True
    
    # Check if both have EPSG codes (generally indicates they can be reprojected)
    try:
        dem_epsg = dem_crs.to_epsg()
        shp_epsg = shapefile_crs.to_epsg()
        
        if dem_epsg is None or shp_epsg is None:
            raise DataValidationError(
                f"Cannot determine EPSG codes for coordinate systems. "
                f"DEM CRS: {dem_crs}, Shapefile CRS: {shapefile_crs}"
            )
        
        return True
        
    except Exception as e:
        raise DataValidationError(
            f"Coordinate systems appear incompatible: "
            f"DEM CRS: {dem_crs}, Shapefile CRS: {shapefile_crs}. Error: {e}"
        )


def validate_numeric_parameter(value: Any,
                             param_name: str,
                             min_value: Optional[float] = None,
                             max_value: Optional[float] = None,
                             allow_none: bool = False) -> float:
    """
    Validate numeric parameter.
    
    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is acceptable
        
    Returns:
        Validated numeric value
        
    Raises:
        ParameterValidationError: If parameter is invalid
    """
    if value is None:
        if allow_none:
            return None
        else:
            raise ParameterValidationError(f"{param_name} cannot be None")
    
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ParameterValidationError(
            f"{param_name} must be numeric, got {type(value).__name__}: {value}"
        )
    
    if np.isnan(numeric_value) or np.isinf(numeric_value):
        raise ParameterValidationError(
            f"{param_name} cannot be NaN or infinite: {numeric_value}"
        )
    
    if min_value is not None and numeric_value < min_value:
        raise ParameterValidationError(
            f"{param_name} must be >= {min_value}, got {numeric_value}"
        )
    
    if max_value is not None and numeric_value > max_value:
        raise ParameterValidationError(
            f"{param_name} must be <= {max_value}, got {numeric_value}"
        )
    
    return numeric_value


def validate_output_directory(output_dir: Union[str, Path], 
                            create_if_missing: bool = True) -> Path:
    """
    Validate output directory.
    
    Args:
        output_dir: Path to output directory
        create_if_missing: Whether to create directory if it doesn't exist
        
    Returns:
        Validated Path object
        
    Raises:
        FileValidationError: If directory is invalid
    """
    if output_dir is None:
        raise FileValidationError("Output directory cannot be None")
    
    path = Path(output_dir)
    
    if path.exists():
        if not path.is_dir():
            raise FileValidationError(f"Output path exists but is not a directory: {path}")
    else:
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileValidationError(f"Cannot create output directory {path}: {e}")
        else:
            raise FileValidationError(f"Output directory does not exist: {path}")
    
    # Check write permissions
    test_file = path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise FileValidationError(f"No write permission for output directory {path}: {e}")
    
    return path


def validate_bounds(bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Validate bounding box coordinates.
    
    Args:
        bounds: Tuple of (min_x, min_y, max_x, max_y)
        
    Returns:
        Validated bounds tuple
        
    Raises:
        ParameterValidationError: If bounds are invalid
    """
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
        raise ParameterValidationError(
            f"Bounds must be a tuple/list of 4 numbers (min_x, min_y, max_x, max_y), "
            f"got {type(bounds).__name__} with {len(bounds) if hasattr(bounds, '__len__') else 'unknown'} elements"
        )
    
    try:
        min_x, min_y, max_x, max_y = [float(b) for b in bounds]
    except (TypeError, ValueError):
        raise ParameterValidationError(f"All bounds values must be numeric: {bounds}")
    
    if min_x >= max_x:
        raise ParameterValidationError(
            f"min_x ({min_x}) must be less than max_x ({max_x})"
        )
    
    if min_y >= max_y:
        raise ParameterValidationError(
            f"min_y ({min_y}) must be less than max_y ({max_y})"
        )
    
    # Basic sanity check for Earth coordinates
    if min_x < -180 or max_x > 180:
        raise ParameterValidationError(
            f"X coordinates appear to be outside valid longitude range: {min_x}, {max_x}"
        )
    
    if min_y < -90 or max_y > 90:
        raise ParameterValidationError(
            f"Y coordinates appear to be outside valid latitude range: {min_y}, {max_y}"
        )
    
    return (min_x, min_y, max_x, max_y)