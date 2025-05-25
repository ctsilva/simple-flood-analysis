# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a geospatial flood risk analysis toolkit for urban environments, specifically designed for NYC data analysis. The repository contains Python tools for processing Digital Elevation Models (DEMs), road network shapefiles, and performing comprehensive flood risk assessments.

## Core Architecture

### Main Analysis Components

1. **`elevation_data_analyzer.py`** - Core DEM analysis engine with two main classes:
   - `ElevationDataAnalyzer`: Comprehensive elevation data analysis, statistics, and visualization
   - `RoadNetworkAnalyzer`: Road network processing and flood risk assessment
   - Supports combined elevation + road visualization overlays

2. **`nyc_data_clipping.py`** - Geospatial data clipping toolkit:
   - `GeospatialDataClipper`: Multiple clipping methods (bounding box, coordinates, DEM bounds, etc.)
   - Handles coordinate system reprojection automatically
   - Specialized DEM bounds clipping for matching vector/raster extents

3. **`raster_clipping_methods.py`** - Memory-efficient raster processing:
   - `MemoryEfficientRasterClipper`: Large raster clipping without memory overload
   - Multiple clipping strategies (window-based, GDAL command-line, chunk processing)

4. **`flood_analysis.py`** - Pure Python flood risk analysis:
   - `PythonFloodAnalysis`: Watershed analysis, flow accumulation, risk scoring using numpy/scipy
   - No external GIS dependencies required

5. **`nyc_data_download_guide.py`** - Data acquisition automation:
   - `NYCDataDownloader`: Automated download from NYC Open Data portal
   - Alternative data source guidance

## Command Line Usage

### Primary Analysis Workflows

```bash
# DEM analysis only
python elevation_data_analyzer.py dem_file.img

# DEM + Road flood risk analysis
python elevation_data_analyzer.py dem_file.img --roads roads.shp

# Quick summary mode
python elevation_data_analyzer.py dem_file.img --roads roads.shp --quick
```

### Data Preprocessing

```bash
# Clip shapefile to DEM bounds (ensures matching extents)
python nyc_data_clipping.py --dem dem_file.img --shapefile roads.shp

# Memory-efficient raster clipping
python raster_clipping_methods.py large_dem.img
```

### Data Download

```bash
# Download NYC datasets
python nyc_data_download_guide.py
```

## Key Design Patterns

### Coordinate System Handling
- All tools automatically detect and handle CRS mismatches
- Reprojection is performed transparently when needed
- DEM CRS is typically used as the target coordinate system

### Memory Management
- Large raster files are processed using windowing and sampling
- Visualization uses data sampling for performance (configurable sample_size parameters)
- Chunk processing available for extremely large datasets

### Analysis Pipeline
1. **Data Loading**: DEM and vector data loaded with metadata extraction
2. **Coordinate Alignment**: Automatic reprojection to common CRS
3. **Elevation Sampling**: Extract elevation values along road segments
4. **Risk Assessment**: Multi-level flood risk classification
5. **Visualization**: Combined elevation + risk overlay maps

## Data Directory Structure

```
nyc_data/
├── DEM_LiDAR_1ft_2010_Improved_NYC.img     # Primary DEM file
├── geo_export_*.shp                         # Road network shapefile
└── [clipped outputs]                        # Generated clipped datasets
```

## Integration Points

### Between Analysis Components
- `elevation_data_analyzer.py` can accept pre-clipped shapefiles from `nyc_data_clipping.py`
- `raster_clipping_methods.py` can prepare DEM tiles for analysis
- All tools share common coordinate system handling patterns

### Visualization Strategy
- Primary: Combined elevation map + road risk overlay
- Secondary: Individual component visualizations
- Uses matplotlib with terrain colormap for elevation, risk-based color coding for roads

## Dependencies

Core geospatial stack:
- `rasterio`: Raster data I/O and processing
- `geopandas`: Vector data processing and analysis
- `matplotlib`: Visualization and plotting
- `numpy`: Numerical computations
- `shapely`: Geometric operations

Additional libraries:
- `scipy`: Scientific computing for flow accumulation and watershed analysis
- `requests`: Data download automation

## Common Development Patterns

When adding new analysis methods:
1. Follow the established class-based architecture
2. Include comprehensive docstrings with Args/Returns
3. Handle coordinate system mismatches automatically
4. Provide both programmatic API and command-line interface
5. Use progress indicators for long-running operations
6. Include data validation and error handling

### NumPy Integer Formatting
- Always wrap numpy integers with `int()` before using comma formatting in f-strings
- Example: `f"{int(np_value):,}"` instead of `f"{np_value:,}"`
- This prevents "Unknown format code" errors when numpy returns specific integer types

## Recent Improvements

### Code Modernization (2025)
- Added comprehensive type annotations throughout codebase
- Implemented structured logging system with progress tracking
- Created robust input validation with custom exception hierarchy
- Added pytest testing framework with fixtures and coverage
- Centralized configuration management system
- Eliminated GRASS GIS dependencies with pure Python alternatives
- Fixed numpy integer formatting issues in f-string expressions