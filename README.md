# Simple Flood Analysis

A geospatial flood risk analysis toolkit for urban environments, specifically designed for NYC data analysis. This repository contains Python tools for processing Digital Elevation Models (DEMs), road network shapefiles, and performing comprehensive flood risk assessments.

## Features

- **Elevation Data Analysis**: Comprehensive DEM analysis with statistics and visualization
- **Road Network Analysis**: Flood risk assessment for transportation infrastructure
- **Geospatial Data Processing**: Automated clipping, reprojection, and coordinate system handling
- **Memory-Efficient Processing**: Handle large raster datasets without memory overload
- **Automated Data Download**: Integration with NYC Open Data portal
- **Advanced Flood Modeling**: Pure Python watershed analysis and flow accumulation

## Quick Start

### Basic Analysis

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
python nyc_data_download_guide2.py
```

## Core Components

### Analysis Modules

- **`elevation_data_analyzer.py`** - Core DEM analysis engine with elevation statistics and road network flood risk assessment
- **`nyc_data_clipping.py`** - Geospatial data clipping toolkit with multiple clipping methods
- **`raster_clipping_methods.py`** - Memory-efficient raster processing for large datasets
- **`flood_analysis.py`** - Pure Python flood risk analysis with watershed modeling
- **`nyc_data_download_guide2.py`** - Automated data acquisition from NYC Open Data

### Key Classes

- `ElevationDataAnalyzer`: Comprehensive elevation data analysis and visualization
- `RoadNetworkAnalyzer`: Road network processing and flood risk assessment
- `GeospatialDataClipper`: Multiple clipping methods with automatic reprojection
- `MemoryEfficientRasterClipper`: Large raster processing strategies
- `PythonFloodAnalysis`: Pure Python watershed analysis and flow accumulation
- `NYCDataDownloader`: Automated dataset downloads

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple-flood-analysis.git
cd simple-flood-analysis

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

### Core Requirements
- `rasterio` - Raster data I/O and processing
- `geopandas` - Vector data processing and analysis
- `matplotlib` - Visualization and plotting
- `numpy` - Numerical computations
- `shapely` - Geometric operations

### Additional Libraries
- `scipy` - Scientific computing for watershed analysis
- `requests` - Data download automation

## Data Requirements

This toolkit requires:
- **DEM files**: Digital Elevation Models (typically .img or .tif format)
- **Shapefile data**: Road network or other vector data (.shp format)

Place your data files in a `nyc_data/` directory or specify paths directly in the command line.

## Analysis Pipeline

1. **Data Loading**: DEM and vector data loaded with metadata extraction
2. **Coordinate Alignment**: Automatic reprojection to common CRS
3. **Elevation Sampling**: Extract elevation values along road segments
4. **Risk Assessment**: Multi-level flood risk classification
5. **Visualization**: Combined elevation + risk overlay maps

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NYC Open Data portal for providing comprehensive geospatial datasets
- SciPy community for scientific computing algorithms
- Python geospatial ecosystem (GDAL, Rasterio, GeoPandas)