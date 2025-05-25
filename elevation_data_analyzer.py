#!/usr/bin/env python3
"""
NYC Elevation Data Analyzer

A comprehensive tool for analyzing Digital Elevation Model (DEM) data, specifically designed 
for urban flood risk assessment and elevation analysis. This script provides statistical 
analysis, visualization, and flood risk evaluation capabilities for raster elevation data.

Key Features:
- Load and analyze DEM raster files (supports multiple formats via rasterio)
- Calculate comprehensive elevation statistics
- Generate histogram and elevation map visualizations  
- Perform flood risk analysis at different water levels
- Data quality assessment and validation
- Command line interface for easy usage

Dependencies:
- rasterio: Reading geospatial raster data
- numpy: Numerical computations
- matplotlib: Data visualization
- seaborn: Statistical plotting (imported but may be unused)
- pathlib: File path handling
- pandas: Data manipulation (imported but may be unused)

Usage:
    python elevation_data_analyzer.py [dem_file_path]
    
    If no file path is provided, uses default NYC DEM file.
    Offers interactive choice between complete analysis or quick summary.

Author: Elevation Analysis Tool
Version: 1.0
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Any

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import Counter
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS
from shapely.geometry import Point, LineString
import warnings
warnings.filterwarnings('ignore')

class ElevationDataAnalyzer:
    """
    A comprehensive analyzer for Digital Elevation Model (DEM) data.
    
    This class provides methods to load, analyze, and visualize elevation data
    from raster files, with specific focus on flood risk assessment and 
    urban elevation analysis.
    
    Attributes:
        dem_path (Path): Path to the DEM file
        elevation_data (numpy.ndarray): 2D array of elevation values
        src_info (dict): Metadata about the raster source (CRS, bounds, etc.)
        valid_data (numpy.ndarray): 1D array of valid elevation values (no-data removed)
    """
    def __init__(self, dem_path: Union[str, Path]) -> None:
        """Initialize the analyzer with a DEM file path.
        
        Args:
            dem_path: Path to the DEM raster file
            
        Raises:
            FileNotFoundError: If the DEM file doesn't exist
        """
        self.dem_path: Path = Path(dem_path)
        self.elevation_data: Optional[np.ndarray] = None
        self.src_info: Optional[Dict[str, Any]] = None
        self.valid_data: Optional[np.ndarray] = None
        self.load_data()
    
    def load_data(self) -> bool:
        """Load the DEM data and extract metadata.
        
        Reads the elevation raster and stores both the data array and 
        metadata including CRS, bounds, resolution, and data type.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        print(f"Loading elevation data from: {self.dem_path.name}")
        print("=" * 60)
        
        try:
            with rasterio.open(self.dem_path) as src:
                # Store source information
                self.src_info = {
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': src.nodata,
                    'res': src.res
                }
                
                # Read elevation data
                print("Reading elevation data...")
                self.elevation_data = src.read(1)
                
                print("✅ Data loaded successfully!")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def basic_info(self) -> None:
        """Display comprehensive information about the elevation dataset.
        
        Prints formatted information including:
        - File format and dimensions
        - Coordinate reference system
        - Spatial resolution and coverage
        - Estimated real-world area (if CRS is known)
        """
        print("\n📊 BASIC DATASET INFORMATION")
        print("-" * 40)
        
        info = self.src_info
        
        print(f"File Format: {self.dem_path.suffix.upper()}")
        print(f"Dimensions: {info['width']:,} x {info['height']:,} pixels")
        print(f"Total Pixels: {info['width'] * info['height']:,}")
        print(f"Data Type: {info['dtype']}")
        print(f"Number of Bands: {info['count']}")
        print(f"No Data Value: {info['nodata']}")
        
        # File size
        file_size_mb = self.dem_path.stat().st_size / (1024 * 1024)
        print(f"File Size: {file_size_mb:.1f} MB")
        
        # Coordinate system
        print(f"\n🗺️ GEOGRAPHIC INFORMATION")
        print("-" * 30)
        print(f"Coordinate System: {info['crs']}")
        if info['crs']:
            print(f"EPSG Code: {info['crs'].to_epsg()}")
        
        # Resolution
        print(f"Pixel Resolution: {info['res'][0]:.2f} x {info['res'][1]:.2f} units")
        
        # Calculate area coverage
        bounds = info['bounds']
        width_units = bounds.right - bounds.left
        height_units = bounds.top - bounds.bottom
        
        print(f"\n📏 SPATIAL COVERAGE")
        print("-" * 25)
        print(f"Bounds (left, bottom, right, top):")
        print(f"  {bounds.left:.2f}, {bounds.bottom:.2f}, {bounds.right:.2f}, {bounds.top:.2f}")
        print(f"Width: {width_units:.2f} units")
        print(f"Height: {height_units:.2f} units")
        
        # Estimate real-world area
        if info['crs'] and info['crs'].to_epsg() == 2263:  # NY State Plane (feet)
            area_sq_ft = width_units * height_units
            area_sq_miles = area_sq_ft / (5280 * 5280)
            print(f"Area: {area_sq_miles:.1f} square miles")
        elif info['crs'] and info['crs'].to_epsg() in [32618, 32617]:  # UTM (meters)
            area_sq_m = width_units * height_units
            area_sq_km = area_sq_m / (1000 * 1000)
            print(f"Area: {area_sq_km:.1f} square kilometers")
    
    def elevation_statistics(self) -> Optional[Dict[str, float]]:
        """Calculate comprehensive elevation statistics.
        
        Computes descriptive statistics, percentiles, and sea level analysis
        for valid elevation data (excluding no-data values). Stores the
        processed valid data as a class member for use by other methods.
        
        Returns:
            Dictionary containing statistical measures, or None if no valid data
        """
        print("\n🏔️ ELEVATION STATISTICS")
        print("-" * 30)
        
        # Handle no-data values
        if self.src_info['nodata'] is not None:
            self.valid_data = self.elevation_data[self.elevation_data != self.src_info['nodata']]
        else:
            self.valid_data = self.elevation_data.flatten()
        
        # Remove any remaining invalid values
        self.valid_data = self.valid_data[~np.isnan(self.valid_data)]
        self.valid_data = self.valid_data[np.isfinite(self.valid_data)]
        
        if len(self.valid_data) == 0:
            print("❌ No valid elevation data found!")
            return None
        
        # Basic statistics
        stats = {
            'count': len(self.valid_data),
            'min': np.min(self.valid_data),
            'max': np.max(self.valid_data),
            'mean': np.mean(self.valid_data),
            'median': np.median(self.valid_data),
            'std': np.std(self.valid_data),
            'range': np.max(self.valid_data) - np.min(self.valid_data)
        }
        
        print(f"Valid Pixels: {stats['count']:,} ({stats['count']/(self.src_info['width']*self.src_info['height'])*100:.1f}%)")
        print(f"Minimum Elevation: {stats['min']:.2f}")
        print(f"Maximum Elevation: {stats['max']:.2f}")
        print(f"Mean Elevation: {stats['mean']:.2f}")
        print(f"Median Elevation: {stats['median']:.2f}")
        print(f"Standard Deviation: {stats['std']:.2f}")
        print(f"Elevation Range: {stats['range']:.2f}")
        
        # Percentiles
        percentiles = [5, 10, 25, 75, 90, 95]
        print(f"\n📈 ELEVATION PERCENTILES")
        print("-" * 25)
        for p in percentiles:
            value = np.percentile(self.valid_data, p)
            print(f"{p}th percentile: {value:.2f}")
        
        # Sea level analysis
        print(f"\n🌊 SEA LEVEL ANALYSIS")
        print("-" * 23)
        below_sea_level = np.sum(self.valid_data < 0)
        at_sea_level = np.sum((self.valid_data >= 0) & (self.valid_data <= 1))
        low_elevation = np.sum((self.valid_data > 1) & (self.valid_data <= 10))
        
        print(f"Below Sea Level (<0): {below_sea_level:,} pixels ({below_sea_level/len(self.valid_data)*100:.1f}%)")
        print(f"At Sea Level (0-1): {at_sea_level:,} pixels ({at_sea_level/len(self.valid_data)*100:.1f}%)")
        print(f"Low Elevation (1-10): {low_elevation:,} pixels ({low_elevation/len(self.valid_data)*100:.1f}%)")
        
        return stats
    
    def elevation_distribution(self) -> Tuple[List[int], List[str]]:
        """Analyze elevation distribution across predefined bins.
        
        Categorizes elevation data into meaningful elevation ranges
        and calculates pixel counts and percentages for each bin.
        Uses the class's valid_data computed by elevation_statistics().
            
        Returns:
            Tuple of (bin_counts, bin_labels) for further analysis
        """
        print(f"\n📊 ELEVATION DISTRIBUTION ANALYSIS")
        print("-" * 35)
        
        # Create elevation bins for analysis
        bins = [-50, -10, -5, 0, 5, 10, 20, 50, 100, 200, 500, 1000]
        bin_counts = []
        bin_labels = []
        
        for i in range(len(bins)-1):
            count = np.sum((self.valid_data >= bins[i]) & (self.valid_data < bins[i+1]))
            percentage = count / len(self.valid_data) * 100
            bin_counts.append(count)
            bin_labels.append(f"{bins[i]} to {bins[i+1]}")
            print(f"{bins[i]:>4} to {bins[i+1]:>4} ft: {count:>15,} pixels ({percentage:>5.1f}%)")
        
        # Handle values above highest bin
        above_max = np.sum(self.valid_data >= bins[-1])
        if above_max > 0:
            percentage = above_max / len(self.valid_data) * 100
            print(f"{bins[-1]:>4}+ ft:      {above_max:>8,} pixels ({percentage:>5.1f}%)")
        
        return bin_counts, bin_labels
    
    def create_visualizations(self, sample_size: int = 50000) -> None:
        """Generate visualizations of elevation data.
        
        Creates two key visualizations:
        1. Histogram showing elevation distribution with sea level reference
        2. Spatial elevation map using terrain colormap
        
        Uses the class's valid_data computed by elevation_statistics().
        
        Args:
            sample_size: Maximum number of points to use for histogram 
                        (for performance with large datasets)
        """
        print(f"\n📈 CREATING VISUALIZATIONS")
        print("-" * 30)
        
        # Sample data for faster plotting if dataset is large
        if len(self.valid_data) > sample_size:
            plot_data = np.random.choice(self.valid_data, sample_size, replace=False)
            print(f"Using {sample_size:,} random samples for plotting")
        else:
            plot_data = self.valid_data
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('NYC Elevation Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0].hist(plot_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Elevation Distribution')
        axes[0].set_xlabel('Elevation (feet)')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Sea Level')
        axes[0].legend()
        
        # 2. Elevation map (sample)
        # Show a sample of the elevation data
        sample_rows = min(1000, self.elevation_data.shape[0])
        sample_cols = min(1000, self.elevation_data.shape[1])
        row_step = max(1, self.elevation_data.shape[0] // sample_rows)
        col_step = max(1, self.elevation_data.shape[1] // sample_cols)
        
        elevation_sample = self.elevation_data[::row_step, ::col_step]
        
        # Handle no-data values for plotting
        if self.src_info['nodata'] is not None:
            elevation_sample = np.where(elevation_sample == self.src_info['nodata'], 
                                      np.nan, elevation_sample)
        
        im = axes[1].imshow(elevation_sample, cmap='terrain', aspect='equal')
        axes[1].set_title('Elevation Map (Sample)')
        axes[1].set_xlabel('East-West')
        axes[1].set_ylabel('North-South')
        plt.colorbar(im, ax=axes[1], label='Elevation (feet)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_flood_risk_map(self, flood_level: float = 10, figsize: Tuple[int, int] = (12, 8)) -> Optional[matplotlib.figure.Figure]:
        """Visualize elevation map with flood risk zones overlay.
        
        Creates a map showing the elevation data with flood risk zones highlighted
        based on the specified flood level. Areas at or below the flood level
        are colored to indicate flood risk.
        
        Args:
            flood_level: Water level threshold in feet for flood analysis
            figsize: Figure size for the plot
            
        Returns:
            The matplotlib Figure object, or None if no data available
        """
        if self.elevation_data is None:
            print("❌ No elevation data available!")
            return
        
        print(f"\n🌊 FLOOD RISK ELEVATION MAP ({flood_level}ft level)")
        print("-" * 50)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sample elevation data for visualization (for performance)
        sample_rows = min(2000, self.elevation_data.shape[0])
        sample_cols = min(2000, self.elevation_data.shape[1])
        row_step = max(1, self.elevation_data.shape[0] // sample_rows)
        col_step = max(1, self.elevation_data.shape[1] // sample_cols)
        
        elevation_sample = self.elevation_data[::row_step, ::col_step]
        
        # Handle no-data values for plotting
        if self.src_info['nodata'] is not None:
            elevation_sample = np.where(elevation_sample == self.src_info['nodata'], 
                                      np.nan, elevation_sample)
        
        # Get extent for the elevation map
        bounds = self.src_info['bounds']
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # Create flood risk mask
        flood_mask = elevation_sample <= flood_level
        safe_mask = elevation_sample > flood_level
        
        # Plot base elevation map
        im_base = ax.imshow(elevation_sample, 
                           cmap='terrain', 
                           aspect='equal',
                           extent=extent,
                           alpha=0.8,
                           origin='upper')
        
        # Create flood risk overlay
        flood_overlay = np.full_like(elevation_sample, np.nan)
        flood_overlay[flood_mask] = elevation_sample[flood_mask]
        
        # Plot flood risk areas with red overlay
        im_flood = ax.imshow(flood_overlay,
                            cmap='Reds',
                            aspect='equal', 
                            extent=extent,
                            alpha=0.6,
                            origin='upper',
                            vmin=0,
                            vmax=flood_level)
        
        # Add elevation colorbar
        cbar_elev = plt.colorbar(im_base, ax=ax, shrink=0.6, pad=0.02, aspect=20)
        cbar_elev.set_label('Elevation (feet)', fontsize=12)
        
        # Add flood depth colorbar
        cbar_flood = plt.colorbar(im_flood, ax=ax, shrink=0.6, pad=0.08, aspect=20)
        cbar_flood.set_label(f'Flood Risk Areas (≤{flood_level}ft)', fontsize=12)
        
        # Calculate flood statistics
        if self.valid_data is not None:
            total_pixels = len(self.valid_data)
            flooded_pixels = np.sum(self.valid_data <= flood_level)
            flood_percentage = flooded_pixels / total_pixels * 100
            
            # Calculate area if coordinate system is known
            pixel_area = abs(self.src_info['res'][0] * self.src_info['res'][1])
            if self.src_info['crs'] and self.src_info['crs'].to_epsg() == 2263:  # NY State Plane (feet)
                flood_area_sqft = flooded_pixels * pixel_area
                flood_area_sqmiles = flood_area_sqft / (5280 * 5280)
                area_text = f"Flood Area: {flood_area_sqmiles:.2f} sq miles"
            elif self.src_info['crs'] and self.src_info['crs'].to_epsg() in [32618, 32617]:  # UTM (meters)
                flood_area_sqm = flooded_pixels * pixel_area
                flood_area_sqkm = flood_area_sqm / (1000 * 1000)
                area_text = f"Flood Area: {flood_area_sqkm:.2f} sq km"
            else:
                area_text = f"Flood Area: {flooded_pixels:,} pixels"
        else:
            flood_percentage = 0
            area_text = "No valid data"
        
        # Customize the plot
        ax.set_title(f'Elevation Map with Flood Risk Zones - {flood_level}ft Water Level', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add flood risk statistics text box
        stats_text = f'''Flood Risk Summary ({flood_level}ft):
        
Flooded Area: {flood_percentage:.1f}% of region
{area_text}
        
Legend:
🔴 Red areas: Flood risk zones
🟫 Brown/Green: Safe elevations'''
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, zorder=15)
        
        # Add grid and formatting
        ax.grid(True, alpha=0.3, zorder=1)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def flood_risk_analysis(self, water_levels: List[float] = [5, 10, 15, 20]) -> None:
        """Perform flood risk assessment at various water level scenarios.
        
        Calculates the number and percentage of pixels that would be flooded
        at different water level thresholds. Provides area estimates when
        coordinate system information is available.
        
        Uses the class's valid_data computed by elevation_statistics().
        
        Args:
            water_levels: List of water level thresholds (in feet) to analyze
        """
        print(f"\n🌊 FLOOD RISK ANALYSIS")
        print("-" * 25)
        
        total_pixels = len(self.valid_data)
        
        print("Potential flood zones at different water levels:")
        print("(assuming water level = elevation threshold)")
        print()
        
        for water_level in water_levels:
            flooded_pixels = np.sum(self.valid_data <= water_level)
            percentage = flooded_pixels / total_pixels * 100
            
            print(f"Water Level {water_level:2d} ft: {flooded_pixels:>8,} pixels ({percentage:>5.1f}% of area)")
        
        # Calculate area if we know the coordinate system
        pixel_area = abs(self.src_info['res'][0] * self.src_info['res'][1])
        
        if self.src_info['crs'] and self.src_info['crs'].to_epsg() == 2263:  # NY State Plane (feet)
            print(f"\nPixel area: {pixel_area:.1f} square feet")
            print(f"Each 1% of area ≈ {total_pixels * pixel_area / 100 / (5280*5280):.2f} square miles")
        elif self.src_info['crs'] and self.src_info['crs'].to_epsg() in [32618, 32617]:  # UTM (meters)
            print(f"\nPixel area: {pixel_area:.1f} square meters")
            print(f"Each 1% of area ≈ {total_pixels * pixel_area / 100 / 1000000:.2f} square kilometers")
    
    def data_quality_check(self) -> None:
        """Perform comprehensive data quality assessment.
        
        Checks for:
        - No-data value distribution
        - Statistical outliers (beyond 3 standard deviations)
        - Data type and precision analysis
        - Potential data anomalies
        """
        print(f"\n🔍 DATA QUALITY CHECK")
        print("-" * 25)
        
        # Check for no-data values
        if self.src_info['nodata'] is not None:
            nodata_count = np.sum(self.elevation_data == self.src_info['nodata'])
            nodata_percent = nodata_count / (self.src_info['width'] * self.src_info['height']) * 100
            print(f"No-data pixels: {nodata_count:,} ({nodata_percent:.1f}%)")
        
        # Check for extreme values
        valid_mask = self.elevation_data != self.src_info['nodata'] if self.src_info['nodata'] is not None else np.ones_like(self.elevation_data, dtype=bool)
        valid_elevations = self.elevation_data[valid_mask]
        
        # Identify outliers (beyond 3 standard deviations)
        mean_elev = np.mean(valid_elevations)
        std_elev = np.std(valid_elevations)
        outlier_threshold = 3 * std_elev
        
        low_outliers = np.sum(valid_elevations < (mean_elev - outlier_threshold))
        high_outliers = np.sum(valid_elevations > (mean_elev + outlier_threshold))
        
        print(f"Potential outliers:")
        print(f"  Extremely low: {low_outliers:,} pixels (< {mean_elev - outlier_threshold:.1f} ft)")
        print(f"  Extremely high: {high_outliers:,} pixels (> {mean_elev + outlier_threshold:.1f} ft)")
        
        # Check data type and precision
        print(f"\nData precision:")
        print(f"  Data type: {self.src_info['dtype']}")
        
        # Check for integer vs float precision
        if np.issubdtype(self.elevation_data.dtype, np.integer):
            print(f"  Integer data - elevation precision: 1 ft")
        else:
            # Check decimal precision for float data
            sample_data = valid_elevations[:1000]  # Sample for speed
            decimal_places = []
            for val in sample_data:
                if val != int(val):
                    decimal_str = f"{val:.10f}".rstrip('0')
                    if '.' in decimal_str:
                        decimal_places.append(len(decimal_str.split('.')[1]))
            
            if decimal_places:
                avg_precision = np.mean(decimal_places)
                print(f"  Float data - average decimal places: {avg_precision:.1f}")
            else:
                print(f"  Float data but appears to be whole numbers")
    
    def run_complete_analysis(self) -> None:
        """Execute the full analysis workflow.
        
        Runs all analysis methods in sequence:
        1. Basic dataset information
        2. Elevation statistics
        3. Distribution analysis
        4. Visualizations
        5. Flood risk assessment
        6. Data quality check
        """
        print("NYC ELEVATION DATA COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        if self.elevation_data is None:
            print("❌ No data loaded!")
            return
        
        # Basic information
        self.basic_info()
        
        # Elevation statistics  
        stats = self.elevation_statistics()
        if stats is None:
            print("❌ Cannot continue analysis without valid data!")
            return
        
        # Distribution analysis
        self.elevation_distribution()
        
        # Visualizations
        # self.create_visualizations()
        
        # Flood risk visualization and analysis
        self.visualize_flood_risk_map(flood_level=10)
        self.flood_risk_analysis()
        
        # Data quality check
        self.data_quality_check()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 Data file: {self.dem_path}")
        print(f"📊 {len(self.valid_data):,} valid elevation points analyzed")

def quick_elevation_summary(dem_path: Union[str, Path]) -> None:
    """Generate a quick summary of elevation data without full analysis.
    
    Provides essential statistics including elevation range, mean,
    and basic flood risk indicators without creating visualizations.
    
    Args:
        dem_path: Path to the DEM file
    """
    print("QUICK ELEVATION SUMMARY")
    print("=" * 30)
    
    try:
        with rasterio.open(dem_path) as src:
            elevation = src.read(1)
            
            # Handle no-data
            if src.nodata is not None:
                valid_data = elevation[elevation != src.nodata]
            else:
                valid_data = elevation.flatten()
            
            print(f"File: {Path(dem_path).name}")
            print(f"Size: {src.width:,} x {src.height:,} pixels")
            print(f"Valid pixels: {len(valid_data):,}")
            print(f"Elevation range: {valid_data.min():.1f} to {valid_data.max():.1f}")
            print(f"Mean elevation: {valid_data.mean():.1f}")
            print(f"Below sea level: {np.sum(valid_data < 0):,} pixels")
            print(f"Low elevation (0-10 ft): {np.sum((valid_data >= 0) & (valid_data <= 10)):,} pixels")
            
    except Exception as e:
        print(f"Error: {e}")


class RoadNetworkAnalyzer:
    """
    Analyzer for road network shapefiles with flood risk assessment capabilities.
    
    This class loads road network data from shapefiles and provides methods to
    analyze flood risk for road segments based on elevation data.
    
    Attributes:
        shapefile_path (Path): Path to the road network shapefile
        roads_gdf (GeoDataFrame): GeoPandas dataframe containing road network data
        crs (CRS): Coordinate reference system of the road network
    """
    
    def __init__(self, shapefile_path: Union[str, Path]) -> None:
        """Initialize the road network analyzer.
        
        Args:
            shapefile_path: Path to the road network shapefile
            
        Raises:
            FileNotFoundError: If the shapefile doesn't exist
            ValueError: If the shapefile cannot be read
        """
        self.shapefile_path: Path = Path(shapefile_path)
        self.roads_gdf: Optional[gpd.GeoDataFrame] = None
        self.crs: Optional[CRS] = None
        self.load_road_network()
    
    def load_road_network(self) -> bool:
        """Load the road network from shapefile.
        
        Reads the shapefile using GeoPandas and stores the road network
        data along with coordinate reference system information.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        print(f"Loading road network from: {self.shapefile_path.name}")
        print("=" * 50)
        
        try:
            # Load the shapefile
            self.roads_gdf = gpd.read_file(self.shapefile_path)
            self.crs = self.roads_gdf.crs
            
            print("✅ Road network loaded successfully!")
            print(f"Number of road segments: {len(self.roads_gdf):,}")
            print(f"Coordinate Reference System: {self.crs}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading road network: {e}")
            return False
    
    def basic_road_info(self) -> None:
        """Display basic information about the road network.
        
        Prints comprehensive information about the road dataset including:
        - Number of segments and geometry types
        - Coordinate reference system
        - Spatial bounds
        - Available attribute columns
        """
        if self.roads_gdf is None:
            print("❌ No road network data loaded!")
            return
        
        print("\n🛣️ ROAD NETWORK INFORMATION")
        print("-" * 35)
        
        # Basic statistics
        print(f"Total road segments: {len(self.roads_gdf):,}")
        print(f"Geometry type: {self.roads_gdf.geometry.geom_type.iloc[0]}")
        print(f"CRS: {self.crs}")
        
        # Spatial bounds
        bounds = self.roads_gdf.total_bounds
        print(f"\n📏 SPATIAL BOUNDS")
        print("-" * 20)
        print(f"Min X: {bounds[0]:.2f}")
        print(f"Min Y: {bounds[1]:.2f}")
        print(f"Max X: {bounds[2]:.2f}")
        print(f"Max Y: {bounds[3]:.2f}")
        
        # Calculate total length if possible
        if self.crs and self.crs.is_projected:
            total_length = self.roads_gdf.geometry.length.sum()
            print(f"Total road length: {total_length/1000:.1f} km")
        
        # Available columns
        print(f"\n📋 AVAILABLE ATTRIBUTES")
        print("-" * 25)
        for col in self.roads_gdf.columns:
            if col != 'geometry':
                dtype = self.roads_gdf[col].dtype
                non_null = self.roads_gdf[col].notna().sum()
                print(f"  {col}: {dtype} ({non_null:,}/{len(self.roads_gdf):,} non-null)")
    
    def extract_road_elevations(self, dem_analyzer: ElevationDataAnalyzer) -> Optional[gpd.GeoDataFrame]:
        """Extract elevation values for road segments from DEM data.
        
        Samples elevation values along road segments and calculates
        statistics for each road segment.
        
        Args:
            dem_analyzer: Initialized DEM analyzer
            
        Returns:
            Road network with added elevation statistics, or None if extraction fails
        """
        if self.roads_gdf is None or dem_analyzer.elevation_data is None:
            print("❌ Missing road network or elevation data!")
            return None
        
        print("\n🔍 EXTRACTING ROAD ELEVATIONS")
        print("-" * 35)
        
        # Ensure coordinate systems match
        if self.crs != dem_analyzer.src_info['crs']:
            print(f"⚠️  Reprojecting roads from {self.crs} to {dem_analyzer.src_info['crs']}")
            roads_reproj = self.roads_gdf.to_crs(dem_analyzer.src_info['crs'])
        else:
            roads_reproj = self.roads_gdf.copy()
        
        # Initialize elevation columns
        roads_reproj['min_elevation'] = np.nan
        roads_reproj['max_elevation'] = np.nan
        roads_reproj['mean_elevation'] = np.nan
        roads_reproj['elevation_samples'] = 0
        
        print(f"Processing {len(roads_reproj):,} road segments...")
        
        # Process each road segment
        for idx, road in roads_reproj.iterrows():
            elevations = self._sample_elevation_along_line(
                road.geometry, 
                dem_analyzer.elevation_data,
                dem_analyzer.src_info['transform'],
                dem_analyzer.src_info['nodata']
            )
            
            if len(elevations) > 0:
                roads_reproj.loc[idx, 'min_elevation'] = np.min(elevations)
                roads_reproj.loc[idx, 'max_elevation'] = np.max(elevations)
                roads_reproj.loc[idx, 'mean_elevation'] = np.mean(elevations)
                roads_reproj.loc[idx, 'elevation_samples'] = len(elevations)
            
            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1:,} segments...")
        
        # Summary statistics
        valid_segments = roads_reproj['elevation_samples'] > 0
        print(f"\n📊 ELEVATION EXTRACTION SUMMARY")
        print("-" * 35)
        print(f"Segments with elevation data: {valid_segments.sum():,}/{len(roads_reproj):,}")
        
        if valid_segments.any():
            valid_roads = roads_reproj[valid_segments]
            print(f"Elevation range: {valid_roads['min_elevation'].min():.1f} to {valid_roads['max_elevation'].max():.1f} ft")
            print(f"Average road elevation: {valid_roads['mean_elevation'].mean():.1f} ft")
        
        return roads_reproj
    
    def _sample_elevation_along_line(self, line_geom: LineString, elevation_data: np.ndarray, 
                                   transform: Affine, nodata: Optional[float], 
                                   sample_distance: float = 10) -> List[float]:
        """Sample elevation values along a line geometry.
        
        Args:
            line_geom: Road segment geometry
            elevation_data: DEM elevation array
            transform: Raster transform
            nodata: No-data value
            sample_distance: Distance between samples in map units
            
        Returns:
            List of elevation values along the line
        """
        if not isinstance(line_geom, LineString):
            return []
        
        # Generate points along the line
        line_length = line_geom.length
        if line_length == 0:
            return []
        
        # Calculate number of samples
        num_samples = max(2, int(line_length / sample_distance))
        distances = np.linspace(0, line_length, num_samples)
        
        elevations = []
        for distance in distances:
            point = line_geom.interpolate(distance)
            
            # Convert point coordinates to raster indices
            col, row = ~transform * (point.x, point.y)
            col, row = int(col), int(row)
            
            # Check if point is within raster bounds
            if (0 <= row < elevation_data.shape[0] and 
                0 <= col < elevation_data.shape[1]):
                
                elevation = elevation_data[row, col]
                
                # Check for valid elevation
                if nodata is None or elevation != nodata:
                    if not np.isnan(elevation) and np.isfinite(elevation):
                        elevations.append(elevation)
        
        return elevations
    
    def assess_flood_risk(self, roads_with_elevations: gpd.GeoDataFrame, 
                         flood_levels: List[float] = [5, 10, 15, 20]) -> Optional[gpd.GeoDataFrame]:
        """Assess flood risk for road segments at different water levels.
        
        Args:
            roads_with_elevations: Roads with elevation data
            flood_levels: Water level thresholds for flood analysis
            
        Returns:
            Roads with flood risk classifications, or None if assessment fails
        """
        if roads_with_elevations is None:
            print("❌ No road elevation data available!")
            return None
        
        print(f"\n🌊 ROAD FLOOD RISK ASSESSMENT")
        print("-" * 35)
        
        roads_risk = roads_with_elevations.copy()
        
        # Assess risk at each flood level
        for level in flood_levels:
            risk_column = f'flood_risk_{level}ft'
            
            # Classify flood risk based on minimum elevation
            # High risk: min elevation <= flood level
            # Medium risk: min elevation <= flood level + 5
            # Low risk: min elevation > flood level + 5
            
            conditions = [
                roads_risk['min_elevation'] <= level,
                (roads_risk['min_elevation'] > level) & (roads_risk['min_elevation'] <= level + 5),
                roads_risk['min_elevation'] > level + 5
            ]
            choices = ['High', 'Medium', 'Low']
            
            roads_risk[risk_column] = np.select(conditions, choices, default='Unknown')
        
        # Summary statistics
        print("Flood risk summary (based on minimum elevation):")
        for level in flood_levels:
            risk_column = f'flood_risk_{level}ft'
            risk_counts = roads_risk[risk_column].value_counts()
            
            print(f"\n{level}ft flood level:")
            for risk_level in ['High', 'Medium', 'Low']:
                count = risk_counts.get(risk_level, 0)
                percentage = count / len(roads_risk) * 100
                print(f"  {risk_level} risk: {count:,} segments ({percentage:.1f}%)")
        
        return roads_risk
    
    def visualize_road_flood_risk(self, roads_with_risk: gpd.GeoDataFrame, 
                                 flood_level: int = 10, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Visualize road flood risk on a map.
        
        Args:
            roads_with_risk: Roads with flood risk data
            flood_level: Flood level to visualize
            figsize: Figure size for the plot
        """
        if roads_with_risk is None:
            print("❌ No road risk data available!")
            return
        
        print(f"\n🗺️ VISUALIZING FLOOD RISK ({flood_level}ft level)")
        print("-" * 45)
        
        risk_column = f'flood_risk_{flood_level}ft'
        
        if risk_column not in roads_with_risk.columns:
            print(f"❌ Risk column {risk_column} not found!")
            return
        
        # Create the plot
        _, ax = plt.subplots(figsize=figsize)
        
        # Color mapping for risk levels
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Unknown': 'gray'}
        
        # Plot roads by risk level
        for risk_level in ['Low', 'Medium', 'High', 'Unknown']:
            subset = roads_with_risk[roads_with_risk[risk_column] == risk_level]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color_map[risk_level], 
                           linewidth=1, label=f'{risk_level} Risk', alpha=0.7)
        
        ax.set_title(f'Road Network Flood Risk - {flood_level}ft Water Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_combined_elevation_and_roads(self, roads_with_risk: gpd.GeoDataFrame, 
                                             dem_analyzer: ElevationDataAnalyzer, 
                                             flood_level: int = 10, 
                                             figsize: Tuple[int, int] = (8, 5)) -> Optional[matplotlib.figure.Figure]:
        """Visualize elevation map with road flood risk overlay.
        
        Args:
            roads_with_risk: Roads with flood risk data
            dem_analyzer: DEM analyzer with elevation data
            flood_level: Flood level to visualize
            figsize: Figure size for the plot
            
        Returns:
            The matplotlib Figure object, or None if visualization fails
        """
        if roads_with_risk is None or dem_analyzer.elevation_data is None:
            print("❌ Missing road risk data or elevation data!")
            return
        
        print(f"\n🗺️ COMBINED ELEVATION & ROAD RISK VISUALIZATION ({flood_level}ft level)")
        print("-" * 65)
        
        risk_column = f'flood_risk_{flood_level}ft'
        
        if risk_column not in roads_with_risk.columns:
            print(f"❌ Risk column {risk_column} not found!")
            return
        
        # Ensure roads are in the same CRS as the DEM
        dem_crs = dem_analyzer.src_info['crs']
        if roads_with_risk.crs != dem_crs:
            print(f"⚠️  Reprojecting roads from {roads_with_risk.crs} to {dem_crs} for visualization")
            roads_for_plot = roads_with_risk.to_crs(dem_crs)
        else:
            roads_for_plot = roads_with_risk.copy()
        
        print(f"DEM CRS: {dem_crs}")
        print(f"Roads CRS: {roads_for_plot.crs}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # 1. First, plot the elevation map as background
        # Sample elevation data for visualization
        sample_rows = min(2000, dem_analyzer.elevation_data.shape[0])
        sample_cols = min(2000, dem_analyzer.elevation_data.shape[1])
        row_step = max(1, dem_analyzer.elevation_data.shape[0] // sample_rows)
        col_step = max(1, dem_analyzer.elevation_data.shape[1] // sample_cols)
        
        elevation_sample = dem_analyzer.elevation_data[::row_step, ::col_step]
        
        # Handle no-data values for plotting
        if dem_analyzer.src_info['nodata'] is not None:
            elevation_sample = np.where(elevation_sample == dem_analyzer.src_info['nodata'], 
                                      np.nan, elevation_sample)
        
        # Get extent for the elevation map (in DEM CRS coordinates)
        bounds = dem_analyzer.src_info['bounds']
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        print(f"DEM bounds: {bounds}")
        print(f"Roads bounds: {roads_for_plot.total_bounds}")
        
        # Plot elevation as background
        im = ax.imshow(elevation_sample, 
                      cmap='terrain', 
                      aspect='equal',
                      extent=extent,
                      alpha=0.7,
                      origin='upper')
        
        # Add elevation colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Elevation (feet)', fontsize=12)
        
        # 2. Overlay road network with flood risk colors
        # Color mapping for risk levels
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Unknown': 'gray'}
        linewidth_map = {'High': 2.5, 'Medium': 2.0, 'Low': 1.5, 'Unknown': 1.0}
        
        # Plot roads by risk level (plot low risk first, high risk last for visibility)
        for risk_level in ['Unknown', 'Low', 'Medium', 'High']:
            subset = roads_for_plot[roads_for_plot[risk_column] == risk_level]
            if len(subset) > 0:
                subset.plot(ax=ax, 
                           color=color_map[risk_level], 
                           linewidth=linewidth_map[risk_level], 
                           label=f'{risk_level} Risk Roads',
                           alpha=0.9,
                           zorder=10)  # Ensure roads are on top
        
        # Customize the plot
        ax.set_title(f'Elevation Map with Road Flood Risk - {flood_level}ft Water Level', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add legend for roads
        road_legend = ax.legend(loc='upper right', fontsize=10, 
                               title='Road Flood Risk', title_fontsize=11)
        road_legend.set_zorder(20)
        
        # Add grid
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Add text box with summary statistics
        high_risk_count = (roads_for_plot[risk_column] == 'High').sum()
        medium_risk_count = (roads_for_plot[risk_column] == 'Medium').sum()
        total_roads = len(roads_for_plot)
        
        stats_text = f'''Flood Risk Summary ({flood_level}ft):
High Risk: {high_risk_count:,} segments ({high_risk_count/total_roads*100:.1f}%)
Medium Risk: {medium_risk_count:,} segments ({medium_risk_count/total_roads*100:.1f}%)
Total Roads: {total_roads:,} segments'''
        
        # Add text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, zorder=15)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# Main execution
if __name__ == "__main__":
    """
    Main execution block for command line usage.
    
    Handles command line arguments, file validation, and analysis type selection.
    
    Command line usage:
        python elevation_data_analyzer.py [dem_file_path] [--quick]
        
    Arguments:
        dem_file_path: Path to DEM file (optional, uses default if not provided)
        --quick: Use quick summary instead of detailed analysis (optional)
        
    Default behavior is detailed analysis.
    """
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze elevation data from DEM files with optional road network flood risk assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python elevation_data_analyzer.py                           # Use default DEM file, detailed analysis
  python elevation_data_analyzer.py my_dem.tif                # Use custom DEM file, detailed analysis  
  python elevation_data_analyzer.py --quick                   # Use default DEM file, quick summary
  python elevation_data_analyzer.py my_dem.tif --quick        # Use custom DEM file, quick summary
  python elevation_data_analyzer.py --roads roads.shp         # DEM analysis + road flood risk
  python elevation_data_analyzer.py my_dem.tif --roads roads.shp --quick  # All options combined
        """
    )
    
    parser.add_argument(
        'dem_file', 
        nargs='?', 
        default="nyc_data/DEM_LiDAR_1ft_2010_Improved_NYC.img",
        help='Path to the DEM file (default: nyc_data/DEM_LiDAR_1ft_2010_Improved_NYC.img)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Perform quick summary analysis instead of detailed analysis'
    )
    
    parser.add_argument(
        '--roads',
        type=str,
        help='Path to road network shapefile for flood risk assessment'
    )
    
    # Parse arguments
    args = parser.parse_args()
    dem_file = args.dem_file
    
    # Display what we're using
    if args.dem_file == "nyc_data/DEM_LiDAR_1ft_2010_Improved_NYC.img":
        print(f"Using default DEM file: {dem_file}")
    else:
        print(f"Using DEM file: {dem_file}")
    
    analysis_type = "quick summary" if args.quick else "detailed analysis"
    print(f"Analysis type: {analysis_type}")
    
    if args.roads:
        print(f"Road network: {args.roads}")
    
    # Check if DEM file exists
    if not Path(dem_file).exists():
        print(f"❌ Error: DEM file not found: {dem_file}")
        print("Please provide a valid DEM file path")
        sys.exit(1)
    
    # Check if road file exists (if provided)
    if args.roads and not Path(args.roads).exists():
        print(f"❌ Error: Road network file not found: {args.roads}")
        print("Please provide a valid road network shapefile path")
        sys.exit(1)
    
    # Run DEM analysis
    print("\n" + "="*60)
    print("ELEVATION DATA ANALYSIS")
    print("="*60)
    
    if args.quick:
        quick_elevation_summary(dem_file)
        dem_analyzer = None
    else:
        dem_analyzer = ElevationDataAnalyzer(dem_file)
        dem_analyzer.run_complete_analysis()
    
    # Run road network analysis if requested
    if args.roads:
        print("\n" + "="*60)
        print("ROAD NETWORK FLOOD RISK ANALYSIS")
        print("="*60)
        
        # Load DEM analyzer if not already loaded (quick mode)
        if dem_analyzer is None:
            print("Loading DEM data for road analysis...")
            dem_analyzer = ElevationDataAnalyzer(dem_file)
        
        # Load and analyze road network
        road_analyzer = RoadNetworkAnalyzer(args.roads)
        road_analyzer.basic_road_info()
        
        # Extract elevations for roads
        roads_with_elevations = road_analyzer.extract_road_elevations(dem_analyzer)
        
        if roads_with_elevations is not None:
            # Assess flood risk
            roads_with_risk = road_analyzer.assess_flood_risk(roads_with_elevations)
            
            if roads_with_risk is not None:
                # Create combined visualization (elevation + roads)
                road_analyzer.visualize_combined_elevation_and_roads(
                    roads_with_risk, dem_analyzer, flood_level=10)
                
                # Also create separate road-only visualization
                # road_analyzer.visualize_road_flood_risk(roads_with_risk, flood_level=10)
                
                print(f"\n✅ ROAD ANALYSIS COMPLETE!")
                print(f"📁 Road file: {args.roads}")
                print(f"🛣️  {len(roads_with_risk):,} road segments analyzed")
                
                # Show summary of high-risk segments
                high_risk_10ft = roads_with_risk['flood_risk_10ft'] == 'High'
                high_risk_count = high_risk_10ft.sum()
                print(f"⚠️  {high_risk_count:,} segments at high risk (10ft flood level)")
        
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"DEM file: {dem_file}")
    if args.roads:
        print(f"Road network: {args.roads}")
    print(f"Analysis type: {analysis_type}")
    print("Analysis complete! ✅")
