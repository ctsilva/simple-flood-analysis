#!/usr/bin/env python3
"""
Geospatial Data Clipping Tool

A comprehensive tool for clipping vector data (shapefiles) to various geographic boundaries.
Supports multiple clipping methods including bounding boxes, coordinate ranges, 
name-based filtering, custom polygons, administrative boundaries, and DEM bounds.

Key Features:
- Multiple clipping methods for different use cases
- Automatic coordinate system handling and reprojection
- Command line interface for easy automation
- Visualization of clipping results
- Support for DEM bounds extraction

Dependencies:
- geopandas: Vector data processing
- pandas: Data manipulation
- shapely: Geometric operations
- matplotlib: Visualization
- rasterio: Raster data reading
- pathlib: File path handling

Usage:
    python nyc_data_clipping.py --dem dem_file.img --shapefile roads.shp
    python nyc_data_clipping.py --shapefile roads.shp  # Greenwich Village clip

Author: Geospatial Clipping Tool
Version: 2.0
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point, Polygon
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GeospatialDataClipper:
    """
    A comprehensive tool for clipping vector geospatial data to various boundaries.
    
    This class provides multiple methods to clip shapefiles including bounding box
    clipping, coordinate filtering, name-based filtering, polygon clipping,
    administrative boundary clipping, and DEM bounds clipping.
    
    Attributes:
        shapefile_path (Path): Path to the input shapefile
        full_data (GeoDataFrame): Complete loaded shapefile data
    """
    
    def __init__(self, shapefile_path):
        """Initialize the clipper with a shapefile.
        
        Args:
            shapefile_path (str): Path to the input shapefile
            
        Raises:
            FileNotFoundError: If the shapefile doesn't exist
            ValueError: If the shapefile cannot be read
        """
        self.shapefile_path = Path(shapefile_path)
        print(f"Loading shapefile data from: {self.shapefile_path.name}...")
        self.full_data = gpd.read_file(shapefile_path)
        print(f"Loaded {len(self.full_data):,} features")
        print(f"CRS: {self.full_data.crs}")
        print(f"Bounds: {self.full_data.total_bounds}")
        
    def clip_by_bounding_box(self, bounds):
        """Clip shapefile using rectangular bounding box coordinates.
        
        Creates a rectangular bounding box and clips all features that
        intersect with this box. This is the most efficient clipping method
        for rectangular areas.
        
        Args:
            bounds (tuple): Bounding box as (min_x, min_y, max_x, max_y)
            
        Returns:
            GeoDataFrame: Clipped shapefile data
        """
        print("\n🔲 BOUNDING BOX CLIPPING")
        print("-" * 30)
        
        # Create bounding box geometry
        bbox = box(*bounds)
        
        # Clip data
        clipped = self.full_data.clip(bbox)
        
        print(f"Original features: {len(self.full_data)}")
        print(f"Clipped features: {len(clipped)}")
        
        return clipped
    
    def clip_by_coordinates(self, lat_range, lon_range):
        """Clip shapefile using latitude and longitude coordinate ranges.
        
        Converts data to WGS84 (EPSG:4326) if needed and clips to the
        specified latitude and longitude ranges. Useful when working
        with GPS coordinates or web mapping applications.
        
        Args:
            lat_range (tuple): Latitude range as (min_lat, max_lat)
            lon_range (tuple): Longitude range as (min_lon, max_lon)
            
        Returns:
            GeoDataFrame: Clipped shapefile data in WGS84 coordinates
        """
        print("\n🌍 GEOGRAPHIC COORDINATE CLIPPING")
        print("-" * 35)
        
        # Ensure data is in WGS84 (lat/lon)
        if self.full_data.crs.to_epsg() != 4326:
            print("Converting to WGS84...")
            data_wgs84 = self.full_data.to_crs('EPSG:4326')
        else:
            data_wgs84 = self.full_data
        
        # Create bounding box
        min_lon, max_lon = lon_range
        min_lat, max_lat = lat_range
        bbox = box(min_lon, min_lat, max_lon, max_lat)
        
        # Clip
        clipped = data_wgs84.clip(bbox)
        
        print(f"Lat range: {lat_range}")
        print(f"Lon range: {lon_range}")
        print(f"Clipped features: {len(clipped)}")
        
        return clipped
    
    def clip_by_attribute_names(self, keywords):
        """Clip shapefile by filtering attribute names/values.
        
        Searches through text attributes (like street names, neighborhood names)
        for specified keywords and returns features that match any of the keywords.
        Useful for extracting features related to specific areas or categories.
        
        Args:
            keywords (list): List of strings to search for in attribute fields
            
        Returns:
            GeoDataFrame: Filtered shapefile data containing matching features
            None: If no name columns are found
        """
        print("\n🏷️ ATTRIBUTE NAME FILTERING")
        print("-" * 30)
        
        # Check available columns
        print(f"Available columns: {list(self.full_data.columns)}")
        
        # Look for street name columns (common names in NYC data)
        name_columns = []
        for col in self.full_data.columns:
            if any(term in col.upper() for term in ['NAME', 'STREET', 'ST_', 'FULL']):
                name_columns.append(col)
        
        print(f"Potential name columns: {name_columns}")
        
        if not name_columns:
            print("No street name columns found!")
            return None
        
        # Filter based on street names
        mask = pd.Series([False] * len(self.full_data))
        
        for keyword in keywords:
            for col in name_columns:
                if col in self.full_data.columns:
                    mask |= self.full_data[col].str.contains(keyword, case=False, na=False)
        
        filtered = self.full_data[mask]
        
        print(f"Keywords searched: {keywords}")
        print(f"Filtered features: {len(filtered)}")
        
        return filtered
    
    def clip_by_custom_polygon(self, polygon_coords):
        """Clip shapefile using a custom polygon boundary.
        
        Creates a polygon from the provided coordinates and clips all features
        that intersect with this polygon. Useful for irregular boundaries
        that cannot be represented by simple bounding boxes.
        
        Args:
            polygon_coords (list): List of (x, y) coordinate tuples defining 
                                 the polygon boundary
                                 
        Returns:
            GeoDataFrame: Clipped shapefile data
        """
        print("\n🔶 CUSTOM POLYGON CLIPPING")
        print("-" * 30)
        
        # Create polygon
        polygon = Polygon(polygon_coords)
        
        # Clip data
        clipped = self.full_data.clip(polygon)
        
        print(f"Polygon area: {polygon.area:.6f}")
        print(f"Clipped features: {len(clipped)}")
        
        return clipped
    
    def clip_by_administrative_boundary(self, boundary_file=None, boundary_name="Greenwich Village"):
        """Clip shapefile using administrative boundary data.
        
        Loads administrative boundary shapefiles (like neighborhood or district
        boundaries) and clips the input data to match the specified boundary.
        Useful for clipping to official administrative regions.
        
        Args:
            boundary_file (str): Path to boundary shapefile
            boundary_name (str): Name of the boundary to search for
            
        Returns:
            GeoDataFrame: Clipped shapefile data
            None: If boundary file not provided or boundary not found
        """
        print("\n🏛️ ADMINISTRATIVE BOUNDARY CLIPPING")
        print("-" * 40)
        
        if boundary_file:
            # Load boundary data
            boundaries = gpd.read_file(boundary_file)
            
            # Find specified boundary
            target_boundary = boundaries[
                boundaries['name'].str.contains(boundary_name, case=False, na=False)
            ]
            
            if len(target_boundary) > 0:
                clipped = self.full_data.clip(target_boundary.union_all())
                print(f"Clipped to '{boundary_name}' boundary: {len(clipped)} features")
                return clipped
            else:
                print(f"'{boundary_name}' boundary not found in boundary file")
        
        print("No boundary file provided or boundary not found")
        return None
    
    def visualize_clipping_result(self, original_data, clipped_data, title="Clipped Data"):
        """Create before/after visualization of clipping results.
        
        Generates a side-by-side comparison showing the original dataset
        and the clipped result. Samples large datasets for performance.
        
        Args:
            original_data (GeoDataFrame): Original unclipped data
            clipped_data (GeoDataFrame): Clipped result data
            title (str): Title for the clipped data plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original data (sample for performance)
        if len(original_data) > 10000:
            sample_original = original_data.sample(n=10000)
        else:
            sample_original = original_data
            
        # Plot original
        sample_original.plot(ax=ax1, linewidth=0.5, color='lightgray')
        ax1.set_title(f'Original Data (showing sample)\n{len(original_data)} total features')
        ax1.set_aspect('equal')
        
        # Plot clipped
        clipped_data.plot(ax=ax2, linewidth=1, color='red')
        ax2.set_title(f'{title}\n{len(clipped_data)} features')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def get_greenwich_village_bounds(self):
        """Get approximate Greenwich Village bounding box"""
        # Greenwich Village approximate bounds (in various coordinate systems)
        
        bounds_options = {
            # WGS84 (Lat/Lon) - EPSG:4326
            'wgs84': {
                'lat_range': (40.728, 40.738),  # ~40.733° N center
                'lon_range': (-74.008, -73.995)  # ~-74.002° W center
            },
            
            # NY State Plane Long Island (feet) - EPSG:2263
            'ny_state_plane': (
                983000, 199000,   # min_x, min_y
                990000, 206000    # max_x, max_y  
            ),
            
            # UTM Zone 18N (meters) - EPSG:32618  
            'utm18n': (
                583000, 4511000,  # min_x, min_y
                585000, 4513000   # max_x, max_y
            )
        }
        
        return bounds_options
    
    def clip_by_dem_bounds(self, dem_path):
        """Clip shapefile to match DEM (raster) file bounds.
        
        Reads the spatial extent of a Digital Elevation Model (DEM) or any
        raster file and clips the shapefile to exactly match those bounds.
        Automatically handles coordinate system reprojection if needed.
        Useful for ensuring vector and raster data have identical coverage.
        
        Args:
            dem_path (str): Path to DEM or other raster file
            
        Returns:
            GeoDataFrame: Clipped shapefile data in DEM coordinate system
            None: If clipping fails
        """
        print("\n🗻 DEM BOUNDS CLIPPING")
        print("-" * 25)
        
        try:
            # Read DEM metadata to get bounds and CRS
            with rasterio.open(dem_path) as dem:
                dem_bounds = dem.bounds
                dem_crs = dem.crs
                
            print(f"DEM file: {Path(dem_path).name}")
            print(f"DEM CRS: {dem_crs}")
            print(f"DEM bounds: {dem_bounds}")
            print(f"Shapefile CRS: {self.full_data.crs}")
            
            # Reproject shapefile to DEM CRS if needed
            if self.full_data.crs != dem_crs:
                print(f"Reprojecting shapefile from {self.full_data.crs} to {dem_crs}")
                shapefile_reproj = self.full_data.to_crs(dem_crs)
            else:
                shapefile_reproj = self.full_data
            
            # Create bounding box from DEM bounds
            bbox = box(dem_bounds.left, dem_bounds.bottom, 
                      dem_bounds.right, dem_bounds.top)
            
            # Clip shapefile to DEM bounds
            clipped = shapefile_reproj.clip(bbox)
            
            print(f"Original features: {len(self.full_data)}")
            print(f"Clipped features: {len(clipped)}")
            print(f"Clipped bounds: {clipped.total_bounds}")
            
            return clipped
            
        except Exception as e:
            print(f"Error clipping to DEM bounds: {e}")
            return None

def demonstrate_all_clipping_methods():
    """Demonstrate all available clipping methods with examples.
    
    Shows how to use each clipping method with sample data and parameters.
    Useful for learning the different approaches and their use cases.
    
    Returns:
        tuple: (bbox_clipped_data, name_filtered_data) for further analysis
    """
    # Initialize (replace with your actual shapefile path)
    shapefile_path = "nyc_centerline.shp"  # Replace with actual path
    
    try:
        clipper = GeospatialDataClipper(shapefile_path)
    except:
        print("Please update shapefile_path with your actual shapefile file")
        return
    
    # Get Greenwich Village bounds
    gv_bounds = clipper.get_greenwich_village_bounds()
    
    # Method 1: Bounding Box (using current CRS)
    print("Current CRS:", clipper.full_data.crs)
    if clipper.full_data.crs.to_epsg() == 2263:  # NY State Plane
        clipped1 = clipper.clip_by_bounding_box(gv_bounds['ny_state_plane'])
    elif clipper.full_data.crs.to_epsg() == 32618:  # UTM
        clipped1 = clipper.clip_by_bounding_box(gv_bounds['utm18n'])
    else:  # Default to WGS84 conversion
        clipped1 = clipper.clip_by_coordinates(
            gv_bounds['wgs84']['lat_range'],
            gv_bounds['wgs84']['lon_range']
        )
    
    # Method 3: Name-based filtering
    greenwich_keywords = ['GREENWICH', 'VILLAGE', 'WASHINGTON SQ', 'BLEECKER', 'HOUSTON']
    clipped3 = clipper.clip_by_attribute_names(greenwich_keywords)
    
    # Visualize results
    if clipped1 is not None:
        clipper.visualize_clipping_result(clipper.full_data, clipped1, "Greenwich Village - Bounding Box")
    
    if clipped3 is not None:
        clipper.visualize_clipping_result(clipper.full_data, clipped3, "Greenwich Village - Name Filter")
    
    return clipped1, clipped3

def quick_greenwich_village_clip(shapefile_path):
    """Quick function to clip any shapefile to Greenwich Village bounds.
    
    Provides a fast way to extract Greenwich Village area from any NYC dataset.
    Automatically handles coordinate system conversion and uses predefined bounds.
    
    Args:
        shapefile_path (str): Path to input shapefile
        
    Returns:
        GeoDataFrame: Clipped shapefile data for Greenwich Village area
    """
    print("🏙️ QUICK GREENWICH VILLAGE CLIPPING")
    print("-" * 40)
    
    # Load data
    streets = gpd.read_file(shapefile_path)
    print(f"Original data: {len(streets)} features, CRS: {streets.crs}")
    
    # Greenwich Village bounds (approximate)
    if streets.crs.to_epsg() == 4326:  # WGS84
        bbox = box(-74.008, 40.728, -73.995, 40.738)
    elif streets.crs.to_epsg() == 2263:  # NY State Plane (feet)
        bbox = box(983000, 199000, 990000, 206000)
    else:
        # Convert to WGS84 first
        streets = streets.to_crs('EPSG:4326')
        bbox = box(-74.008, 40.728, -73.995, 40.738)
    
    # Clip
    clipped = streets.clip(bbox)
    
    print(f"Clipped data: {len(clipped)} features")
    print(f"Bounds: {clipped.total_bounds}")
    
    return clipped

def save_clipped_data(clipped_data, output_path):
    """Save clipped GeoDataFrame to a new shapefile.
    
    Args:
        clipped_data (GeoDataFrame): Clipped geospatial data
        output_path (str): Path for the output shapefile
    """
    clipped_data.to_file(output_path)
    print(f"Saved clipped data to: {output_path}")

def clip_shapefile_to_dem_bounds(shapefile_path, dem_path, output_path=None):
    """
    Standalone function to clip a shapefile to DEM bounds
    
    Args:
        shapefile_path (str): Path to input shapefile
        dem_path (str): Path to DEM file  
        output_path (str): Path for output shapefile (optional)
    
    Returns:
        GeoDataFrame: Clipped shapefile data
    """
    print("CLIPPING SHAPEFILE TO DEM BOUNDS")
    print("=" * 40)
    
    # Initialize clipper
    clipper = GeospatialDataClipper(shapefile_path)
    
    # Clip to DEM bounds
    clipped_data = clipper.clip_by_dem_bounds(dem_path)
    
    if clipped_data is not None:
        # Generate output path if not provided
        if output_path is None:
            shapefile_stem = Path(shapefile_path).stem
            dem_stem = Path(dem_path).stem
            output_path = f"{shapefile_stem}_clipped_to_{dem_stem}.shp"
        
        # Save clipped data
        save_clipped_data(clipped_data, output_path)
        
        # Show summary
        print(f"\n📊 CLIPPING SUMMARY")
        print("-" * 25)
        print(f"Input shapefile: {len(clipper.full_data):,} features")
        print(f"Output shapefile: {len(clipped_data):,} features")
        print(f"Reduction: {(1 - len(clipped_data)/len(clipper.full_data))*100:.1f}%")
        
        # Visualize if reasonable size
        if len(clipped_data) < 50000:
            clipper.visualize_clipping_result(clipper.full_data, clipped_data, 
                                            f"Clipped to DEM Bounds")
        else:
            print("Skipping visualization (too many features)")
        
        return clipped_data
    else:
        print("❌ Clipping failed!")
        return None

# Main execution with command line support
if __name__ == "__main__":
    """
    Main execution block providing command line interface for shapefile clipping.
    
    Supports multiple clipping modes:
    - DEM bounds clipping (with --dem option)
    - Greenwich Village clipping (default)
    - Custom output paths (with --output option)
    
    Command line usage:
        python nyc_data_clipping.py [options]
        
    All operations include file validation, progress reporting, and optional visualization.
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clip vector shapefiles to DEM bounds or other geographic areas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nyc_data_clipping.py                                       # Default Greenwich Village clip
  python nyc_data_clipping.py --dem dem.img --shapefile roads.shp   # Clip to DEM bounds
  python nyc_data_clipping.py --shapefile roads.shp --dem dem.img --output clipped_roads.shp
        """
    )
    
    parser.add_argument(
        '--shapefile',
        default="nyc_data/geo_export_9f471af7-3ae8-4c15-9d31-3dd05a80e8f3.shp",
        help='Path to input shapefile (default: NYC centerline data)'
    )
    
    parser.add_argument(
        '--dem',
        help='Path to DEM file for bounds clipping'
    )
    
    parser.add_argument(
        '--output',
        help='Path for output shapefile (auto-generated if not provided)'
    )
    
    args = parser.parse_args()
    
    print("🗺️ GEOSPATIAL DATA CLIPPING TOOL")
    print("=" * 45)
    
    # Check if files exist
    if not Path(args.shapefile).exists():
        print(f"❌ Error: Shapefile not found: {args.shapefile}")
        sys.exit(1)
    
    if args.dem and not Path(args.dem).exists():
        print(f"❌ Error: DEM file not found: {args.dem}")
        sys.exit(1)
    
    try:
        if args.dem:
            # Clip shapefile to DEM bounds
            print(f"Clipping {args.shapefile} to bounds of {args.dem}")
            clipped_data = clip_shapefile_to_dem_bounds(
                args.shapefile, args.dem, args.output
            )
            
            if clipped_data is not None:
                print(f"\n✅ SUCCESS!")
                print(f"Clipped {len(clipped_data):,} features to DEM bounds")
            
        else:
            # Default: Quick clip to Greenwich Village
            print(f"Default mode: Clipping {args.shapefile} to Greenwich Village")
            
            gv_streets = quick_greenwich_village_clip(args.shapefile)
            
            # Save result
            output_path = args.output or "greenwich_village_streets.shp"
            save_clipped_data(gv_streets, output_path)
            
            # Show sample of data
            print("\nSample of clipped data:")
            available_cols = [col for col in ['FULL_STREE', 'BOROCODE', 'ST_NAME', 'NAME'] 
                            if col in gv_streets.columns]
            if available_cols:
                print(gv_streets[available_cols[:2]].head())
            else:
                print(gv_streets.head())
            
            print(f"\n✅ SUCCESS!")
            print(f"Clipped {len(gv_streets):,} features to Greenwich Village")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check file paths and try again")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nUsage examples:")
        print("python nyc_data_clipping.py --dem my_dem.img --shapefile roads.shp")
        print("python nyc_data_clipping.py --shapefile roads.shp  # Greenwich Village clip")
