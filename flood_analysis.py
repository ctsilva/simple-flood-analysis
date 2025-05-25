#!/usr/bin/env python3
"""
NYC Flood Risk Analysis Script
This script analyzes flood risk for NYC road segments based on elevation data,
water height, and watershed analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import grass.script as gscript
from grass.pygrass.modules import Module
from grass.pygrass.raster import RasterRow
from grass.pygrass.vector import VectorTopo
import warnings
warnings.filterwarnings('ignore')

class NYCFloodAnalysis:
    def __init__(self, grass_database_path, location_name="nyc_flood"):
        """
        Initialize GRASS GIS environment and analysis parameters
        
        Args:
            grass_database_path (str): Path to GRASS database
            location_name (str): Name for GRASS location
        """
        self.grass_db = grass_database_path
        self.location = location_name
        self.mapset = "PERMANENT"
        
        # Ensure GRASS database directory exists
        os.makedirs(grass_database_path, exist_ok=True)
        
    def setup_grass_environment(self, dem_path):
        """
        Setup GRASS GIS environment and import DEM data
        
        Args:
            dem_path (str): Path to NYC DEM raster file
        """
        print("Setting up GRASS GIS environment...")
        
        # Initialize GRASS session
        import grass.script.setup as gsetup
        gsetup.init(os.environ['GISBASE'], self.grass_db, self.location, self.mapset)
        
        # Set region from DEM
        gscript.run_command('g.region', raster='elevation', flags='p')
        
    def import_elevation_data(self, dem_path):
        """
        Import NYC elevation raster data into GRASS
        
        Args:
            dem_path (str): Path to DEM file (GeoTIFF)
        """
        print("Importing elevation data...")
        
        # Import DEM into GRASS
        gscript.run_command('r.in.gdal', 
                          input=dem_path, 
                          output='elevation',
                          overwrite=True)
        
        # Set computational region to match DEM
        gscript.run_command('g.region', raster='elevation')
        
        print("Elevation data imported successfully")
    
    def compute_flood_zones(self, water_height):
        """
        Compute flood zones based on water height
        
        Args:
            water_height (float): Water height in meters above sea level
        """
        print(f"Computing flood zones for water height: {water_height}m...")
        
        # Create flood zone map (areas below water height)
        flood_expression = f"flood_zone = if(elevation <= {water_height}, 1, 0)"
        gscript.run_command('r.mapcalc', expression=flood_expression, overwrite=True)
        
        # Create flood depth map
        depth_expression = f"flood_depth = if(elevation <= {water_height}, {water_height} - elevation, 0)"
        gscript.run_command('r.mapcalc', expression=depth_expression, overwrite=True)
        
        print("Flood zones computed")
    
    def import_road_network(self, roads_path):
        """
        Import NYC road network data
        
        Args:
            roads_path (str): Path to road network shapefile
        """
        print("Importing road network...")
        
        # Import roads into GRASS
        gscript.run_command('v.in.ogr',
                          input=roads_path,
                          output='roads',
                          overwrite=True)
        
        print("Road network imported")
    
    def perform_watershed_analysis(self):
        """
        Perform watershed analysis to understand water flow patterns
        """
        print("Performing watershed analysis...")
        
        # Fill sinks in DEM
        gscript.run_command('r.fill.dir',
                          input='elevation',
                          output='elevation_filled',
                          direction='flow_direction',
                          overwrite=True)
        
        # Calculate flow accumulation
        gscript.run_command('r.watershed',
                          elevation='elevation_filled',
                          accumulation='flow_accumulation',
                          drainage='drainage_direction',
                          basin='watersheds',
                          threshold=1000,  # Minimum size for watershed delineation
                          overwrite=True)
        
        # Calculate slope
        gscript.run_command('r.slope.aspect',
                          elevation='elevation_filled',
                          slope='slope',
                          overwrite=True)
        
        print("Watershed analysis completed")
    
    def calculate_flood_probability(self, distance_threshold=100):
        """
        Calculate flood probability for road segments based on multiple factors
        
        Args:
            distance_threshold (float): Distance threshold in meters for proximity analysis
        """
        print("Calculating flood probability for road segments...")
        
        # Convert flood zones to vector for distance calculations
        gscript.run_command('r.to.vect',
                          input='flood_zone',
                          output='flood_zones_vector',
                          type='area',
                          overwrite=True)
        
        # Calculate distance from roads to flood zones
        gscript.run_command('v.distance',
                          from_='roads',
                          to='flood_zones_vector',
                          upload='dist',
                          column='flood_dist',
                          overwrite=True)
        
        # Sample elevation, slope, and flow accumulation at road locations
        gscript.run_command('v.what.rast',
                          map='roads',
                          raster='elevation_filled',
                          column='elevation')
        
        gscript.run_command('v.what.rast',
                          map='roads',
                          raster='slope',
                          column='slope')
        
        gscript.run_command('v.what.rast',
                          map='roads',
                          raster='flow_accumulation',
                          column='flow_accum')
        
        # Sample flood depth at road locations
        gscript.run_command('v.what.rast',
                          map='roads',
                          raster='flood_depth',
                          column='flood_depth')
        
        print("Flood probability calculation completed")
    
    def compute_risk_scores(self):
        """
        Compute composite flood risk scores for road segments
        """
        print("Computing flood risk scores...")
        
        # Add risk score column
        gscript.run_command('v.db.addcolumn',
                          map='roads',
                          columns='risk_score DOUBLE')
        
        # Calculate risk score using SQL
        # This is a simplified risk model - you can make it more sophisticated
        risk_formula = """
        UPDATE roads SET risk_score = 
        CASE 
            WHEN flood_depth > 0 THEN 1.0
            WHEN flood_dist < 50 THEN 0.8
            WHEN flood_dist < 100 THEN 0.6
            WHEN flood_dist < 200 AND slope < 2 THEN 0.4
            WHEN flow_accum > 1000 AND slope < 5 THEN 0.3
            ELSE 0.1
        END
        """
        
        gscript.run_command('db.execute', sql=risk_formula)
        
        print("Risk scores computed")
    
    def export_results(self, output_dir):
        """
        Export analysis results
        
        Args:
            output_dir (str): Directory to save output files
        """
        print("Exporting results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export roads with risk scores
        gscript.run_command('v.out.ogr',
                          input='roads',
                          output=os.path.join(output_dir, 'roads_flood_risk.shp'),
                          overwrite=True)
        
        # Export flood zones
        gscript.run_command('r.out.gdal',
                          input='flood_zone',
                          output=os.path.join(output_dir, 'flood_zones.tif'),
                          overwrite=True)
        
        # Export flood depth
        gscript.run_command('r.out.gdal',
                          input='flood_depth',
                          output=os.path.join(output_dir, 'flood_depth.tif'),
                          overwrite=True)
        
        # Export watersheds
        gscript.run_command('r.out.gdal',
                          input='watersheds',
                          output=os.path.join(output_dir, 'watersheds.tif'),
                          overwrite=True)
        
        print(f"Results exported to {output_dir}")
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics of the analysis
        """
        print("Generating summary statistics...")
        
        # Get basic statistics
        stats = gscript.parse_command('v.db.univar', 
                                    map='roads', 
                                    column='risk_score')
        
        print("\n=== FLOOD RISK ANALYSIS SUMMARY ===")
        print(f"Risk Score Statistics:")
        print(f"- Mean: {stats.get('mean', 'N/A')}")
        print(f"- Standard Deviation: {stats.get('stddev', 'N/A')}")
        print(f"- Minimum: {stats.get('min', 'N/A')}")
        print(f"- Maximum: {stats.get('max', 'N/A')}")
        
        # Count high-risk road segments
        high_risk_count = gscript.read_command('v.db.select',
                                             map='roads',
                                             where='risk_score > 0.7',
                                             flags='c').strip()
        
        print(f"- High Risk Segments (>0.7): {high_risk_count}")
        
    def run_analysis(self, dem_path, roads_path, water_height, output_dir):
        """
        Run complete flood risk analysis
        
        Args:
            dem_path (str): Path to DEM file
            roads_path (str): Path to roads shapefile
            water_height (float): Water height for flood scenario
            output_dir (str): Output directory for results
        """
        try:
            # Setup and import data
            self.import_elevation_data(dem_path)
            self.import_road_network(roads_path)
            
            # Run analysis steps
            self.compute_flood_zones(water_height)
            self.perform_watershed_analysis()
            self.calculate_flood_probability()
            self.compute_risk_scores()
            
            # Generate outputs
            self.export_results(output_dir)
            self.generate_summary_statistics()
            
            print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """
    Main function to run the flood analysis
    """
    # Configuration
    config = {
        'grass_database': '/tmp/grass_flood_analysis',
        'dem_path': 'nyc_dem.tif',  # Replace with actual path
        'roads_path': 'nyc_roads.shp',  # Replace with actual path
        'water_height': 3.0,  # meters above sea level
        'output_dir': 'flood_analysis_results'
    }
    
    # Initialize analysis
    analysis = NYCFloodAnalysis(
        grass_database_path=config['grass_database'],
        location_name='nyc_flood_analysis'
    )
    
    # Run analysis
    analysis.run_analysis(
        dem_path=config['dem_path'],
        roads_path=config['roads_path'],
        water_height=config['water_height'],
        output_dir=config['output_dir']
    )
    
    print("\nAnalysis complete! Check the output directory for results.")
    print("Files generated:")
    print("- roads_flood_risk.shp: Roads with flood risk scores")
    print("- flood_zones.tif: Binary flood zone map")
    print("- flood_depth.tif: Flood depth raster")
    print("- watersheds.tif: Watershed boundaries")

if __name__ == "__main__":
    main()
