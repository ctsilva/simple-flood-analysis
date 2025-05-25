#!/usr/bin/env python3
"""
NYC Flood Risk Analysis Script
This script analyzes flood risk for NYC road segments based on elevation data,
water height, and watershed analysis using pure Python libraries.
"""

from __future__ import annotations
from typing import Optional, Union, List, Dict, Any, Tuple

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial.distance import cdist
from shapely.geometry import Point, LineString, Polygon, box
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PythonFloodAnalysis:
    """
    Pure Python implementation for flood risk analysis without GRASS GIS dependency.
    
    This class provides comprehensive flood risk assessment using rasterio, numpy,
    scipy, and geopandas for all geospatial operations.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "flood_analysis_results") -> None:
        """
        Initialize flood analysis with output directory.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.dem_data: Optional[np.ndarray] = None
        self.dem_transform: Optional[rasterio.transform.Affine] = None
        self.dem_crs: Optional[rasterio.crs.CRS] = None
        self.dem_profile: Optional[Dict[str, Any]] = None
        self.roads_gdf: Optional[gpd.GeoDataFrame] = None
        
        # Analysis results
        self.flood_zones: Optional[np.ndarray] = None
        self.flood_depths: Optional[np.ndarray] = None
        self.flow_accumulation: Optional[np.ndarray] = None
        self.slope: Optional[np.ndarray] = None
        
    def load_elevation_data(self, dem_path: Union[str, Path]) -> None:
        """
        Load elevation data from DEM file.
        
        Args:
            dem_path: Path to DEM raster file
        """
        print(f"Loading elevation data from: {dem_path}")
        
        with rasterio.open(dem_path) as src:
            self.dem_data = src.read(1)
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            self.dem_profile = src.profile.copy()
            
        print(f"✅ DEM loaded: {self.dem_data.shape} pixels")
        print(f"   CRS: {self.dem_crs}")
        print(f"   Bounds: {rasterio.transform.array_bounds(*self.dem_data.shape, self.dem_transform)}")
    
    def load_road_network(self, roads_path: Union[str, Path]) -> None:
        """
        Load road network data from shapefile.
        
        Args:
            roads_path: Path to road network shapefile
        """
        print(f"Loading road network from: {roads_path}")
        
        self.roads_gdf = gpd.read_file(roads_path)
        
        # Reproject roads to match DEM CRS if needed
        if self.roads_gdf.crs != self.dem_crs:
            print(f"Reprojecting roads from {self.roads_gdf.crs} to {self.dem_crs}")
            self.roads_gdf = self.roads_gdf.to_crs(self.dem_crs)
        
        print(f"✅ Roads loaded: {len(self.roads_gdf)} segments")
    
    def compute_flood_zones(self, water_height: float) -> None:
        """
        Compute flood zones based on water height threshold.
        
        Args:
            water_height: Water height in same units as DEM
        """
        print(f"Computing flood zones for water height: {water_height}")
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_elevation_data() first.")
        
        # Create flood zone mask (1 = flooded, 0 = not flooded)
        self.flood_zones = (self.dem_data <= water_height).astype(np.uint8)
        
        # Compute flood depths (positive where flooded)
        self.flood_depths = np.where(
            self.dem_data <= water_height,
            water_height - self.dem_data,
            0
        )
        
        flooded_pixels = np.sum(self.flood_zones)
        total_pixels = self.flood_zones.size
        flood_percentage = flooded_pixels / total_pixels * 100
        
        print(f"✅ Flood zones computed")
        print(f"   Flooded area: {flooded_pixels:,} pixels ({flood_percentage:.1f}%)")
        print(f"   Max flood depth: {np.max(self.flood_depths):.2f}")
    
    def compute_flow_accumulation(self) -> None:
        """
        Compute flow accumulation using simple D8 flow direction algorithm.
        This is a simplified version compared to GRASS r.watershed.
        """
        print("Computing flow accumulation...")
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_elevation_data() first.")
        
        # Fill sinks using scipy ndimage
        filled_dem = self._fill_sinks(self.dem_data)
        
        # Compute flow directions (D8 algorithm)
        flow_dirs = self._compute_flow_directions(filled_dem)
        
        # Compute flow accumulation
        self.flow_accumulation = self._compute_accumulation(flow_dirs)
        
        print(f"✅ Flow accumulation computed")
        print(f"   Max accumulation: {np.max(self.flow_accumulation):,}")
    
    def compute_slope(self) -> None:
        """
        Compute slope from DEM using gradient calculation.
        """
        print("Computing slope...")
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_elevation_data() first.")
        
        # Get pixel size from transform
        pixel_size_x = abs(self.dem_transform[0])
        pixel_size_y = abs(self.dem_transform[4])
        
        # Compute gradients
        grad_y, grad_x = np.gradient(self.dem_data, pixel_size_y, pixel_size_x)
        
        # Compute slope in degrees
        self.slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
        
        print(f"✅ Slope computed")
        print(f"   Max slope: {np.max(self.slope):.1f}°")
        print(f"   Mean slope: {np.mean(self.slope):.1f}°")
    
    def _fill_sinks(self, dem: np.ndarray) -> np.ndarray:
        """
        Simple sink filling using morphological reconstruction.
        """
        # Create a mask of the image border
        h, w = dem.shape
        marker = dem.copy()
        marker[1:-1, 1:-1] = np.inf
        
        # Morphological reconstruction
        filled = dem.copy()
        while True:
            marker_old = marker.copy()
            marker = np.minimum(marker, dem)
            marker = ndimage.maximum_filter(marker, size=3)
            marker[0, :] = dem[0, :]
            marker[-1, :] = dem[-1, :]
            marker[:, 0] = dem[:, 0]
            marker[:, -1] = dem[:, -1]
            
            if np.array_equal(marker, marker_old):
                break
                
        return marker
    
    def _compute_flow_directions(self, dem: np.ndarray) -> np.ndarray:
        """
        Compute D8 flow directions.
        Returns array with flow direction codes (1-8).
        """
        h, w = dem.shape
        flow_dirs = np.zeros((h, w), dtype=np.uint8)
        
        # D8 direction codes and offsets
        directions = [
            (1, (0, 1)),   # East
            (2, (1, 1)),   # Southeast  
            (3, (1, 0)),   # South
            (4, (1, -1)),  # Southwest
            (5, (0, -1)),  # West
            (6, (-1, -1)), # Northwest
            (7, (-1, 0)),  # North
            (8, (-1, 1))   # Northeast
        ]
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center_elev = dem[i, j]
                max_slope = -1
                flow_dir = 0
                
                for code, (di, dj) in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_elev = dem[ni, nj]
                        slope = (center_elev - neighbor_elev) / np.sqrt(di*di + dj*dj)
                        if slope > max_slope:
                            max_slope = slope
                            flow_dir = code
                
                flow_dirs[i, j] = flow_dir
        
        return flow_dirs
    
    def _compute_accumulation(self, flow_dirs: np.ndarray) -> np.ndarray:
        """
        Compute flow accumulation from flow directions.
        """
        h, w = flow_dirs.shape
        accumulation = np.ones((h, w), dtype=np.float32)
        
        # Direction offsets (matching D8 codes)
        offsets = {
            1: (0, 1),   2: (1, 1),   3: (1, 0),   4: (1, -1),
            5: (0, -1),  6: (-1, -1), 7: (-1, 0),  8: (-1, 1)
        }
        
        # Simple iterative accumulation (not the most efficient, but works)
        for _ in range(max(h, w)):  # Multiple passes to propagate flow
            new_accumulation = accumulation.copy()
            for i in range(h):
                for j in range(w):
                    flow_dir = flow_dirs[i, j]
                    if flow_dir in offsets:
                        di, dj = offsets[flow_dir]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            new_accumulation[ni, nj] += accumulation[i, j]
            accumulation = new_accumulation
        
        return accumulation
    
    def extract_road_elevations(self) -> None:
        """
        Extract elevation, slope, and flow accumulation values at road locations.
        """
        print("Extracting values at road locations...")
        
        if self.roads_gdf is None or self.dem_data is None:
            raise ValueError("Roads and DEM data must be loaded first.")
        
        # Initialize columns
        self.roads_gdf['elevation'] = np.nan
        self.roads_gdf['slope'] = np.nan
        self.roads_gdf['flow_accum'] = np.nan
        self.roads_gdf['flood_depth'] = 0.0
        self.roads_gdf['min_distance_to_flood'] = np.inf
        
        for idx, road in self.roads_gdf.iterrows():
            # Sample points along the road geometry
            if hasattr(road.geometry, 'interpolate'):
                # For LineString geometries
                length = road.geometry.length
                if length > 0:
                    # Sample at regular intervals
                    num_samples = max(2, int(length / 10))  # Sample every ~10 units
                    distances = np.linspace(0, length, num_samples)
                    sample_points = [road.geometry.interpolate(d) for d in distances]
                else:
                    sample_points = [road.geometry.centroid]
            else:
                # For Point geometries
                sample_points = [road.geometry]
            
            # Extract values at sample points
            elevations, slopes, flow_accums, flood_depths = [], [], [], []
            
            for point in sample_points:
                # Convert point to raster coordinates
                col, row = ~self.dem_transform * (point.x, point.y)
                col, row = int(col), int(row)
                
                # Check if point is within raster bounds
                if (0 <= row < self.dem_data.shape[0] and 
                    0 <= col < self.dem_data.shape[1]):
                    
                    elevations.append(self.dem_data[row, col])
                    
                    if self.slope is not None:
                        slopes.append(self.slope[row, col])
                    
                    if self.flow_accumulation is not None:
                        flow_accums.append(self.flow_accumulation[row, col])
                    
                    if self.flood_depths is not None:
                        flood_depths.append(self.flood_depths[row, col])
            
            # Store statistics for this road segment
            if elevations:
                self.roads_gdf.loc[idx, 'elevation'] = np.mean(elevations)
                if slopes:
                    self.roads_gdf.loc[idx, 'slope'] = np.mean(slopes)
                if flow_accums:
                    self.roads_gdf.loc[idx, 'flow_accum'] = np.mean(flow_accums)
                if flood_depths:
                    self.roads_gdf.loc[idx, 'flood_depth'] = np.max(flood_depths)
        
        print(f"✅ Values extracted for {len(self.roads_gdf)} road segments")
    
    def compute_flood_risk_scores(self) -> None:
        """
        Compute composite flood risk scores for road segments.
        """
        print("Computing flood risk scores...")
        
        if self.roads_gdf is None:
            raise ValueError("Roads data must be loaded and processed first.")
        
        # Initialize risk score
        self.roads_gdf['risk_score'] = 0.0
        
        # Risk scoring logic (simplified model)
        for idx, road in self.roads_gdf.iterrows():
            score = 0.0
            
            # Direct flooding (highest priority)
            if road['flood_depth'] > 0:
                score = min(1.0, 0.5 + road['flood_depth'] * 0.1)
            
            # Proximity to flood zones
            elif hasattr(road, 'min_distance_to_flood') and road['min_distance_to_flood'] < np.inf:
                if road['min_distance_to_flood'] < 50:
                    score = 0.8
                elif road['min_distance_to_flood'] < 100:
                    score = 0.6
                elif road['min_distance_to_flood'] < 200:
                    score = 0.4
            
            # Low elevation and slope factors
            if not np.isnan(road['elevation']) and road['elevation'] < 5:
                score = max(score, 0.3)
                
            if not np.isnan(road['slope']) and road['slope'] < 2:
                score = max(score, 0.2)
            
            # High flow accumulation
            if not np.isnan(road['flow_accum']) and road['flow_accum'] > 1000:
                score = max(score, 0.3)
            
            self.roads_gdf.loc[idx, 'risk_score'] = score
        
        print(f"✅ Risk scores computed")
        
        # Print summary statistics
        high_risk = (self.roads_gdf['risk_score'] > 0.7).sum()
        medium_risk = ((self.roads_gdf['risk_score'] > 0.4) & 
                      (self.roads_gdf['risk_score'] <= 0.7)).sum()
        low_risk = (self.roads_gdf['risk_score'] <= 0.4).sum()
        
        print(f"   High risk (>0.7): {high_risk} segments")
        print(f"   Medium risk (0.4-0.7): {medium_risk} segments") 
        print(f"   Low risk (≤0.4): {low_risk} segments")
    
    def export_results(self) -> None:
        """
        Export analysis results to files.
        """
        print("Exporting results...")
        
        # Export roads with risk scores
        if self.roads_gdf is not None:
            roads_output = self.output_dir / 'roads_flood_risk.shp'
            self.roads_gdf.to_file(roads_output)
            print(f"✅ Roads exported: {roads_output}")
        
        # Export raster results
        if self.flood_zones is not None:
            self._export_raster(self.flood_zones, 'flood_zones.tif', dtype='uint8')
        
        if self.flood_depths is not None:
            self._export_raster(self.flood_depths, 'flood_depths.tif')
        
        if self.flow_accumulation is not None:
            self._export_raster(self.flow_accumulation, 'flow_accumulation.tif')
        
        if self.slope is not None:
            self._export_raster(self.slope, 'slope.tif')
    
    def _export_raster(self, data: np.ndarray, filename: str, dtype: str = 'float32') -> None:
        """
        Export raster data to GeoTIFF file.
        """
        output_path = self.output_dir / filename
        
        profile = self.dem_profile.copy()
        profile.update({
            'dtype': dtype,
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(dtype), 1)
        
        print(f"✅ Raster exported: {output_path}")
    
    def generate_summary_report(self) -> None:
        """
        Generate a summary report of the flood risk analysis.
        """
        print("\n" + "="*50)
        print("FLOOD RISK ANALYSIS SUMMARY REPORT")
        print("="*50)
        
        if self.roads_gdf is not None:
            print(f"\n📊 ROAD NETWORK STATISTICS")
            print(f"Total road segments analyzed: {len(self.roads_gdf):,}")
            
            if 'risk_score' in self.roads_gdf.columns:
                risk_stats = self.roads_gdf['risk_score'].describe()
                print(f"\n🎯 RISK SCORE STATISTICS")
                print(f"Mean risk score: {risk_stats['mean']:.3f}")
                print(f"Std deviation: {risk_stats['std']:.3f}")
                print(f"Min risk score: {risk_stats['min']:.3f}")
                print(f"Max risk score: {risk_stats['max']:.3f}")
                
                # Risk categories
                high_risk = (self.roads_gdf['risk_score'] > 0.7).sum()
                medium_risk = ((self.roads_gdf['risk_score'] > 0.4) & 
                              (self.roads_gdf['risk_score'] <= 0.7)).sum()
                low_risk = (self.roads_gdf['risk_score'] <= 0.4).sum()
                
                print(f"\n⚠️  RISK CATEGORIES")
                print(f"High risk (>0.7): {high_risk:,} segments ({high_risk/len(self.roads_gdf)*100:.1f}%)")
                print(f"Medium risk (0.4-0.7): {medium_risk:,} segments ({medium_risk/len(self.roads_gdf)*100:.1f}%)")
                print(f"Low risk (≤0.4): {low_risk:,} segments ({low_risk/len(self.roads_gdf)*100:.1f}%)")
        
        if self.flood_zones is not None:
            flooded_pixels = np.sum(self.flood_zones)
            total_pixels = self.flood_zones.size
            print(f"\n🌊 FLOOD ZONE STATISTICS")
            print(f"Flooded pixels: {flooded_pixels:,}")
            print(f"Total pixels: {total_pixels:,}")
            print(f"Flooded area: {flooded_pixels/total_pixels*100:.1f}%")
            
            if self.flood_depths is not None:
                max_depth = np.max(self.flood_depths)
                mean_depth = np.mean(self.flood_depths[self.flood_depths > 0])
                print(f"Max flood depth: {max_depth:.2f}")
                print(f"Mean flood depth (flooded areas): {mean_depth:.2f}")
        
        print(f"\n📁 OUTPUT FILES")
        print(f"Results saved to: {self.output_dir}")
        for file in self.output_dir.glob('*'):
            print(f"  - {file.name}")
    
    def run_complete_analysis(self, dem_path: Union[str, Path], 
                            roads_path: Union[str, Path], 
                            water_height: float) -> None:
        """
        Run the complete flood risk analysis workflow.
        
        Args:
            dem_path: Path to DEM raster file
            roads_path: Path to roads shapefile
            water_height: Water height threshold for flooding
        """
        print("🌊 STARTING COMPLETE FLOOD RISK ANALYSIS")
        print("="*50)
        
        try:
            # Load data
            self.load_elevation_data(dem_path)
            self.load_road_network(roads_path)
            
            # Perform analyses
            self.compute_flood_zones(water_height)
            self.compute_slope()
            self.compute_flow_accumulation()
            
            # Extract values for roads
            self.extract_road_elevations()
            
            # Compute risk scores
            self.compute_flood_risk_scores()
            
            # Export results
            self.export_results()
            
            # Generate summary
            self.generate_summary_report()
            
            print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            raise


def main():
    """
    Main function to run the flood analysis with example configuration.
    """
    # Example configuration - update paths as needed
    config = {
        'dem_path': 'nyc_data/DEM_LiDAR_1ft_2010_Improved_NYC.img',
        'roads_path': 'nyc_data/geo_export_streets.shp', 
        'water_height': 10.0,  # feet above sea level
        'output_dir': 'flood_analysis_results'
    }
    
    # Check if files exist
    dem_file = Path(config['dem_path'])
    roads_file = Path(config['roads_path'])
    
    if not dem_file.exists():
        print(f"❌ DEM file not found: {dem_file}")
        print("Please update the dem_path in the configuration")
        return
    
    if not roads_file.exists():
        print(f"❌ Roads file not found: {roads_file}")
        print("Please update the roads_path in the configuration")
        return
    
    # Initialize and run analysis
    analysis = PythonFloodAnalysis(output_dir=config['output_dir'])
    
    analysis.run_complete_analysis(
        dem_path=config['dem_path'],
        roads_path=config['roads_path'],
        water_height=config['water_height']
    )


if __name__ == "__main__":
    main()