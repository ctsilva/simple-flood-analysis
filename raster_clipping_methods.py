#!/usr/bin/env python3
"""
Memory-Efficient Raster Clipping Methods
Clip large raster files without loading entire dataset into memory
"""

from __future__ import annotations
from typing import Tuple, Optional, Union, Any, Dict

import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.transform import Affine
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from pathlib import Path
import subprocess
import os

class MemoryEfficientRasterClipper:
    def __init__(self, raster_path: Union[str, Path]) -> None:
        """Initialize with raster file path.
        
        Args:
            raster_path: Path to the raster file
        """
        self.raster_path: Path = Path(raster_path)
        self.width: int = 0
        self.height: int = 0
        self.crs: Optional[CRS] = None
        self.transform: Optional[Affine] = None
        self.bounds: Optional[rasterio.coords.BoundingBox] = None
        self.nodata: Optional[float] = None
        self.dtype: str = ''
        self.get_raster_info()
    
    def get_raster_info(self) -> None:
        """Get basic raster information without loading data."""
        print(f"Analyzing raster: {self.raster_path.name}")
        print("=" * 50)
        
        with rasterio.open(self.raster_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds
            self.nodata = src.nodata
            self.dtype = src.dtypes[0]
            
            # Calculate file size info
            file_size_mb = self.raster_path.stat().st_size / (1024 * 1024)
            pixel_count = self.width * self.height
            
            print(f"Dimensions: {self.width:,} x {self.height:,} pixels")
            print(f"File size: {file_size_mb:.1f} MB")
            print(f"Total pixels: {pixel_count:,}")
            print(f"CRS: {self.crs}")
            print(f"Bounds: {self.bounds}")
    
    def method1_window_clipping(self, bounds: Tuple[float, float, float, float], 
                               output_path: Union[str, Path]) -> Optional[str]:
        """
        Method 1: Window-based clipping (Most Memory Efficient)
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in same CRS as raster
            output_path: Path for output file
            
        Returns:
            Path to output file if successful, None otherwise
        """
        print(f"\n🔪 METHOD 1: WINDOW CLIPPING")
        print("-" * 30)
        print(f"Target bounds: {bounds}")
        
        with rasterio.open(self.raster_path) as src:
            # Calculate window from bounds
            window = from_bounds(*bounds, src.transform)
            
            # Round window to integer pixels
            window = Window(
                col_off=int(window.col_off),
                row_off=int(window.row_off), 
                width=int(window.width),
                height=int(window.height)
            )
            
            print(f"Window: {window}")
            print(f"Output size: {window.width} x {window.height} pixels")
            
            # Calculate memory usage
            pixels = window.width * window.height
            if self.dtype == 'float32':
                memory_mb = pixels * 4 / (1024 * 1024)
            elif self.dtype == 'int16':
                memory_mb = pixels * 2 / (1024 * 1024)
            else:
                memory_mb = pixels * 8 / (1024 * 1024)  # Assume 8 bytes
            
            print(f"Estimated memory usage: {memory_mb:.1f} MB")
            
            if memory_mb > 500:
                print("⚠️ Warning: This will still use significant memory!")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    return None
            
            # Read only the window
            clipped_data = src.read(1, window=window)
            
            # Calculate transform for the clipped area
            clipped_transform = src.window_transform(window)
            
            # Write clipped raster
            profile = src.profile.copy()
            profile.update({
                'height': window.height,
                'width': window.width,
                'transform': clipped_transform
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(clipped_data, 1)
            
            print(f"✅ Clipped raster saved: {output_path}")
            print(f"   Size: {window.width} x {window.height}")
            
            return output_path
    
    def method2_gdal_clip(self, bounds: Tuple[float, float, float, float], 
                         output_path: Union[str, Path]) -> Optional[str]:
        """
        Method 2: GDAL command-line clipping (Most Memory Efficient)
        Uses GDAL translate command - no Python memory usage!
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in same CRS as raster
            output_path: Path for output file
            
        Returns:
            Path to output file if successful, None otherwise
        """
        print(f"\n🔪 METHOD 2: GDAL COMMAND-LINE CLIPPING")
        print("-" * 40)
        
        # Build GDAL translate command
        cmd = [
            'gdal_translate',
            '-projwin', str(bounds[0]), str(bounds[3]), str(bounds[2]), str(bounds[1]),
            '-of', 'GTiff',
            str(self.raster_path),
            str(output_path)
        ]
        
        print(f"GDAL command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ GDAL clipping successful!")
            print(f"   Output: {output_path}")
            
            # Check output file
            if Path(output_path).exists():
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                print(f"   Output size: {size_mb:.1f} MB")
                return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ GDAL error: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            print("\n💡 Make sure GDAL is installed:")
            print("   conda install gdal")
            print("   or: pip install gdal")
            
        except FileNotFoundError:
            print("❌ GDAL not found in system PATH")
            print("💡 Install GDAL:")
            print("   conda install gdal")
            print("   or: pip install gdal")
        
        return None
    
    def method3_chunk_processing(self, bounds: Tuple[float, float, float, float], 
                               output_path: Union[str, Path], chunk_size: int = 1000) -> Optional[str]:
        """
        Method 3: Process in chunks (For very large clips)
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in same CRS as raster
            output_path: Path for output file
            chunk_size: Size of chunks to process
            
        Returns:
            Path to output file if successful, None otherwise
        """
        print(f"\n🔪 METHOD 3: CHUNK PROCESSING")
        print("-" * 30)
        
        with rasterio.open(self.raster_path) as src:
            # Calculate window
            window = from_bounds(*bounds, src.transform)
            window = Window(
                col_off=int(window.col_off),
                row_off=int(window.row_off),
                width=int(window.width), 
                height=int(window.height)
            )
            
            print(f"Total window: {window.width} x {window.height}")
            print(f"Chunk size: {chunk_size} x {chunk_size}")
            
            # Prepare output file
            profile = src.profile.copy()
            profile.update({
                'height': window.height,
                'width': window.width,
                'transform': src.window_transform(window)
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Process in chunks
                for row_start in range(0, window.height, chunk_size):
                    for col_start in range(0, window.width, chunk_size):
                        
                        # Define chunk window
                        chunk_height = min(chunk_size, window.height - row_start)
                        chunk_width = min(chunk_size, window.width - col_start)
                        
                        chunk_window = Window(
                            col_off=window.col_off + col_start,
                            row_off=window.row_off + row_start,
                            width=chunk_width,
                            height=chunk_height
                        )
                        
                        # Read chunk
                        chunk_data = src.read(1, window=chunk_window)
                        
                        # Write chunk to output
                        dst.write(chunk_data, 1, 
                                window=Window(col_start, row_start, chunk_width, chunk_height))
                        
                        print(f"\rProcessed chunk {col_start//chunk_size + 1}, "
                              f"{row_start//chunk_size + 1}", end="", flush=True)
                
                print(f"\n✅ Chunk processing complete: {output_path}")
                return output_path
    
    def method4_polygon_mask(self, polygon_coords: list[Tuple[float, float]], 
                           output_path: Union[str, Path]) -> Optional[str]:
        """
        Method 4: Clip by polygon using rasterio mask (moderate memory)
        
        Args:
            polygon_coords: List of (x, y) coordinate tuples
            output_path: Path for output file
            
        Returns:
            Path to output file if successful, None otherwise
        """
        print(f"\n🔪 METHOD 4: POLYGON MASK CLIPPING")
        print("-" * 35)
        
        # Create polygon geometry
        polygon = Polygon(polygon_coords)
        print(f"Polygon bounds: {polygon.bounds}")
        
        with rasterio.open(self.raster_path) as src:
            # Use mask function - more memory efficient than loading full raster
            try:
                clipped_data, clipped_transform = mask(src, [polygon], crop=True, nodata=src.nodata)
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_data.shape[1],
                    'width': clipped_data.shape[2], 
                    'transform': clipped_transform
                })
                
                # Write output
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(clipped_data)
                
                print(f"✅ Polygon clipping successful: {output_path}")
                print(f"   Output size: {clipped_data.shape[2]} x {clipped_data.shape[1]}")
                return output_path
                
            except MemoryError:
                print("❌ Memory error - polygon clip area too large")
                print("💡 Try method 1 or 2 with smaller bounds")
                return None

def get_greenwich_village_bounds() -> Dict[str, Tuple[float, float, float, float]]:
    """Get Greenwich Village bounds in different coordinate systems.
    
    Returns:
        Dictionary mapping coordinate system names to bounds tuples
    """
    bounds_options = {
        'epsg_4326': (-74.008, 40.728, -73.995, 40.738),  # WGS84
        'epsg_2263': (983000, 199000, 990000, 206000),     # NY State Plane (feet)
        'epsg_32618': (583000, 4511000, 585000, 4513000)   # UTM Zone 18N
    }
    return bounds_options

def interactive_bounds_selector():
    """Interactive bounds selection"""
    print("\n🎯 INTERACTIVE BOUNDS SELECTION")
    print("-" * 35)
    
    print("Choose area to clip:")
    print("1. Greenwich Village")
    print("2. Manhattan (approx)")
    print("3. Custom bounds")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        bounds = get_greenwich_village_bounds()
        print("Greenwich Village bounds:")
        for crs, bbox in bounds.items():
            print(f"  {crs}: {bbox}")
        return bounds['epsg_2263']  # Default to NY State Plane
    
    elif choice == "2":
        # Manhattan approximate bounds in EPSG:2263
        return (980000, 190000, 1000000, 250000)
    
    elif choice == "3":
        print("Enter bounds (min_x, min_y, max_x, max_y):")
        try:
            min_x = float(input("Min X: "))
            min_y = float(input("Min Y: "))
            max_x = float(input("Max X: "))
            max_y = float(input("Max Y: "))
            return (min_x, min_y, max_x, max_y)
        except ValueError:
            print("Invalid input. Using Greenwich Village defaults.")
            return get_greenwich_village_bounds()['epsg_2263']
    
    else:
        return get_greenwich_village_bounds()['epsg_2263']

def quick_clip(raster_path, bounds=None, output_path=None, method='gdal'):
    """Quick clipping function"""
    
    if bounds is None:
        bounds = get_greenwich_village_bounds()['epsg_2263']
    
    if output_path is None:
        input_path = Path(raster_path)
        output_path = input_path.parent / f"{input_path.stem}_clipped{input_path.suffix}"
    
    clipper = MemoryEfficientRasterClipper(raster_path)
    
    if method == 'gdal':
        return clipper.method2_gdal_clip(bounds, output_path)
    elif method == 'window':
        return clipper.method1_window_clipping(bounds, output_path)
    elif method == 'chunk':
        return clipper.method3_chunk_processing(bounds, output_path)
    else:
        print(f"Unknown method: {method}")
        return None

def main():
    """Main function for interactive clipping"""
    import sys
    
    print("MEMORY-EFFICIENT RASTER CLIPPING TOOL")
    print("=" * 50)
    
    # Get input file from command line or prompt
    if len(sys.argv) > 1:
        raster_file = sys.argv[1]
        print(f"Using file from command line: {raster_file}")
    else:
        raster_file = input("Enter path to your DEM file: ").strip()
        if not raster_file:
            raster_file = "DEM_LiDAR_1ft_2010_Improved_NYC.img"
    
    if not Path(raster_file).exists():
        print(f"❌ File not found: {raster_file}")
        return
    
    # Initialize clipper
    clipper = MemoryEfficientRasterClipper(raster_file)
    
    # Get bounds
    bounds = interactive_bounds_selector()
    print(f"\nUsing bounds: {bounds}")
    
    # Choose method
    print("\nChoose clipping method:")
    print("1. GDAL command-line (fastest, least memory)")
    print("2. Window clipping (fast, low memory)")
    print("3. Chunk processing (slower, handles any size)")
    
    method_choice = input("Enter choice (1-3): ").strip()
    
    # Generate output filename
    input_path = Path(raster_file)
    output_file = input_path.parent / f"{input_path.stem}_clipped.tif"
    
    print(f"\nOutput will be saved as: {output_file}")
    
    # Execute clipping
    if method_choice == "1":
        result = clipper.method2_gdal_clip(bounds, output_file)
    elif method_choice == "2":
        result = clipper.method1_window_clipping(bounds, output_file)
    elif method_choice == "3":
        result = clipper.method3_chunk_processing(bounds, output_file)
    else:
        print("Invalid choice, defaulting to GDAL method")
        result = clipper.method2_gdal_clip(bounds, output_file)
    
    if result:
        print(f"\n🎉 SUCCESS!")
        print(f"Clipped raster saved: {result}")
        
        # Show info about clipped file
        try:
            with rasterio.open(result) as src:
                print(f"Clipped size: {src.width} x {src.height}")
                print(f"Clipped bounds: {src.bounds}")
        except:
            pass
    else:
        print(f"\n❌ Clipping failed")

if __name__ == "__main__":
    # Example usage
    print("Example usage:")
    print("1. Run main() for interactive clipping")
    print("2. Use quick_clip() for programmatic clipping")
    print()
    
    # Quick example
    # result = quick_clip("DEM_LiDAR_1ft_2010_Improved_NYC.img", method='gdal')
    
    main()
