#!/usr/bin/env python3
"""
Lower Manhattan DEM Clipping Script

This script clips the NYC DEM to focus on Lower Manhattan area using the
memory-efficient raster clipping methods. Lower Manhattan is defined as
the area south of Canal Street (approximately south of 40.72°N).

Usage:
    python clip_lower_manhattan.py [input_dem_path] [output_path]
    
If no arguments provided, uses default paths.
"""

import sys
from pathlib import Path
from raster_clipping_methods import MemoryEfficientRasterClipper, quick_clip

def get_lower_manhattan_bounds():
    """
    Get Lower Manhattan bounding box coordinates in different coordinate systems.
    
    Lower Manhattan is roughly defined as:
    - North: Canal Street (approximately 40.72°N)
    - South: Battery Park (approximately 40.70°N) 
    - East: East River
    - West: Hudson River
    
    Returns:
        dict: Bounds in different coordinate reference systems
    """
    bounds_options = {
        # WGS84 (Lat/Lon) - EPSG:4326
        'epsg_4326': (-74.02, 40.70, -73.97, 40.72),
        
        # NY State Plane Long Island (feet) - EPSG:2263  
        'epsg_2263': (977000, 193000, 986000, 201000),
        
        # UTM Zone 18N (meters) - EPSG:32618
        'epsg_32618': (580000, 4505000, 586000, 4509000)
    }
    
    return bounds_options

def clip_to_lower_manhattan(input_dem_path, output_path=None, method='gdal'):
    """
    Clip NYC DEM to Lower Manhattan bounds.
    
    Args:
        input_dem_path (str): Path to input NYC DEM file
        output_path (str): Path for output clipped DEM (optional)
        method (str): Clipping method ('gdal', 'window', or 'chunk')
        
    Returns:
        str: Path to clipped output file
    """
    print("🏙️ LOWER MANHATTAN DEM CLIPPING")
    print("=" * 45)
    
    # Validate input file
    input_path = Path(input_dem_path)
    if not input_path.exists():
        print(f"❌ Error: Input DEM file not found: {input_dem_path}")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_lower_manhattan{input_path.suffix}"
    
    print(f"Input DEM: {input_dem_path}")
    print(f"Output file: {output_path}")
    print(f"Clipping method: {method}")
    
    # Initialize clipper
    clipper = MemoryEfficientRasterClipper(input_dem_path)
    
    # Get Lower Manhattan bounds
    bounds_options = get_lower_manhattan_bounds()
    
    # Determine which coordinate system to use based on the DEM's CRS
    dem_crs_code = None
    try:
        if clipper.crs:
            dem_crs_code = clipper.crs.to_epsg()
    except:
        pass
    
    # Select appropriate bounds
    if dem_crs_code == 4326:  # WGS84
        bounds = bounds_options['epsg_4326']
        print(f"Using WGS84 bounds: {bounds}")
    elif dem_crs_code == 2263:  # NY State Plane
        bounds = bounds_options['epsg_2263'] 
        print(f"Using NY State Plane bounds: {bounds}")
    elif dem_crs_code in [32618, 32617]:  # UTM
        bounds = bounds_options['epsg_32618']
        print(f"Using UTM bounds: {bounds}")
    else:
        # Default to NY State Plane (most common for NYC data)
        bounds = bounds_options['epsg_2263']
        print(f"Unknown CRS, defaulting to NY State Plane bounds: {bounds}")
        print(f"⚠️ Warning: CRS mismatch possible - check results carefully")
    
    # Perform clipping based on selected method
    print(f"\n🔪 Starting clipping operation...")
    
    if method == 'gdal':
        result = clipper.method2_gdal_clip(bounds, output_path)
    elif method == 'window':
        result = clipper.method1_window_clipping(bounds, output_path)
    elif method == 'chunk':
        result = clipper.method3_chunk_processing(bounds, output_path)
    else:
        print(f"❌ Unknown method: {method}. Using 'gdal' as fallback.")
        result = clipper.method2_gdal_clip(bounds, output_path)
    
    if result:
        print(f"\n✅ SUCCESS!")
        print(f"Lower Manhattan DEM clipped successfully")
        print(f"Output saved to: {result}")
        
        # Display file size comparison
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        output_size_mb = Path(result).stat().st_size / (1024 * 1024)
        reduction_percent = (1 - output_size_mb / input_size_mb) * 100
        
        print(f"\n📊 FILE SIZE COMPARISON")
        print("-" * 25)
        print(f"Original: {input_size_mb:.1f} MB")
        print(f"Clipped:  {output_size_mb:.1f} MB")
        print(f"Reduction: {reduction_percent:.1f}%")
        
        return result
    else:
        print(f"\n❌ CLIPPING FAILED!")
        return None

def main():
    """Main function with command line argument handling"""
    print("Lower Manhattan DEM Clipping Tool")
    print("-" * 40)
    
    # Default paths - update these for your setup
    default_input = "nyc_data/DEM_LiDAR_1ft_2010_Improved_NYC.img"
    default_output = None  # Will be auto-generated
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        input_dem = sys.argv[1]
        output_dem = sys.argv[2] if len(sys.argv) >= 3 else default_output
    else:
        input_dem = default_input
        output_dem = default_output
        print(f"Using default input: {input_dem}")
        print("To specify custom paths: python clip_lower_manhattan.py <input_dem> [output_dem]")
    
    # Check if input file exists
    if not Path(input_dem).exists():
        print(f"\n❌ Input file not found: {input_dem}")
        print("\nAvailable options:")
        print("1. Update the default_input path in the script")
        print("2. Run with command line arguments:")
        print("   python clip_lower_manhattan.py path/to/your/dem.img")
        return
    
    # Perform clipping
    result = clip_to_lower_manhattan(
        input_dem_path=input_dem,
        output_path=output_dem,
        method='gdal'  # Change to 'window' or 'chunk' if needed
    )
    
    if result:
        print(f"\n🎉 CLIPPING COMPLETE!")
        print(f"Your Lower Manhattan DEM is ready: {result}")
        print(f"\nNext steps:")
        print(f"1. Load the clipped DEM in your GIS software")
        print(f"2. Run elevation analysis:")
        print(f"   python elevation_data_analyzer.py {result}")
        print(f"3. Combine with road data for flood risk analysis")
    else:
        print(f"\n💡 Troubleshooting tips:")
        print(f"1. Verify the input DEM file path is correct")
        print(f"2. Ensure you have write permissions in the output directory")
        print(f"3. Try a different clipping method (window, chunk)")
        print(f"4. Check if GDAL is properly installed")

if __name__ == "__main__":
    main()