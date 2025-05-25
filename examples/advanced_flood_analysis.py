#!/usr/bin/env python3
"""
Advanced Flood Analysis Example

This example demonstrates advanced flood risk analysis capabilities
including custom water levels, different analysis scenarios,
and integration with the pure Python flood analysis module.

Requirements:
- DEM file (Digital Elevation Model)
- Optional: Road network shapefile

Usage:
    python examples/advanced_flood_analysis.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flood_analysis import PythonFloodAnalysis
from elevation_data_analyzer import ElevationDataAnalyzer
from nyc_data_clipping import GeospatialDataClipper
from config import config
from logger import setup_logging, get_logger
from validation import FileValidationError, DataValidationError


def run_comprehensive_flood_analysis():
    """
    Run comprehensive flood analysis using the PythonFloodAnalysis class.
    
    This example demonstrates:
    1. Loading DEM and road data
    2. Computing flood zones at custom water levels
    3. Computing slope and flow accumulation
    4. Extracting road elevations
    5. Computing flood risk scores
    6. Exporting results
    """
    print("="*70)
    print("COMPREHENSIVE FLOOD ANALYSIS EXAMPLE")
    print("="*70)
    
    # Setup logging
    setup_logging(log_level='INFO')
    logger = get_logger("AdvancedExample")
    
    # Define file paths
    dem_path = config.get_default_dem_path()
    roads_path = config.get_default_roads_path()
    output_dir = Path("advanced_flood_results")
    
    # Check if required files exist
    if not dem_path.exists():
        print(f"❌ DEM file not found: {dem_path}")
        print("Please ensure the DEM file exists or run nyc_data_download_guide2.py")
        return False
    
    if not roads_path.exists():
        print(f"❌ Roads file not found: {roads_path}")
        print("Will run DEM-only analysis...")
        roads_path = None
    
    try:
        # Initialize flood analysis
        logger.info("Initializing comprehensive flood analysis")
        analysis = PythonFloodAnalysis(output_dir=output_dir)
        
        # Load elevation data
        print("🏔️ Loading elevation data...")
        analysis.load_elevation_data(dem_path)
        
        # Load road network if available
        if roads_path:
            print("🛣️ Loading road network...")
            analysis.load_road_network(roads_path)
        
        # Define custom water levels for analysis
        custom_water_levels = [3, 6, 10, 15, 25]  # feet above sea level
        
        for water_level in custom_water_levels:
            print(f"\n🌊 Analyzing flood scenario: {water_level}ft water level")
            
            # Compute flood zones
            analysis.compute_flood_zones(water_level)
            
            # Show flood statistics
            if analysis.flood_zones is not None:
                flooded_pixels = np.sum(analysis.flood_zones)
                total_pixels = analysis.flood_zones.size
                flood_percentage = flooded_pixels / total_pixels * 100
                
                print(f"   💧 Flooded area: {flood_percentage:.1f}% ({flooded_pixels:,} pixels)")
                
                if analysis.flood_depths is not None:
                    max_depth = np.max(analysis.flood_depths)
                    avg_depth = np.mean(analysis.flood_depths[analysis.flood_depths > 0])
                    print(f"   📏 Max flood depth: {max_depth:.2f}ft")
                    print(f"   📊 Avg flood depth: {avg_depth:.2f}ft")
        
        # Use the last water level for detailed analysis
        final_water_level = custom_water_levels[-1]
        analysis.compute_flood_zones(final_water_level)
        
        # Compute slope and flow accumulation
        print(f"\n⛰️ Computing slope from DEM...")
        analysis.compute_slope()
        
        print(f"💧 Computing flow accumulation...")
        analysis.compute_flow_accumulation()
        
        # Extract road elevations if roads are available
        if roads_path and analysis.roads_gdf is not None:
            print(f"\n🛣️ Extracting road elevations...")
            analysis.extract_road_elevations()
            
            print(f"⚠️ Computing flood risk scores...")
            analysis.compute_flood_risk_scores()
        
        # Export all results
        print(f"\n💾 Exporting results...")
        analysis.export_results()
        
        # Generate summary report
        print(f"\n📋 Generating summary report...")
        analysis.generate_summary_report()
        
        print(f"\n✅ Comprehensive flood analysis completed!")
        print(f"📁 Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in comprehensive flood analysis: {e}")
        print(f"❌ Error: {e}")
        return False


def run_data_clipping_example():
    """
    Demonstrate data clipping capabilities.
    
    This example shows how to:
    1. Clip shapefile data to specific geographic bounds
    2. Clip to DEM bounds for exact alignment
    3. Filter data by attributes
    """
    print("="*70)
    print("DATA CLIPPING EXAMPLE")
    print("="*70)
    
    roads_path = config.get_default_roads_path()
    dem_path = config.get_default_dem_path()
    
    if not roads_path.exists():
        print(f"❌ Roads file not found: {roads_path}")
        return False
    
    try:
        # Initialize clipper
        print("🔧 Initializing data clipper...")
        clipper = GeospatialDataClipper(roads_path)
        
        # Example 1: Clip to Greenwich Village bounds
        print("\n🏘️ Clipping to Greenwich Village bounds...")
        gv_bounds = config.GREENWICH_VILLAGE_BOUNDS
        
        # Use state plane coordinates if available
        if clipper.full_data.crs.to_epsg() == 2263:  # NY State Plane
            bounds = gv_bounds['state_plane']
            clipped_gv = clipper.clip_by_bounding_box(bounds)
        else:
            # Use WGS84 coordinates
            lat_range = gv_bounds['wgs84']['lat_range']
            lon_range = gv_bounds['wgs84']['lon_range']
            clipped_gv = clipper.clip_by_coordinates(lat_range, lon_range)
        
        print(f"✅ Greenwich Village clip: {len(clipped_gv)} road segments")
        
        # Example 2: Clip to DEM bounds (if DEM exists)
        if dem_path.exists():
            print("\n🗻 Clipping to DEM bounds...")
            clipped_dem_bounds = clipper.clip_by_dem_bounds(dem_path)
            
            if clipped_dem_bounds is not None:
                print(f"✅ DEM bounds clip: {len(clipped_dem_bounds)} road segments")
        
        # Example 3: Filter by street names
        print("\n🏷️ Filtering by street names...")
        street_keywords = ['BROADWAY', 'AVENUE', 'STREET']
        filtered_streets = clipper.clip_by_attribute_names(street_keywords)
        
        if filtered_streets is not None:
            print(f"✅ Street name filter: {len(filtered_streets)} road segments")
        
        # Visualize one of the clipping results
        if len(clipped_gv) > 0:
            print("\n🎨 Creating visualization...")
            clipper.visualize_clipping_result(
                clipper.full_data, 
                clipped_gv, 
                "Greenwich Village Roads"
            )
        
        print(f"\n✅ Data clipping examples completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in data clipping: {e}")
        return False


def run_scenario_comparison():
    """
    Compare different flood scenarios side by side.
    
    This example demonstrates:
    1. Running multiple flood scenarios
    2. Comparing results across scenarios
    3. Identifying critical water levels
    """
    print("="*70)
    print("FLOOD SCENARIO COMPARISON EXAMPLE")
    print("="*70)
    
    dem_path = config.get_default_dem_path()
    
    if not dem_path.exists():
        print(f"❌ DEM file not found: {dem_path}")
        return False
    
    try:
        # Initialize analyzer
        print("🔍 Initializing elevation analyzer...")
        analyzer = ElevationDataAnalyzer(dem_path)
        
        # Compute elevation statistics (required)
        analyzer.elevation_statistics()
        
        # Define scenarios
        scenarios = {
            "Minor Flooding": 5,    # 5 ft above sea level
            "Moderate Flooding": 10, # 10 ft above sea level  
            "Major Flooding": 15,   # 15 ft above sea level
            "Extreme Flooding": 25  # 25 ft above sea level
        }
        
        print(f"\n🌊 Analyzing {len(scenarios)} flood scenarios...")
        
        results = {}
        
        for scenario_name, water_level in scenarios.items():
            print(f"\n📊 Scenario: {scenario_name} ({water_level}ft)")
            
            # Calculate flood statistics
            if analyzer.valid_data is not None:
                flooded_pixels = np.sum(analyzer.valid_data <= water_level)
                total_pixels = len(analyzer.valid_data)
                flood_percentage = flooded_pixels / total_pixels * 100
                
                results[scenario_name] = {
                    'water_level': water_level,
                    'flooded_pixels': flooded_pixels,
                    'flood_percentage': flood_percentage
                }
                
                print(f"   💧 Flooded area: {flood_percentage:.1f}% ({flooded_pixels:,} pixels)")
        
        # Generate comparison report
        print(f"\n📋 SCENARIO COMPARISON SUMMARY")
        print("-" * 50)
        
        for scenario_name, data in results.items():
            print(f"{scenario_name:.<25} {data['flood_percentage']:>6.1f}% flooded")
        
        # Find critical thresholds
        print(f"\n⚠️ CRITICAL THRESHOLDS")
        print("-" * 30)
        
        for threshold in [5, 10, 25, 50]:  # percentage thresholds
            for scenario_name, data in results.items():
                if data['flood_percentage'] >= threshold:
                    print(f"{threshold:>2d}% flooding reached at: {data['water_level']}ft ({scenario_name})")
                    break
        
        # Create visualization for most extreme scenario
        extreme_level = max(scenarios.values())
        print(f"\n🎨 Creating visualization for extreme scenario ({extreme_level}ft)...")
        analyzer.visualize_flood_risk_map(extreme_level)
        
        print(f"\n✅ Scenario comparison completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in scenario comparison: {e}")
        return False


def main():
    """
    Main function for advanced flood analysis examples.
    """
    print("🌊 Advanced Flood Analysis Examples")
    print("="*50)
    print("Select an advanced example to run:")
    print("1. Comprehensive flood analysis (all features)")
    print("2. Data clipping and preprocessing")
    print("3. Flood scenario comparison")
    print("4. Run all advanced examples")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                print("Goodbye! 👋")
                break
            elif choice == '1':
                run_comprehensive_flood_analysis()
            elif choice == '2':
                run_data_clipping_example()
            elif choice == '3':
                run_scenario_comparison()
            elif choice == '4':
                print("🚀 Running all advanced examples...\n")
                run_data_clipping_example()
                print("\n" + "="*70 + "\n")
                run_scenario_comparison()
                print("\n" + "="*70 + "\n")
                run_comprehensive_flood_analysis()
                print("\n🎉 All advanced examples completed!")
            else:
                print("❌ Invalid choice. Please enter 0-4.")
                continue
                
            print(f"\n{'='*70}")
            print("Select another example or enter 0 to exit:")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()