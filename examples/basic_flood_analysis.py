#!/usr/bin/env python3
"""
Basic Flood Analysis Example

This example demonstrates how to perform a basic flood risk analysis
using the flood analysis toolkit.

Requirements:
- DEM file (Digital Elevation Model)
- Optional: Road network shapefile

Usage:
    python examples/basic_flood_analysis.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from elevation_data_analyzer import ElevationDataAnalyzer, RoadNetworkAnalyzer
from config import config
from logger import setup_logging
from validation import FileValidationError, DataValidationError


def run_basic_elevation_analysis():
    """
    Run basic elevation analysis without road network.
    
    This example shows how to:
    1. Load and validate DEM data
    2. Compute elevation statistics
    3. Analyze elevation distribution
    4. Generate flood risk maps
    5. Perform flood risk analysis at different water levels
    """
    print("="*60)
    print("BASIC ELEVATION ANALYSIS EXAMPLE")
    print("="*60)
    
    # Configure logging
    setup_logging(log_level='INFO', console_output=True)
    
    # Define file paths
    dem_path = config.get_default_dem_path()
    
    # Check if DEM file exists
    if not dem_path.exists():
        print(f"❌ DEM file not found: {dem_path}")
        print(f"Please ensure the DEM file exists at: {dem_path}")
        print("You can download NYC DEM data using nyc_data_download_guide2.py")
        return False
    
    try:
        # Initialize the elevation analyzer
        print(f"🔍 Initializing elevation analyzer...")
        analyzer = ElevationDataAnalyzer(dem_path)
        
        # Display basic information about the dataset
        print(f"\n📋 Dataset Information:")
        analyzer.basic_info()
        
        # Compute elevation statistics
        print(f"\n📊 Computing elevation statistics...")
        stats = analyzer.elevation_statistics()
        
        if stats:
            print(f"✅ Statistics computed successfully!")
            print(f"   - Elevation range: {stats['min']:.1f} to {stats['max']:.1f} feet")
            print(f"   - Mean elevation: {stats['mean']:.1f} feet")
            print(f"   - Valid pixels: {stats['count']:,}")
        
        # Analyze elevation distribution
        print(f"\n📈 Analyzing elevation distribution...")
        analyzer.elevation_distribution()
        
        # Perform flood risk analysis
        print(f"\n🌊 Performing flood risk analysis...")
        water_levels = config.DEFAULT_WATER_LEVELS
        analyzer.flood_risk_analysis(water_levels)
        
        # Create flood risk visualization
        print(f"\n🎨 Creating flood risk visualization...")
        flood_level = config.DEFAULT_FLOOD_LEVEL
        analyzer.visualize_flood_risk_map(flood_level)
        
        # Perform data quality check
        print(f"\n🔍 Checking data quality...")
        analyzer.data_quality_check()
        
        print(f"\n✅ Basic elevation analysis completed successfully!")
        return True
        
    except (FileValidationError, DataValidationError) as e:
        print(f"❌ Validation error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def run_elevation_with_roads_analysis():
    """
    Run elevation analysis with road network integration.
    
    This example shows how to:
    1. Load DEM and road network data
    2. Extract elevation values for road segments
    3. Assess flood risk for roads
    4. Create combined visualizations
    """
    print("="*60)
    print("ELEVATION + ROADS ANALYSIS EXAMPLE")
    print("="*60)
    
    # Define file paths
    dem_path = config.get_default_dem_path()
    roads_path = config.get_default_roads_path()
    
    # Check if required files exist
    if not dem_path.exists():
        print(f"❌ DEM file not found: {dem_path}")
        return False
    
    if not roads_path.exists():
        print(f"❌ Roads file not found: {roads_path}")
        return False
    
    try:
        # Initialize analyzers
        print(f"🔍 Initializing analyzers...")
        dem_analyzer = ElevationDataAnalyzer(dem_path)
        road_analyzer = RoadNetworkAnalyzer(roads_path)
        
        # Display basic information
        print(f"\n📋 Dataset Information:")
        dem_analyzer.basic_info()
        road_analyzer.basic_road_info()
        
        # Compute DEM statistics (required for road analysis)
        print(f"\n📊 Computing elevation statistics...")
        dem_analyzer.elevation_statistics()
        
        # Extract elevation values for road segments
        print(f"\n🛣️ Extracting road elevations...")
        roads_with_elevations = road_analyzer.extract_road_elevations(dem_analyzer)
        
        if roads_with_elevations is not None:
            print(f"✅ Road elevations extracted successfully!")
            
            # Assess flood risk for roads
            print(f"\n⚠️ Assessing flood risk for roads...")
            roads_with_risk = road_analyzer.assess_flood_risk(roads_with_elevations)
            
            if roads_with_risk is not None:
                print(f"✅ Flood risk assessment completed!")
                
                # Create combined visualization
                print(f"\n🎨 Creating combined visualization...")
                flood_level = config.DEFAULT_FLOOD_LEVEL
                road_analyzer.visualize_combined_elevation_and_roads(
                    roads_with_risk, dem_analyzer, flood_level
                )
                
                # Print summary statistics
                print(f"\n📈 Summary Statistics:")
                total_roads = len(roads_with_risk)
                high_risk = (roads_with_risk['flood_risk_10ft'] == 'High').sum()
                medium_risk = (roads_with_risk['flood_risk_10ft'] == 'Medium').sum()
                low_risk = (roads_with_risk['flood_risk_10ft'] == 'Low').sum()
                
                print(f"   - Total road segments: {total_roads:,}")
                print(f"   - High risk segments: {high_risk:,} ({high_risk/total_roads*100:.1f}%)")
                print(f"   - Medium risk segments: {medium_risk:,} ({medium_risk/total_roads*100:.1f}%)")
                print(f"   - Low risk segments: {low_risk:,} ({low_risk/total_roads*100:.1f}%)")
        
        print(f"\n✅ Combined analysis completed successfully!")
        return True
        
    except (FileValidationError, DataValidationError) as e:
        print(f"❌ Validation error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def run_quick_summary():
    """
    Run a quick elevation summary without full analysis.
    
    This is useful for:
    1. Quick data exploration
    2. Validating data files
    3. Getting basic statistics without time-intensive processing
    """
    print("="*60)
    print("QUICK ELEVATION SUMMARY EXAMPLE")
    print("="*60)
    
    from elevation_data_analyzer import quick_elevation_summary
    
    dem_path = config.get_default_dem_path()
    
    if not dem_path.exists():
        print(f"❌ DEM file not found: {dem_path}")
        return False
    
    try:
        quick_elevation_summary(dem_path)
        print(f"\n✅ Quick summary completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in quick summary: {e}")
        return False


def main():
    """
    Main function to run example analyses.
    
    This function provides a menu for selecting different analysis examples.
    """
    print("🌊 Flood Analysis Toolkit - Examples")
    print("="*50)
    print("Select an example to run:")
    print("1. Basic elevation analysis (DEM only)")
    print("2. Elevation + roads analysis (DEM + roads)")
    print("3. Quick elevation summary")
    print("4. Run all examples")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                print("Goodbye! 👋")
                break
            elif choice == '1':
                run_basic_elevation_analysis()
            elif choice == '2':
                run_elevation_with_roads_analysis()
            elif choice == '3':
                run_quick_summary()
            elif choice == '4':
                print("🚀 Running all examples...\n")
                run_quick_summary()
                print("\n" + "="*50 + "\n")
                run_basic_elevation_analysis()
                print("\n" + "="*50 + "\n")
                run_elevation_with_roads_analysis()
                print("\n🎉 All examples completed!")
            else:
                print("❌ Invalid choice. Please enter 0-4.")
                continue
                
            print(f"\n{'='*50}")
            print("Select another example or enter 0 to exit:")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()