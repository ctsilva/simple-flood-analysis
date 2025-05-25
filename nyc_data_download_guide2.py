#!/usr/bin/env python3
"""
NYC Open Data - CORRECT Download Information and Manual Instructions

The previous script had incorrect API URLs. NYC Open Data doesn't provide 
direct API download URLs in the format I specified. Here are the CORRECT 
methods to get the data.
"""

import os
import webbrowser
from pathlib import Path

def print_correct_download_instructions():
    """Print the correct manual download instructions"""
    print("=" * 80)
    print("NYC OPEN DATA - CORRECT DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    
    print("\n🏗️  NYC DIGITAL ELEVATION MODEL (DEM)")
    print("-" * 50)
    print("✅ WORKING URL: https://data.cityofnewyork.us/City-Government/1-foot-Digital-Elevation-Model-DEM-/dpc8-z3jc")
    print("📂 DIRECT DOWNLOAD LINK FOUND:")
    print("   https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Float_2.zip")
    print("\n🔧 Manual Steps:")
    print("   1. Go to the dataset page above")
    print("   2. Click the 'Export' dropdown button")
    print("   3. Select 'Download' > 'Original' format")
    print("   4. This will download 'NYC_DEM_1ft_Float_2.zip' (~1.5GB)")
    
    print("\n🛣️  NYC STREET CENTERLINES (ROADS)")
    print("-" * 50)
    print("✅ CURRENT DATASET: https://data.cityofnewyork.us/City-Government/Centerline/3mf9-qshr")
    print("⚠️  NOTE: The old LION dataset (2v4z-66xt) is deprecated")
    
    print("\n🔧 Manual Steps:")
    print("   1. Go to the current Centerline dataset URL above")
    print("   2. Click the 'Export' dropdown button")
    print("   3. Select 'Shapefile' format")
    print("   4. Download and extract the ZIP file")
    
    print("\n📋 ALTERNATIVE METHODS:")
    print("-" * 50)
    print("1. 🌐 BROWSER METHOD (Recommended):")
    print("   - Visit the URLs above directly")
    print("   - Use the Export buttons on each page")
    print("   - This is the most reliable method")
    
    print("\n2. 🐍 PYTHON WITH REQUESTS (Try this):")
    print("   - The S3 link above for DEM might work with requests")
    print("   - Street centerlines need to be downloaded manually")
    
    print("\n3. 🗺️  ALTERNATIVE DATA SOURCES:")
    print("   DEM Alternatives:")
    print("   - USGS National Map: https://apps.nationalmap.gov/downloader/")
    print("   - NY State GIS FTP: ftp://ftp.gis.ny.gov/elevation/DEM/")
    print("   - OpenTopography: https://portal.opentopography.org/")
    
    print("\n   Road Alternatives:")
    print("   - OpenStreetMap NYC: https://download.geofabrik.de/north-america/us/new-york.html")
    print("   - TIGER/Line (Census): https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
    
    print("\n📁 EXPECTED FILES AFTER DOWNLOAD:")
    print("-" * 50)
    print("nyc_data/")
    print("├── NYC_DEM_1ft_Float_2.zip (extract to get .img or .tif files)")
    print("└── centerline_shapefile.zip (extract to get .shp, .shx, .dbf, .prj)")

def attempt_direct_dem_download():
    """Attempt to download DEM using the direct S3 link found"""
    import requests
    
    dem_url = "https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Float_2.zip"
    
    print("\n🔄 Attempting direct DEM download...")
    print(f"URL: {dem_url}")
    
    try:
        response = requests.head(dem_url, timeout=10)
        if response.status_code == 200:
            print("✅ Direct download link is accessible!")
            
            download_choice = input("Download the DEM file now? (y/n): ").lower().strip()
            if download_choice == 'y':
                print("Starting download... (This will take several minutes)")
                
                # Create data directory
                data_dir = Path("nyc_data")
                data_dir.mkdir(exist_ok=True)
                
                # Download file
                response = requests.get(dem_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                filepath = data_dir / "NYC_DEM_1ft_Float_2.zip"
                downloaded = 0
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                
                print(f"\n✅ Downloaded: {filepath}")
                return filepath
        else:
            print(f"❌ Direct link not accessible (Status: {response.status_code})")
            
    except Exception as e:
        print(f"❌ Error accessing direct link: {e}")
    
    return None

def open_data_pages():
    """Open the data pages in browser"""
    print("\n🌐 Opening NYC Open Data pages in your browser...")
    
    urls = [
        "https://data.cityofnewyork.us/City-Government/1-foot-Digital-Elevation-Model-DEM-/dpc8-z3jc",
        "https://data.cityofnewyork.us/City-Government/Centerline/3mf9-qshr"
    ]
    
    for url in urls:
        try:
            webbrowser.open(url)
            print(f"✅ Opened: {url}")
        except:
            print(f"❌ Could not open: {url}")
    
    print("\nℹ️  Look for the 'Export' button on each page and select your preferred format.")

def verify_downloaded_data():
    """Help verify the downloaded data"""
    print("\n🔍 DATA VERIFICATION HELPER")
    print("-" * 30)
    
    data_dir = Path("nyc_data")
    if not data_dir.exists():
        print("❌ No nyc_data directory found. Please download data first.")
        return
    
    files = list(data_dir.glob("*"))
    if not files:
        print("❌ No files found in nyc_data directory.")
        return
    
    print("📁 Files found:")
    for file in files:
        print(f"   - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    
    # Try to verify with geopandas/rasterio if available
    try:
        import geopandas as gpd
        import rasterio
        
        # Look for shapefiles
        shp_files = list(data_dir.glob("**/*.shp"))
        for shp in shp_files:
            try:
                gdf = gpd.read_file(shp)
                print(f"\n✅ Shapefile: {shp.name}")
                print(f"   Features: {len(gdf)}")
                print(f"   CRS: {gdf.crs}")
                print(f"   Columns: {list(gdf.columns[:5])}...")  # First 5 columns
            except Exception as e:
                print(f"❌ Error reading {shp.name}: {e}")
        
        # Look for raster files
        raster_extensions = ['*.tif', '*.img', '*.tiff']
        for ext in raster_extensions:
            raster_files = list(data_dir.glob(f"**/{ext}"))
            for raster in raster_files:
                try:
                    with rasterio.open(raster) as src:
                        print(f"\n✅ Raster: {raster.name}")
                        print(f"   Shape: {src.width} x {src.height}")
                        print(f"   CRS: {src.crs}")
                        print(f"   Bounds: {src.bounds}")
                except Exception as e:
                    print(f"❌ Error reading {raster.name}: {e}")
                    
    except ImportError:
        print("\nℹ️  Install geopandas and rasterio to verify data:")
        print("   pip install geopandas rasterio")

def main():
    """Main menu for NYC data download assistance"""
    print("NYC OPEN DATA DOWNLOAD HELPER")
    print("=" * 40)
    print("Choose an option:")
    print("1. 📖 Show correct download instructions")
    print("2. 🔄 Try direct DEM download (experimental)")
    print("3. 🌐 Open data pages in browser")
    print("4. 🔍 Verify downloaded data")
    print("5. ❌ Exit")
    
    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print_correct_download_instructions()
        
        elif choice == "2":
            result = attempt_direct_dem_download()
            if result:
                print(f"\n✅ DEM downloaded successfully!")
                print("⚠️  You still need to manually download the street centerlines.")
        
        elif choice == "3":
            open_data_pages()
        
        elif choice == "4":
            verify_downloaded_data()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
