#!/usr/bin/env python3
"""
Weather Risk Maps Generator

Creates NWS-style weather risk maps for the US, Australia, and Europe using Open-Meteo API.
Generates maps for:
- Severe Weather Risk (5 categories)
- Flooding Risk (4 categories)
- Fire Risk (4 categories)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import requests
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import os
import sys
import threading
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Import tkinter
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False


# =============================================================================
# REGION CONFIGURATIONS
# =============================================================================

REGIONS = {
    'us': {
        'name': 'United States',
        'bounds': [-125, -66, 24, 50],  # [lon_min, lon_max, lat_min, lat_max]
        'grid_resolution': 1.0,
        'projection': ccrs.LambertConformal(central_longitude=-96, central_latitude=39),
    },
    'australia': {
        'name': 'Australia',
        'bounds': [112, 154, -44, -10],
        'grid_resolution': 1.0,
        'projection': ccrs.LambertConformal(central_longitude=133, central_latitude=-27),
    },
    'europe': {
        'name': 'Europe',
        'bounds': [-12, 40, 35, 72],
        'grid_resolution': 1.0,
        'projection': ccrs.LambertConformal(central_longitude=10, central_latitude=50),
    }
}


# =============================================================================
# RISK CATEGORY DEFINITIONS
# =============================================================================

# Severe Weather Risk - 5 Categories (NWS SPC Style)
SEVERE_WEATHER_CATEGORIES = {
    0: {'name': 'None', 'color': '#FFFFFF', 'description': 'No severe weather expected'},
    1: {'name': 'Marginal', 'color': '#66A366', 'description': 'Isolated severe storms possible'},
    2: {'name': 'Slight', 'color': '#FFE066', 'description': 'Scattered severe storms possible'},
    3: {'name': 'Enhanced', 'color': '#FFA500', 'description': 'Numerous severe storms likely'},
    4: {'name': 'Moderate', 'color': '#FF0000', 'description': 'Widespread severe storms expected'},
    5: {'name': 'High', 'color': '#FF00FF', 'description': 'Major severe weather outbreak expected'}
}

# Flooding Risk - 4 Categories
FLOOD_RISK_CATEGORIES = {
    0: {'name': 'None', 'color': '#FFFFFF', 'description': 'No flooding expected'},
    1: {'name': 'Marginal', 'color': '#7CCD7C', 'description': 'Minor flooding possible'},
    2: {'name': 'Moderate', 'color': '#FFFF00', 'description': 'Moderate flooding likely'},
    3: {'name': 'High', 'color': '#FF6600', 'description': 'Significant flooding expected'},
    4: {'name': 'Extreme', 'color': '#CC0000', 'description': 'Major/flash flooding likely'}
}

# Fire Risk - 4 Categories
FIRE_RISK_CATEGORIES = {
    0: {'name': 'None', 'color': '#FFFFFF', 'description': 'No significant fire weather'},
    1: {'name': 'Elevated', 'color': '#FFFF99', 'description': 'Elevated fire weather conditions'},
    2: {'name': 'Critical', 'color': '#FF9900', 'description': 'Critical fire weather conditions'},
    3: {'name': 'Extreme', 'color': '#FF0000', 'description': 'Extreme fire weather conditions'},
    4: {'name': 'Exceptional', 'color': '#CC00CC', 'description': 'Exceptionally dangerous fire weather'}
}


# =============================================================================
# OPEN-METEO API INTERFACE
# =============================================================================

class OpenMeteoClient:
    """Client for fetching weather data from Open-Meteo API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.session = requests.Session()

    def fetch_weather_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Fetch weather data for a single location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dictionary with weather data or None if request fails
        """
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'rain',
                'showers',
                'snowfall',
                'weather_code',
                'wind_speed_10m',
                'wind_gusts_10m',
                'cape',
                'soil_moisture_0_to_7cm'
            ],
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'rain_sum',
                'precipitation_hours',
                'wind_speed_10m_max',
                'wind_gusts_10m_max'
            ],
            'timezone': 'UTC',
            'forecast_days': 3
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data for ({lat}, {lon}): {e}")
            return None

    def fetch_grid_data(self, bounds: List[float], resolution: float) -> Dict:
        """
        Fetch weather data for a grid of points.

        Args:
            bounds: [lon_min, lon_max, lat_min, lat_max]
            resolution: Grid spacing in degrees

        Returns:
            Dictionary with grid coordinates and weather data
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lats = np.arange(lat_min, lat_max + resolution, resolution)

        grid_data = {
            'lons': lons,
            'lats': lats,
            'data': {}
        }

        total_points = len(lons) * len(lats)
        print(f"Fetching data for {total_points} grid points...")

        count = 0
        for lat in lats:
            for lon in lons:
                data = self.fetch_weather_data(lat, lon)
                if data:
                    grid_data['data'][(lat, lon)] = data
                count += 1
                if count % 10 == 0:
                    print(f"  Progress: {count}/{total_points} points")

        print(f"Completed: {len(grid_data['data'])}/{total_points} points fetched successfully")
        return grid_data


# =============================================================================
# RISK CALCULATION ALGORITHMS
# =============================================================================

class RiskCalculator:
    """Calculate weather risk indices from meteorological data."""

    @staticmethod
    def calculate_severe_weather_risk(data: Dict) -> int:
        """
        Calculate severe weather risk (0-5) based on convective parameters.

        Uses: CAPE, wind gusts, precipitation intensity, weather codes

        Categories:
        0 - None: No severe weather expected
        1 - Marginal: Isolated severe storms possible
        2 - Slight: Scattered severe storms possible
        3 - Enhanced: Numerous severe storms likely
        4 - Moderate: Widespread severe storms expected
        5 - High: Major severe weather outbreak expected
        """
        if not data or 'hourly' not in data:
            return 0

        hourly = data['hourly']

        # Get maximum values for key parameters (next 24 hours)
        cape_values = hourly.get('cape', [0])[:24]
        wind_gust_values = hourly.get('wind_gusts_10m', [0])[:24]
        precip_values = hourly.get('precipitation', [0])[:24]
        weather_codes = hourly.get('weather_code', [0])[:24]

        # Handle None values
        cape_max = max([v for v in cape_values if v is not None] or [0])
        gust_max = max([v for v in wind_gust_values if v is not None] or [0])
        precip_max = max([v for v in precip_values if v is not None] or [0])

        # Check for thunderstorm weather codes (95-99)
        has_thunderstorm = any(95 <= (c or 0) <= 99 for c in weather_codes)

        # Calculate component scores
        cape_score = 0
        if cape_max > 3000:
            cape_score = 5
        elif cape_max > 2000:
            cape_score = 4
        elif cape_max > 1000:
            cape_score = 3
        elif cape_max > 500:
            cape_score = 2
        elif cape_max > 100:
            cape_score = 1

        gust_score = 0
        if gust_max > 130:  # km/h
            gust_score = 5
        elif gust_max > 100:
            gust_score = 4
        elif gust_max > 75:
            gust_score = 3
        elif gust_max > 50:
            gust_score = 2
        elif gust_max > 30:
            gust_score = 1

        precip_score = 0
        if precip_max > 50:  # mm/hr
            precip_score = 4
        elif precip_max > 25:
            precip_score = 3
        elif precip_max > 10:
            precip_score = 2
        elif precip_max > 2:
            precip_score = 1

        # Combine scores with weighting
        combined_score = (cape_score * 0.4 + gust_score * 0.35 + precip_score * 0.25)

        # Bonus for active thunderstorms
        if has_thunderstorm:
            combined_score += 0.5

        # Convert to category
        if combined_score >= 4.0:
            return 5
        elif combined_score >= 3.0:
            return 4
        elif combined_score >= 2.0:
            return 3
        elif combined_score >= 1.0:
            return 2
        elif combined_score >= 0.5:
            return 1
        else:
            return 0

    @staticmethod
    def calculate_flood_risk(data: Dict) -> int:
        """
        Calculate flooding risk (0-4) based on precipitation and soil moisture.

        Uses: Precipitation totals, precipitation duration, soil moisture

        Categories:
        0 - None: No flooding expected
        1 - Marginal: Minor flooding possible
        2 - Moderate: Moderate flooding likely
        3 - High: Significant flooding expected
        4 - Extreme: Major/flash flooding likely
        """
        if not data or 'daily' not in data:
            return 0

        daily = data['daily']
        hourly = data.get('hourly', {})

        # Get precipitation data
        precip_sum = daily.get('precipitation_sum', [0])[:3]
        precip_hours = daily.get('precipitation_hours', [0])[:3]
        rain_sum = daily.get('rain_sum', [0])[:3]

        # Get hourly precipitation for intensity
        hourly_precip = hourly.get('precipitation', [0])[:72]
        soil_moisture = hourly.get('soil_moisture_0_to_7cm', [0.3])[:24]

        # Handle None values
        total_precip = sum([v for v in precip_sum if v is not None] or [0])
        max_daily_precip = max([v for v in precip_sum if v is not None] or [0])
        total_hours = sum([v for v in precip_hours if v is not None] or [0])
        max_hourly = max([v for v in hourly_precip if v is not None] or [0])
        avg_soil_moisture = np.mean([v for v in soil_moisture if v is not None] or [0.3])

        # Precipitation volume score
        precip_score = 0
        if total_precip > 150:
            precip_score = 4
        elif total_precip > 100:
            precip_score = 3
        elif total_precip > 50:
            precip_score = 2
        elif total_precip > 25:
            precip_score = 1

        # Intensity score (flash flood potential)
        intensity_score = 0
        if max_hourly > 50:
            intensity_score = 4
        elif max_hourly > 30:
            intensity_score = 3
        elif max_hourly > 15:
            intensity_score = 2
        elif max_hourly > 5:
            intensity_score = 1

        # Soil saturation modifier
        saturation_modifier = 1.0
        if avg_soil_moisture > 0.4:
            saturation_modifier = 1.3
        elif avg_soil_moisture > 0.35:
            saturation_modifier = 1.15

        # Combined score
        combined_score = (precip_score * 0.5 + intensity_score * 0.5) * saturation_modifier

        # Convert to category
        if combined_score >= 3.5:
            return 4
        elif combined_score >= 2.5:
            return 3
        elif combined_score >= 1.5:
            return 2
        elif combined_score >= 0.5:
            return 1
        else:
            return 0

    @staticmethod
    def calculate_fire_risk(data: Dict) -> int:
        """
        Calculate fire weather risk (0-4) based on temperature, humidity, wind, and precipitation.

        Uses: Temperature, relative humidity, wind speed/gusts, precipitation history

        Categories:
        0 - None: No significant fire weather
        1 - Elevated: Elevated fire weather conditions
        2 - Critical: Critical fire weather conditions
        3 - Extreme: Extreme fire weather conditions
        4 - Exceptional: Exceptionally dangerous fire weather
        """
        if not data or 'hourly' not in data:
            return 0

        hourly = data['hourly']
        daily = data.get('daily', {})

        # Get relevant parameters
        temp_values = hourly.get('temperature_2m', [20])[:24]
        rh_values = hourly.get('relative_humidity_2m', [50])[:24]
        wind_values = hourly.get('wind_speed_10m', [10])[:24]
        gust_values = hourly.get('wind_gusts_10m', [15])[:24]
        precip_values = daily.get('precipitation_sum', [0])[:3]
        soil_moisture = hourly.get('soil_moisture_0_to_7cm', [0.3])[:24]

        # Handle None values and calculate statistics
        max_temp = max([v for v in temp_values if v is not None] or [20])
        min_rh = min([v for v in rh_values if v is not None] or [50])
        max_wind = max([v for v in wind_values if v is not None] or [10])
        max_gust = max([v for v in gust_values if v is not None] or [15])
        total_precip = sum([v for v in precip_values if v is not None] or [0])
        avg_soil = np.mean([v for v in soil_moisture if v is not None] or [0.3])

        # Temperature score (hot and dry conditions)
        temp_score = 0
        if max_temp > 40:
            temp_score = 4
        elif max_temp > 35:
            temp_score = 3
        elif max_temp > 30:
            temp_score = 2
        elif max_temp > 25:
            temp_score = 1

        # Humidity score (low humidity = high fire risk)
        rh_score = 0
        if min_rh < 10:
            rh_score = 4
        elif min_rh < 15:
            rh_score = 3
        elif min_rh < 25:
            rh_score = 2
        elif min_rh < 35:
            rh_score = 1

        # Wind score
        wind_score = 0
        effective_wind = max(max_wind, max_gust * 0.7)
        if effective_wind > 80:
            wind_score = 4
        elif effective_wind > 50:
            wind_score = 3
        elif effective_wind > 30:
            wind_score = 2
        elif effective_wind > 15:
            wind_score = 1

        # Drought modifier (low soil moisture and no recent precip)
        drought_modifier = 1.0
        if total_precip < 1 and avg_soil < 0.2:
            drought_modifier = 1.4
        elif total_precip < 5 and avg_soil < 0.25:
            drought_modifier = 1.2

        # Combined score with weighting
        combined_score = (temp_score * 0.25 + rh_score * 0.35 + wind_score * 0.4) * drought_modifier

        # Convert to category
        if combined_score >= 3.5:
            return 4
        elif combined_score >= 2.5:
            return 3
        elif combined_score >= 1.5:
            return 2
        elif combined_score >= 0.5:
            return 1
        else:
            return 0


# =============================================================================
# MAP GENERATION
# =============================================================================

class WeatherMapGenerator:
    """Generate NWS-style weather risk maps."""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.client = OpenMeteoClient()
        self.calculator = RiskCalculator()

    def _create_colormap(self, categories: Dict) -> Tuple[LinearSegmentedColormap, BoundaryNorm]:
        """Create a discrete colormap from category definitions."""
        colors = [categories[i]['color'] for i in sorted(categories.keys())]
        n_categories = len(categories)
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(-0.5, n_categories, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm

    def _create_legend_patches(self, categories: Dict) -> List[mpatches.Patch]:
        """Create legend patches for risk categories."""
        patches = []
        for i in sorted(categories.keys()):
            cat = categories[i]
            if i > 0:  # Skip 'None' category in legend
                patches.append(mpatches.Patch(
                    color=cat['color'],
                    label=f"{cat['name']}: {cat['description']}"
                ))
        return patches

    def _interpolate_grid(self, lons: np.ndarray, lats: np.ndarray,
                          risk_data: Dict, resolution: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate sparse risk data to a finer grid."""
        # Create fine grid
        lon_fine = np.arange(lons.min(), lons.max(), resolution)
        lat_fine = np.arange(lats.min(), lats.max(), resolution)
        lon_grid, lat_grid = np.meshgrid(lon_fine, lat_fine)

        # Prepare data for interpolation
        points = []
        values = []
        for (lat, lon), risk in risk_data.items():
            points.append([lon, lat])
            values.append(risk)

        if len(points) < 4:
            # Not enough points for interpolation
            risk_grid = np.zeros_like(lon_grid)
        else:
            points = np.array(points)
            values = np.array(values)

            # Interpolate
            risk_grid = griddata(points, values, (lon_grid, lat_grid), method='nearest')
            risk_grid = np.nan_to_num(risk_grid, nan=0)

        return lon_grid, lat_grid, risk_grid

    def generate_map(self, region_key: str, risk_type: str,
                     risk_data: Dict, categories: Dict,
                     title: str, grid_lons: np.ndarray, grid_lats: np.ndarray) -> str:
        """
        Generate a single risk map.

        Args:
            region_key: Region identifier
            risk_type: Type of risk map
            risk_data: Dictionary mapping (lat, lon) to risk level
            categories: Category definitions
            title: Map title
            grid_lons: Longitude array
            grid_lats: Latitude array

        Returns:
            Path to saved image
        """
        region = REGIONS[region_key]

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=region['projection'])

        # Set map extent
        ax.set_extent(region['bounds'], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='#F5F5F5', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF')
        ax.add_feature(cfeature.LAKES, facecolor='#E6F3FF', edgecolor='#6699CC', linewidth=0.5)
        ax.add_feature(cfeature.RIVERS, edgecolor='#6699CC', linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, edgecolor='#666666', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor='#333333', linewidth=0.8)

        # Add states/provinces for US
        if region_key == 'us':
            ax.add_feature(cfeature.STATES, edgecolor='#888888', linewidth=0.3)

        # Create colormap
        cmap, norm = self._create_colormap(categories)

        # Interpolate and plot risk data
        if risk_data:
            lon_grid, lat_grid, risk_grid = self._interpolate_grid(
                grid_lons, grid_lats, risk_data, resolution=0.25
            )

            # Plot filled contours
            mesh = ax.pcolormesh(
                lon_grid, lat_grid, risk_grid,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(),
                alpha=0.7,
                shading='auto'
            )

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Add title
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        plt.title(f"{title}\n{region['name']} - Valid: {timestamp}",
                  fontsize=14, fontweight='bold', pad=10)

        # Add legend
        legend_patches = self._create_legend_patches(categories)
        ax.legend(handles=legend_patches, loc='lower left',
                  fontsize=8, framealpha=0.9, title='Risk Categories')

        # Add data source attribution
        plt.figtext(0.99, 0.01, 'Data: Open-Meteo API',
                    ha='right', fontsize=8, style='italic', alpha=0.7)

        # Save figure
        filename = f"{region_key}_{risk_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved: {filepath}")
        return filepath

    def generate_region_maps(self, region_key: str, use_sample_data: bool = False) -> Dict[str, str]:
        """
        Generate all risk maps for a region.

        Args:
            region_key: Region identifier ('us', 'australia', 'europe')
            use_sample_data: If True, use sample data instead of API calls

        Returns:
            Dictionary mapping risk type to output file path
        """
        region = REGIONS[region_key]
        print(f"\n{'='*60}")
        print(f"Generating maps for {region['name']}")
        print('='*60)

        if use_sample_data:
            grid_data = self._generate_sample_data(region['bounds'], region['grid_resolution'])
        else:
            grid_data = self.client.fetch_grid_data(
                region['bounds'],
                region['grid_resolution']
            )

        # Calculate risks for each point
        severe_risks = {}
        flood_risks = {}
        fire_risks = {}

        print("Calculating risk levels...")
        for (lat, lon), weather_data in grid_data['data'].items():
            severe_risks[(lat, lon)] = self.calculator.calculate_severe_weather_risk(weather_data)
            flood_risks[(lat, lon)] = self.calculator.calculate_flood_risk(weather_data)
            fire_risks[(lat, lon)] = self.calculator.calculate_fire_risk(weather_data)

        # Generate maps
        output_files = {}

        print("Generating Severe Weather Risk map...")
        output_files['severe'] = self.generate_map(
            region_key, 'severe_weather',
            severe_risks, SEVERE_WEATHER_CATEGORIES,
            'Severe Weather Risk Outlook',
            grid_data['lons'], grid_data['lats']
        )

        print("Generating Flood Risk map...")
        output_files['flood'] = self.generate_map(
            region_key, 'flood_risk',
            flood_risks, FLOOD_RISK_CATEGORIES,
            'Flooding Risk Outlook',
            grid_data['lons'], grid_data['lats']
        )

        print("Generating Fire Risk map...")
        output_files['fire'] = self.generate_map(
            region_key, 'fire_risk',
            fire_risks, FIRE_RISK_CATEGORIES,
            'Fire Weather Risk Outlook',
            grid_data['lons'], grid_data['lats']
        )

        return output_files

    def _generate_sample_data(self, bounds: List[float], resolution: float) -> Dict:
        """Generate sample/demo data for testing without API calls."""
        lon_min, lon_max, lat_min, lat_max = bounds

        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lats = np.arange(lat_min, lat_max + resolution, resolution)

        grid_data = {
            'lons': lons,
            'lats': lats,
            'data': {}
        }

        # Generate synthetic weather data
        np.random.seed(42)  # For reproducibility

        for lat in lats:
            for lon in lons:
                # Create synthetic but realistic-looking weather patterns
                lat_factor = (lat - lat_min) / (lat_max - lat_min)
                lon_factor = (lon - lon_min) / (lon_max - lon_min)

                # Add some spatial correlation and randomness
                noise = np.random.random()
                pattern = np.sin(lat_factor * np.pi) * np.cos(lon_factor * np.pi * 2)

                grid_data['data'][(lat, lon)] = {
                    'hourly': {
                        'cape': [max(0, 500 + 2500 * (pattern + noise * 0.5))] * 24,
                        'wind_gusts_10m': [max(0, 20 + 80 * noise * abs(pattern))] * 24,
                        'precipitation': [max(0, 5 * (noise + pattern * 0.3))] * 24,
                        'weather_code': [int(95 * noise) if noise > 0.7 else int(3 * noise)] * 24,
                        'temperature_2m': [15 + 25 * lat_factor + 5 * noise] * 24,
                        'relative_humidity_2m': [30 + 40 * (1 - lat_factor) + 20 * noise] * 24,
                        'wind_speed_10m': [10 + 30 * noise * abs(pattern)] * 24,
                        'soil_moisture_0_to_7cm': [0.2 + 0.2 * (1 - noise)] * 24,
                    },
                    'daily': {
                        'precipitation_sum': [10 + 50 * noise * abs(pattern)] * 3,
                        'precipitation_hours': [int(5 + 10 * noise)] * 3,
                        'rain_sum': [8 + 40 * noise * abs(pattern)] * 3,
                        'temperature_2m_max': [20 + 20 * lat_factor] * 3,
                        'temperature_2m_min': [10 + 10 * lat_factor] * 3,
                        'wind_speed_10m_max': [15 + 40 * noise] * 3,
                        'wind_gusts_10m_max': [25 + 60 * noise] * 3,
                    }
                }

        return grid_data

    def generate_all_maps(self, use_sample_data: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Generate all risk maps for all regions.

        Args:
            use_sample_data: If True, use sample data instead of API calls

        Returns:
            Nested dictionary: region -> risk_type -> file_path
        """
        all_outputs = {}

        for region_key in REGIONS.keys():
            all_outputs[region_key] = self.generate_region_maps(region_key, use_sample_data)

        return all_outputs


# =============================================================================
# GRAPHICAL USER INTERFACE
# =============================================================================

class WeatherMapGUI:
    """Graphical User Interface for Weather Risk Maps Generator."""

    def __init__(self):
        if not HAS_TK:
            raise ImportError("tkinter is required for GUI mode")

        self.root = tk.Tk()
        self.root.title("Weather Risk Maps Generator")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # Set icon and styling
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 9))
        self.style.configure('Generate.TButton', font=('Helvetica', 11, 'bold'), padding=10)

        # Variables
        self.selected_region = tk.StringVar(value='us')
        self.selected_risk = tk.StringVar(value='all')
        self.use_demo_data = tk.BooleanVar(value=True)
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'output'))
        self.is_generating = False
        self.current_maps = {}

        # Generator
        self.generator = None

        # Build UI
        self._build_ui()

        # Center window
        self._center_window()

    def _center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _build_ui(self):
        """Build the main user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - Map preview
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_control_panel(left_panel)
        self._build_preview_panel(right_panel)
        self._build_status_bar()

    def _build_control_panel(self, parent):
        """Build the control panel with options."""
        # Title
        title_label = ttk.Label(parent, text="Weather Risk Maps", style='Title.TLabel')
        title_label.pack(pady=(0, 20))

        # Region Selection
        region_frame = ttk.LabelFrame(parent, text="Region", padding="10")
        region_frame.pack(fill=tk.X, pady=(0, 10))

        regions = [
            ('us', 'United States', 'Continental US with state boundaries'),
            ('australia', 'Australia', 'Full continental coverage'),
            ('europe', 'Europe', 'Western to Eastern Europe')
        ]

        for value, name, desc in regions:
            rb = ttk.Radiobutton(region_frame, text=name, value=value,
                                  variable=self.selected_region)
            rb.pack(anchor=tk.W)
            desc_label = ttk.Label(region_frame, text=f"  {desc}",
                                    font=('Helvetica', 8), foreground='gray')
            desc_label.pack(anchor=tk.W)

        # Risk Type Selection
        risk_frame = ttk.LabelFrame(parent, text="Risk Type", padding="10")
        risk_frame.pack(fill=tk.X, pady=(0, 10))

        risks = [
            ('all', 'All Risk Types'),
            ('severe', 'Severe Weather (5 categories)'),
            ('flood', 'Flooding Risk (4 categories)'),
            ('fire', 'Fire Weather (4 categories)')
        ]

        for value, name in risks:
            rb = ttk.Radiobutton(risk_frame, text=name, value=value,
                                  variable=self.selected_risk)
            rb.pack(anchor=tk.W, pady=2)

        # Data Source
        data_frame = ttk.LabelFrame(parent, text="Data Source", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        demo_cb = ttk.Checkbutton(data_frame, text="Use Demo Data (faster)",
                                   variable=self.use_demo_data)
        demo_cb.pack(anchor=tk.W)

        demo_desc = ttk.Label(data_frame,
                               text="Uncheck to fetch live data from Open-Meteo API\n(requires internet, may take several minutes)",
                               font=('Helvetica', 8), foreground='gray')
        demo_desc.pack(anchor=tk.W, pady=(5, 0))

        # Output Directory
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill=tk.X)

        dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir, width=25)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(dir_frame, text="Browse", command=self._browse_output)
        browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Generate Button
        self.generate_btn = ttk.Button(parent, text="Generate Maps",
                                        style='Generate.TButton',
                                        command=self._generate_maps)
        self.generate_btn.pack(fill=tk.X, pady=(20, 10))

        # Progress
        self.progress_frame = ttk.Frame(parent)
        self.progress_frame.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                             maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)

        self.progress_label = ttk.Label(self.progress_frame, text="", style='Status.TLabel')
        self.progress_label.pack(anchor=tk.W, pady=(5, 0))

        # Generated Maps List
        maps_frame = ttk.LabelFrame(parent, text="Generated Maps", padding="10")
        maps_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Listbox with scrollbar
        list_frame = ttk.Frame(maps_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.maps_listbox = tk.Listbox(list_frame, height=6,
                                        yscrollcommand=scrollbar.set,
                                        font=('Helvetica', 9))
        self.maps_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.maps_listbox.yview)

        self.maps_listbox.bind('<<ListboxSelect>>', self._on_map_select)

        # Open folder button
        open_folder_btn = ttk.Button(maps_frame, text="Open Output Folder",
                                      command=self._open_output_folder)
        open_folder_btn.pack(fill=tk.X, pady=(10, 0))

    def _build_preview_panel(self, parent):
        """Build the map preview panel."""
        # Preview header
        preview_header = ttk.Frame(parent)
        preview_header.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(preview_header, text="Map Preview", style='Header.TLabel').pack(side=tk.LEFT)

        self.preview_label = ttk.Label(preview_header, text="", foreground='gray')
        self.preview_label.pack(side=tk.RIGHT)

        # Preview canvas frame with border
        self.preview_frame = ttk.Frame(parent, relief='sunken', borderwidth=2)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        # Placeholder
        self.placeholder_label = ttk.Label(
            self.preview_frame,
            text="Select options and click 'Generate Maps'\nto create weather risk maps.\n\n"
                 "Generated maps will appear here.",
            font=('Helvetica', 12),
            foreground='gray',
            justify=tk.CENTER
        )
        self.placeholder_label.pack(expand=True)

        # Canvas for map display (hidden initially)
        self.canvas = None
        self.current_figure = None

    def _build_status_bar(self):
        """Build the status bar at the bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Separator(status_frame, orient='horizontal').pack(fill=tk.X)

        status_inner = ttk.Frame(status_frame, padding="5")
        status_inner.pack(fill=tk.X)

        self.status_label = ttk.Label(status_inner, text="Ready", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT)

        # Data source indicator
        self.data_source_label = ttk.Label(status_inner, text="Data: Open-Meteo API",
                                            style='Status.TLabel', foreground='gray')
        self.data_source_label.pack(side=tk.RIGHT)

    def _browse_output(self):
        """Open directory browser for output folder."""
        directory = filedialog.askdirectory(
            initialdir=self.output_dir.get(),
            title="Select Output Directory"
        )
        if directory:
            self.output_dir.set(directory)

    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            if sys.platform == 'win32':
                os.startfile(output_path)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_path}"')
            else:
                os.system(f'xdg-open "{output_path}"')
        else:
            messagebox.showwarning("Folder Not Found",
                                   f"Output folder does not exist:\n{output_path}")

    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def _update_progress(self, value: float, message: str = ""):
        """Update progress bar and label."""
        self.progress_var.set(value)
        if message:
            self.progress_label.config(text=message)
        self.root.update_idletasks()

    def _generate_maps(self):
        """Start map generation in a separate thread."""
        if self.is_generating:
            return

        self.is_generating = True
        self.generate_btn.config(state='disabled')
        self.maps_listbox.delete(0, tk.END)
        self.current_maps = {}

        # Clear preview
        self._clear_preview()
        self.placeholder_label.config(text="Generating maps...")

        # Start generation thread
        thread = threading.Thread(target=self._generate_maps_thread, daemon=True)
        thread.start()

    def _generate_maps_thread(self):
        """Generate maps in background thread."""
        try:
            region = self.selected_region.get()
            risk_type = self.selected_risk.get()
            use_demo = self.use_demo_data.get()
            output_dir = self.output_dir.get()

            self.root.after(0, lambda: self._update_status(f"Initializing generator..."))
            self.root.after(0, lambda: self._update_progress(5, "Initializing..."))

            # Create generator
            self.generator = WeatherMapGenerator(output_dir=output_dir)

            # Get region config
            region_config = REGIONS[region]

            self.root.after(0, lambda: self._update_progress(10, "Fetching weather data..."))
            self.root.after(0, lambda: self._update_status(
                f"{'Generating sample' if use_demo else 'Fetching'} data for {region_config['name']}..."))

            # Fetch/generate data
            if use_demo:
                grid_data = self.generator._generate_sample_data(
                    region_config['bounds'],
                    region_config['grid_resolution']
                )
            else:
                grid_data = self.generator.client.fetch_grid_data(
                    region_config['bounds'],
                    region_config['grid_resolution']
                )

            self.root.after(0, lambda: self._update_progress(40, "Calculating risk levels..."))
            self.root.after(0, lambda: self._update_status("Calculating risk levels..."))

            # Calculate risks
            severe_risks = {}
            flood_risks = {}
            fire_risks = {}

            for (lat, lon), weather_data in grid_data['data'].items():
                severe_risks[(lat, lon)] = self.generator.calculator.calculate_severe_weather_risk(weather_data)
                flood_risks[(lat, lon)] = self.generator.calculator.calculate_flood_risk(weather_data)
                fire_risks[(lat, lon)] = self.generator.calculator.calculate_fire_risk(weather_data)

            # Generate requested maps
            output_files = {}
            progress_base = 50
            maps_to_generate = []

            if risk_type in ['all', 'severe']:
                maps_to_generate.append(('severe', 'severe_weather', severe_risks,
                                         SEVERE_WEATHER_CATEGORIES, 'Severe Weather Risk Outlook'))
            if risk_type in ['all', 'flood']:
                maps_to_generate.append(('flood', 'flood_risk', flood_risks,
                                         FLOOD_RISK_CATEGORIES, 'Flooding Risk Outlook'))
            if risk_type in ['all', 'fire']:
                maps_to_generate.append(('fire', 'fire_risk', fire_risks,
                                         FIRE_RISK_CATEGORIES, 'Fire Weather Risk Outlook'))

            progress_per_map = 45 / len(maps_to_generate)

            for i, (key, risk_name, risks, categories, title) in enumerate(maps_to_generate):
                progress = progress_base + (i * progress_per_map)
                self.root.after(0, lambda p=progress, t=title: (
                    self._update_progress(p, f"Generating {t}..."),
                    self._update_status(f"Generating {t}...")
                ))

                filepath = self.generator.generate_map(
                    region, risk_name, risks, categories, title,
                    grid_data['lons'], grid_data['lats']
                )
                output_files[key] = filepath

            # Update UI with results
            self.root.after(0, lambda: self._on_generation_complete(output_files))

        except Exception as e:
            self.root.after(0, lambda: self._on_generation_error(str(e)))

    def _on_generation_complete(self, output_files: Dict[str, str]):
        """Handle successful generation completion."""
        self.is_generating = False
        self.generate_btn.config(state='normal')
        self._update_progress(100, "Complete!")
        self._update_status(f"Generated {len(output_files)} map(s) successfully")

        # Store and display results
        self.current_maps = output_files

        for risk_type, filepath in output_files.items():
            filename = os.path.basename(filepath)
            self.maps_listbox.insert(tk.END, filename)

        # Show first map
        if output_files:
            self.maps_listbox.selection_set(0)
            self._on_map_select(None)

        self.placeholder_label.pack_forget()

    def _on_generation_error(self, error_msg: str):
        """Handle generation error."""
        self.is_generating = False
        self.generate_btn.config(state='normal')
        self._update_progress(0, "")
        self._update_status("Error occurred")
        self.placeholder_label.config(text=f"Error: {error_msg}\n\nPlease try again.")
        messagebox.showerror("Generation Error", f"An error occurred:\n{error_msg}")

    def _on_map_select(self, event):
        """Handle map selection from listbox."""
        selection = self.maps_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        filename = self.maps_listbox.get(index)

        # Find corresponding filepath
        for risk_type, filepath in self.current_maps.items():
            if os.path.basename(filepath) == filename:
                self._show_map_preview(filepath)
                break

    def _clear_preview(self):
        """Clear the map preview."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None

    def _show_map_preview(self, filepath: str):
        """Display a map in the preview panel."""
        self._clear_preview()

        try:
            # Load and display image using matplotlib
            from PIL import Image

            img = Image.open(filepath)

            # Create figure for display
            self.current_figure = plt.figure(figsize=(10, 7), dpi=100)
            ax = self.current_figure.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            self.current_figure.tight_layout(pad=0)

            # Embed in tkinter
            self.canvas = FigureCanvasTkAgg(self.current_figure, master=self.preview_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update preview label
            self.preview_label.config(text=os.path.basename(filepath))

        except Exception as e:
            self.placeholder_label.config(text=f"Error loading preview:\n{e}")
            self.placeholder_label.pack(expand=True)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for the weather risk maps generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate NWS-style weather risk maps using Open-Meteo data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                Launch GUI (default)
  %(prog)s --gui                          Launch GUI explicitly
  %(prog)s --cli --region us              Generate US maps via CLI
  %(prog)s --cli --region europe --demo   Generate Europe maps with sample data
  %(prog)s --cli --all --demo             Generate all maps with sample data
  %(prog)s --cli --all --output ./maps    Generate all maps, save to ./maps/
        """
    )

    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch graphical user interface (default if no arguments)'
    )

    parser.add_argument(
        '--cli', '-c',
        action='store_true',
        help='Run in command-line mode'
    )

    parser.add_argument(
        '--region', '-r',
        choices=['us', 'australia', 'europe'],
        help='Region to generate maps for (CLI mode)'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate maps for all regions (CLI mode)'
    )

    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Use sample data instead of live API calls (for testing)'
    )

    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory for generated maps (default: output)'
    )

    parser.add_argument(
        '--risk-type', '-t',
        choices=['severe', 'flood', 'fire', 'all'],
        default='all',
        help='Type of risk map to generate (default: all)'
    )

    args = parser.parse_args()

    # Determine mode: GUI by default, CLI if --cli or other CLI args specified
    use_cli = args.cli or args.region or args.all

    if not use_cli:
        # Launch GUI
        if not HAS_TK:
            print("Error: tkinter is not available. Please install it or use --cli mode.")
            print("On Ubuntu/Debian: sudo apt-get install python3-tk")
            print("On Fedora: sudo dnf install python3-tkinter")
            print("On macOS: tkinter is usually included with Python")
            sys.exit(1)

        try:
            print("Launching Weather Risk Maps GUI...")
            gui = WeatherMapGUI()
            gui.run()
        except Exception as e:
            print(f"Error launching GUI: {e}")
            print("Try running with --cli mode instead.")
            sys.exit(1)
    else:
        # CLI mode
        if not args.region and not args.all:
            parser.error("In CLI mode, please specify --region or --all")

        print("="*60)
        print("Weather Risk Maps Generator (CLI Mode)")
        print("="*60)
        print(f"Output directory: {args.output}")
        print(f"Data source: {'Sample data' if args.demo else 'Open-Meteo API'}")
        print()

        generator = WeatherMapGenerator(output_dir=args.output)

        if args.all:
            outputs = generator.generate_all_maps(use_sample_data=args.demo)
        else:
            outputs = {args.region: generator.generate_region_maps(args.region, use_sample_data=args.demo)}

        print("\n" + "="*60)
        print("Generation Complete!")
        print("="*60)
        print("\nGenerated files:")
        for region, files in outputs.items():
            print(f"\n{REGIONS[region]['name']}:")
            for risk_type, filepath in files.items():
                print(f"  - {risk_type}: {filepath}")

        return outputs


if __name__ == '__main__':
    main()
