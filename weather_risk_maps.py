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

    # Predefined regions with display info
    REGION_INFO = {
        'us': {'name': 'United States', 'flag': 'US', 'desc': 'Continental US with state boundaries'},
        'australia': {'name': 'Australia', 'flag': 'AU', 'desc': 'Full continental coverage'},
        'europe': {'name': 'Europe', 'flag': 'EU', 'desc': 'Western to Eastern Europe'},
        'custom': {'name': 'Custom Area', 'flag': 'CUSTOM', 'desc': 'Define your own region'}
    }

    # Event type info with colors
    EVENT_TYPES = {
        'severe': {
            'name': 'Severe Weather',
            'desc': 'Thunderstorms, tornadoes, hail, damaging winds',
            'color': '#FF6B6B',
            'categories': 5
        },
        'flood': {
            'name': 'Flooding Risk',
            'desc': 'Flash floods, river flooding, urban flooding',
            'color': '#4ECDC4',
            'categories': 4
        },
        'fire': {
            'name': 'Fire Weather',
            'desc': 'Wildfire conditions, red flag warnings',
            'color': '#FF8C42',
            'categories': 4
        }
    }

    def __init__(self):
        if not HAS_TK:
            raise ImportError("tkinter is required for GUI mode")

        self.root = tk.Tk()
        self.root.title("Weather Risk Maps Generator")
        self.root.geometry("1300x850")
        self.root.minsize(1000, 700)

        # Set icon and styling
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('SubHeader.TLabel', font=('Helvetica', 10, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 9))
        self.style.configure('Generate.TButton', font=('Helvetica', 12, 'bold'), padding=12)
        self.style.configure('Region.TRadiobutton', font=('Helvetica', 11))
        self.style.configure('Event.TCheckbutton', font=('Helvetica', 10))

        # Variables
        self.selected_region = tk.StringVar(value='us')
        self.use_demo_data = tk.BooleanVar(value=True)
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'output'))
        self.is_generating = False
        self.current_maps = {}

        # Event type checkboxes
        self.event_severe = tk.BooleanVar(value=True)
        self.event_flood = tk.BooleanVar(value=True)
        self.event_fire = tk.BooleanVar(value=True)

        # Custom region coordinates
        self.custom_lat_min = tk.StringVar(value='25.0')
        self.custom_lat_max = tk.StringVar(value='50.0')
        self.custom_lon_min = tk.StringVar(value='-125.0')
        self.custom_lon_max = tk.StringVar(value='-65.0')

        # Generator
        self.generator = None

        # Map click state
        self.map_click_start = None
        self.selection_rect = None

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

        # Left panel - Controls (wider now)
        left_panel = ttk.Frame(main_frame, width=380)
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
        # Scrollable frame for controls
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Title with icon
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = ttk.Label(title_frame, text="Weather Risk Maps",
                                 style='Title.TLabel')
        title_label.pack()

        subtitle = ttk.Label(title_frame, text="Generate NWS-style risk outlook maps",
                              font=('Helvetica', 9), foreground='gray')
        subtitle.pack()

        # ============ STEP 1: SELECT REGION ============
        step1_frame = ttk.LabelFrame(scrollable_frame, text="Step 1: Select Region", padding="10")
        step1_frame.pack(fill=tk.X, pady=(0, 10))

        # Region radio buttons with better layout
        regions_data = [
            ('us', 'United States', 'Continental US'),
            ('australia', 'Australia', 'Full continent'),
            ('europe', 'Europe', 'West to East'),
            ('custom', 'Custom Region', 'Define coordinates')
        ]

        for value, name, desc in regions_data:
            region_row = ttk.Frame(step1_frame)
            region_row.pack(fill=tk.X, pady=2)

            rb = ttk.Radiobutton(region_row, text=name, value=value,
                                  variable=self.selected_region,
                                  command=self._on_region_change,
                                  style='Region.TRadiobutton')
            rb.pack(side=tk.LEFT)

            desc_label = ttk.Label(region_row, text=f"({desc})",
                                    font=('Helvetica', 8), foreground='gray')
            desc_label.pack(side=tk.LEFT, padx=(5, 0))

        # Custom coordinates frame (hidden by default)
        self.custom_coords_frame = ttk.Frame(step1_frame)

        ttk.Separator(self.custom_coords_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        coord_label = ttk.Label(self.custom_coords_frame, text="Enter coordinates or click map to select:",
                                 font=('Helvetica', 9, 'italic'))
        coord_label.pack(anchor=tk.W)

        # Latitude row
        lat_frame = ttk.Frame(self.custom_coords_frame)
        lat_frame.pack(fill=tk.X, pady=5)

        ttk.Label(lat_frame, text="Latitude:", width=10).pack(side=tk.LEFT)
        ttk.Label(lat_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(lat_frame, textvariable=self.custom_lat_min, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(lat_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(lat_frame, textvariable=self.custom_lat_max, width=8).pack(side=tk.LEFT, padx=2)

        # Longitude row
        lon_frame = ttk.Frame(self.custom_coords_frame)
        lon_frame.pack(fill=tk.X, pady=5)

        ttk.Label(lon_frame, text="Longitude:", width=10).pack(side=tk.LEFT)
        ttk.Label(lon_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(lon_frame, textvariable=self.custom_lon_min, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(lon_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(lon_frame, textvariable=self.custom_lon_max, width=8).pack(side=tk.LEFT, padx=2)

        # Quick presets for custom
        presets_frame = ttk.Frame(self.custom_coords_frame)
        presets_frame.pack(fill=tk.X, pady=5)

        ttk.Label(presets_frame, text="Quick presets:", font=('Helvetica', 8)).pack(side=tk.LEFT)

        presets = [
            ('Texas', -107, -93, 25, 37),
            ('California', -125, -114, 32, 42),
            ('UK', -8, 2, 50, 59),
            ('Japan', 129, 146, 31, 46),
        ]

        for name, lon_min, lon_max, lat_min, lat_max in presets:
            btn = ttk.Button(presets_frame, text=name, width=8,
                             command=lambda a=lon_min, b=lon_max, c=lat_min, d=lat_max:
                                 self._set_custom_coords(a, b, c, d))
            btn.pack(side=tk.LEFT, padx=2)

        # ============ STEP 2: SELECT EVENT TYPES ============
        step2_frame = ttk.LabelFrame(scrollable_frame, text="Step 2: Select Event Types", padding="10")
        step2_frame.pack(fill=tk.X, pady=(0, 10))

        instruction = ttk.Label(step2_frame, text="Select which risk maps to generate:",
                                 font=('Helvetica', 9))
        instruction.pack(anchor=tk.W, pady=(0, 5))

        # Event checkboxes with color indicators
        events = [
            (self.event_severe, 'severe', 'Severe Weather', '#FF6B6B', '5 categories: Marginal to High'),
            (self.event_flood, 'flood', 'Flooding Risk', '#4ECDC4', '4 categories: Marginal to Extreme'),
            (self.event_fire, 'fire', 'Fire Weather', '#FF8C42', '4 categories: Elevated to Exceptional'),
        ]

        for var, key, name, color, desc in events:
            event_row = ttk.Frame(step2_frame)
            event_row.pack(fill=tk.X, pady=3)

            # Color indicator
            color_canvas = tk.Canvas(event_row, width=16, height=16, highlightthickness=1,
                                      highlightbackground='gray')
            color_canvas.pack(side=tk.LEFT, padx=(0, 5))
            color_canvas.create_rectangle(2, 2, 14, 14, fill=color, outline='')

            cb = ttk.Checkbutton(event_row, text=name, variable=var,
                                  style='Event.TCheckbutton')
            cb.pack(side=tk.LEFT)

            desc_label = ttk.Label(event_row, text=f"- {desc}",
                                    font=('Helvetica', 8), foreground='gray')
            desc_label.pack(side=tk.LEFT, padx=(5, 0))

        # Select all/none buttons
        select_btns = ttk.Frame(step2_frame)
        select_btns.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(select_btns, text="Select All", width=10,
                   command=self._select_all_events).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_btns, text="Select None", width=10,
                   command=self._select_no_events).pack(side=tk.LEFT, padx=2)

        # ============ STEP 3: DATA SOURCE ============
        step3_frame = ttk.LabelFrame(scrollable_frame, text="Step 3: Data Source", padding="10")
        step3_frame.pack(fill=tk.X, pady=(0, 10))

        demo_cb = ttk.Checkbutton(step3_frame, text="Use Demo Data (faster, for testing)",
                                   variable=self.use_demo_data)
        demo_cb.pack(anchor=tk.W)

        demo_desc = ttk.Label(step3_frame,
                               text="Uncheck to fetch live weather data from Open-Meteo API.\n"
                                    "Live data requires internet and takes longer to process.",
                               font=('Helvetica', 8), foreground='gray')
        demo_desc.pack(anchor=tk.W, pady=(5, 0))

        # ============ STEP 4: OUTPUT LOCATION ============
        step4_frame = ttk.LabelFrame(scrollable_frame, text="Step 4: Save Location", padding="10")
        step4_frame.pack(fill=tk.X, pady=(0, 10))

        dir_frame = ttk.Frame(step4_frame)
        dir_frame.pack(fill=tk.X)

        dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(dir_frame, text="Browse...", command=self._browse_output)
        browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # ============ GENERATE BUTTON ============
        self.generate_btn = ttk.Button(scrollable_frame, text="Generate Maps",
                                        style='Generate.TButton',
                                        command=self._generate_maps)
        self.generate_btn.pack(fill=tk.X, pady=(15, 10))

        # Progress section
        self.progress_frame = ttk.Frame(scrollable_frame)
        self.progress_frame.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                             maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)

        self.progress_label = ttk.Label(self.progress_frame, text="Ready to generate",
                                         style='Status.TLabel')
        self.progress_label.pack(anchor=tk.W, pady=(5, 0))

        # ============ GENERATED MAPS LIST ============
        maps_frame = ttk.LabelFrame(scrollable_frame, text="Generated Maps", padding="10")
        maps_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Listbox with scrollbar
        list_frame = ttk.Frame(maps_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        list_scrollbar = ttk.Scrollbar(list_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.maps_listbox = tk.Listbox(list_frame, height=5,
                                        yscrollcommand=list_scrollbar.set,
                                        font=('Helvetica', 9),
                                        selectbackground='#4ECDC4')
        self.maps_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.maps_listbox.yview)

        self.maps_listbox.bind('<<ListboxSelect>>', self._on_map_select)

        # Button row
        btn_row = ttk.Frame(maps_frame)
        btn_row.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_row, text="Open Folder",
                   command=self._open_output_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="View Selected",
                   command=lambda: self._on_map_select(None)).pack(side=tk.LEFT, padx=2)

    def _on_region_change(self):
        """Handle region selection change."""
        if self.selected_region.get() == 'custom':
            self.custom_coords_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.custom_coords_frame.pack_forget()

    def _set_custom_coords(self, lon_min, lon_max, lat_min, lat_max):
        """Set custom coordinates from preset."""
        self.custom_lon_min.set(str(lon_min))
        self.custom_lon_max.set(str(lon_max))
        self.custom_lat_min.set(str(lat_min))
        self.custom_lat_max.set(str(lat_max))

    def _select_all_events(self):
        """Select all event types."""
        self.event_severe.set(True)
        self.event_flood.set(True)
        self.event_fire.set(True)

    def _select_no_events(self):
        """Deselect all event types."""
        self.event_severe.set(False)
        self.event_flood.set(False)
        self.event_fire.set(False)

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
            use_demo = self.use_demo_data.get()
            output_dir = self.output_dir.get()

            # Get selected event types
            selected_events = []
            if self.event_severe.get():
                selected_events.append('severe')
            if self.event_flood.get():
                selected_events.append('flood')
            if self.event_fire.get():
                selected_events.append('fire')

            if not selected_events:
                raise ValueError("Please select at least one event type")

            self.root.after(0, lambda: self._update_status("Initializing generator..."))
            self.root.after(0, lambda: self._update_progress(5, "Initializing..."))

            # Create generator
            self.generator = WeatherMapGenerator(output_dir=output_dir)

            # Get region config - handle custom region
            if region == 'custom':
                try:
                    lon_min = float(self.custom_lon_min.get())
                    lon_max = float(self.custom_lon_max.get())
                    lat_min = float(self.custom_lat_min.get())
                    lat_max = float(self.custom_lat_max.get())

                    if lon_min >= lon_max or lat_min >= lat_max:
                        raise ValueError("Invalid coordinates: min must be less than max")

                    region_config = {
                        'name': 'Custom Region',
                        'bounds': [lon_min, lon_max, lat_min, lat_max],
                        'grid_resolution': 1.0,
                        'projection': ccrs.PlateCarree()
                    }
                    # Add custom region to REGIONS temporarily
                    REGIONS['custom'] = region_config
                except ValueError as e:
                    raise ValueError(f"Invalid coordinates: {e}")
            else:
                region_config = REGIONS[region]

            region_name = region_config['name']

            self.root.after(0, lambda: self._update_progress(10, "Fetching weather data..."))
            self.root.after(0, lambda: self._update_status(
                f"{'Generating sample' if use_demo else 'Fetching'} data for {region_name}..."))

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

            # Calculate risks only for selected event types
            severe_risks = {}
            flood_risks = {}
            fire_risks = {}

            for (lat, lon), weather_data in grid_data['data'].items():
                if 'severe' in selected_events:
                    severe_risks[(lat, lon)] = self.generator.calculator.calculate_severe_weather_risk(weather_data)
                if 'flood' in selected_events:
                    flood_risks[(lat, lon)] = self.generator.calculator.calculate_flood_risk(weather_data)
                if 'fire' in selected_events:
                    fire_risks[(lat, lon)] = self.generator.calculator.calculate_fire_risk(weather_data)

            # Generate requested maps
            output_files = {}
            progress_base = 50
            maps_to_generate = []

            if 'severe' in selected_events:
                maps_to_generate.append(('severe', 'severe_weather', severe_risks,
                                         SEVERE_WEATHER_CATEGORIES, 'Severe Weather Risk Outlook'))
            if 'flood' in selected_events:
                maps_to_generate.append(('flood', 'flood_risk', flood_risks,
                                         FLOOD_RISK_CATEGORIES, 'Flooding Risk Outlook'))
            if 'fire' in selected_events:
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
