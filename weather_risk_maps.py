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
import matplotlib
# Use Agg backend for thread-safe rendering (no GUI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import requests
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import os
import sys
import threading
import time
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
# COUNTRY BOUNDING BOXES
# Format: [lon_min, lon_max, lat_min, lat_max]
# =============================================================================

COUNTRIES = {
    # Africa
    'Algeria': [-9.0, 12.0, 19.0, 37.1],
    'Angola': [11.7, 24.1, -18.0, -4.4],
    'Benin': [0.8, 3.9, 6.2, 12.4],
    'Botswana': [20.0, 29.4, -27.0, -17.8],
    'Burkina Faso': [-5.5, 2.4, 9.4, 15.1],
    'Burundi': [29.0, 30.9, -4.5, -2.3],
    'Cameroon': [8.5, 16.2, 1.7, 13.1],
    'Central African Republic': [14.4, 27.5, 2.2, 11.0],
    'Chad': [13.5, 24.0, 7.4, 23.5],
    'Congo': [11.2, 18.6, -5.0, 3.7],
    'DR Congo': [12.2, 31.3, -13.5, 5.4],
    'Djibouti': [41.8, 43.4, 10.9, 12.7],
    'Egypt': [24.7, 36.9, 22.0, 31.7],
    'Equatorial Guinea': [9.3, 11.3, 1.0, 2.3],
    'Eritrea': [36.4, 43.1, 12.4, 18.0],
    'Eswatini': [30.8, 32.1, -27.3, -25.7],
    'Ethiopia': [33.0, 48.0, 3.4, 15.0],
    'Gabon': [8.7, 14.5, -4.0, 2.3],
    'Gambia': [-16.8, -13.8, 13.1, 13.8],
    'Ghana': [-3.3, 1.2, 4.7, 11.2],
    'Guinea': [-15.1, -7.6, 7.2, 12.7],
    'Guinea-Bissau': [-16.7, -13.6, 10.9, 12.7],
    'Ivory Coast': [-8.6, -2.5, 4.4, 10.7],
    'Kenya': [33.9, 42.0, -4.7, 5.5],
    'Lesotho': [27.0, 29.5, -30.7, -28.6],
    'Liberia': [-11.5, -7.4, 4.4, 8.6],
    'Libya': [9.3, 25.2, 19.5, 33.2],
    'Madagascar': [43.2, 50.5, -25.6, -11.9],
    'Malawi': [32.7, 35.9, -17.1, -9.4],
    'Mali': [-12.2, 4.2, 10.2, 25.0],
    'Mauritania': [-17.1, -4.8, 14.7, 27.3],
    'Morocco': [-13.2, -1.0, 27.7, 35.9],
    'Mozambique': [30.2, 40.8, -26.9, -10.5],
    'Namibia': [11.7, 25.3, -29.0, -17.0],
    'Niger': [0.2, 16.0, 11.7, 23.5],
    'Nigeria': [2.7, 14.7, 4.3, 13.9],
    'Rwanda': [29.0, 30.9, -2.8, -1.1],
    'Senegal': [-17.5, -11.4, 12.3, 16.7],
    'Sierra Leone': [-13.3, -10.3, 6.9, 10.0],
    'Somalia': [40.9, 51.4, -1.7, 12.0],
    'South Africa': [16.5, 33.0, -35.0, -22.1],
    'South Sudan': [24.1, 35.9, 3.5, 12.2],
    'Sudan': [21.8, 38.6, 8.7, 22.2],
    'Tanzania': [29.3, 40.4, -11.7, -1.0],
    'Togo': [-0.1, 1.8, 6.1, 11.1],
    'Tunisia': [7.5, 11.6, 30.2, 37.4],
    'Uganda': [29.6, 35.0, -1.5, 4.2],
    'Zambia': [22.0, 33.7, -18.1, -8.2],
    'Zimbabwe': [25.2, 33.1, -22.4, -15.6],

    # Asia
    'Afghanistan': [60.5, 75.0, 29.4, 38.5],
    'Armenia': [43.4, 46.6, 38.8, 41.3],
    'Azerbaijan': [44.8, 50.4, 38.4, 41.9],
    'Bahrain': [50.4, 50.7, 25.8, 26.3],
    'Bangladesh': [88.0, 92.7, 20.7, 26.6],
    'Bhutan': [88.8, 92.1, 26.7, 28.3],
    'Brunei': [114.0, 115.4, 4.0, 5.1],
    'Cambodia': [102.3, 107.6, 10.4, 14.7],
    'China': [73.5, 135.0, 18.2, 53.6],
    'Cyprus': [32.3, 34.6, 34.6, 35.7],
    'Georgia': [40.0, 46.7, 41.1, 43.6],
    'India': [68.2, 97.4, 6.8, 35.5],
    'Indonesia': [95.0, 141.0, -11.0, 6.1],
    'Iran': [44.0, 63.3, 25.1, 39.8],
    'Iraq': [38.8, 48.6, 29.1, 37.4],
    'Israel': [34.3, 35.9, 29.5, 33.3],
    'Japan': [122.9, 153.0, 24.0, 45.5],
    'Jordan': [34.9, 39.3, 29.2, 33.4],
    'Kazakhstan': [46.5, 87.3, 40.6, 55.4],
    'Kuwait': [46.6, 48.4, 28.5, 30.1],
    'Kyrgyzstan': [69.3, 80.3, 39.2, 43.3],
    'Laos': [100.1, 107.7, 13.9, 22.5],
    'Lebanon': [35.1, 36.6, 33.1, 34.7],
    'Malaysia': [99.6, 119.3, 0.9, 7.4],
    'Maldives': [72.7, 73.8, -0.7, 7.1],
    'Mongolia': [87.8, 120.0, 41.6, 52.2],
    'Myanmar': [92.2, 101.2, 9.8, 28.5],
    'Nepal': [80.1, 88.2, 26.4, 30.4],
    'North Korea': [124.3, 130.7, 37.7, 43.0],
    'Oman': [52.0, 59.8, 16.7, 26.4],
    'Pakistan': [60.9, 77.8, 23.7, 37.1],
    'Palestine': [34.2, 35.6, 31.2, 32.6],
    'Philippines': [116.9, 126.6, 4.6, 21.1],
    'Qatar': [50.8, 51.6, 24.5, 26.2],
    'Saudi Arabia': [34.6, 55.7, 16.4, 32.2],
    'Singapore': [103.6, 104.0, 1.2, 1.5],
    'South Korea': [125.1, 129.6, 33.1, 38.6],
    'Sri Lanka': [79.7, 81.9, 5.9, 9.8],
    'Syria': [35.7, 42.4, 32.3, 37.3],
    'Taiwan': [120.0, 122.0, 21.9, 25.3],
    'Tajikistan': [67.4, 75.1, 36.7, 41.0],
    'Thailand': [97.3, 105.6, 5.6, 20.5],
    'Timor-Leste': [124.0, 127.3, -9.5, -8.1],
    'Turkey': [26.0, 45.0, 36.0, 42.1],
    'Turkmenistan': [52.4, 66.7, 35.1, 42.8],
    'United Arab Emirates': [51.6, 56.4, 22.6, 26.1],
    'Uzbekistan': [56.0, 73.1, 37.2, 45.6],
    'Vietnam': [102.1, 109.5, 8.6, 23.4],
    'Yemen': [42.6, 54.5, 12.1, 19.0],

    # Europe
    'Albania': [19.3, 21.1, 39.6, 42.7],
    'Andorra': [1.4, 1.8, 42.4, 42.7],
    'Austria': [9.5, 17.2, 46.4, 49.0],
    'Belarus': [23.2, 32.8, 51.3, 56.2],
    'Belgium': [2.5, 6.4, 49.5, 51.5],
    'Bosnia and Herzegovina': [15.7, 19.6, 42.6, 45.3],
    'Bulgaria': [22.4, 28.6, 41.2, 44.2],
    'Croatia': [13.5, 19.4, 42.4, 46.5],
    'Czech Republic': [12.1, 18.9, 48.6, 51.1],
    'Denmark': [8.1, 15.2, 54.6, 57.8],
    'Estonia': [21.8, 28.2, 57.5, 59.7],
    'Finland': [20.6, 31.6, 59.8, 70.1],
    'France': [-5.1, 9.6, 41.3, 51.1],
    'Germany': [5.9, 15.0, 47.3, 55.1],
    'Greece': [19.4, 29.6, 34.8, 41.7],
    'Hungary': [16.1, 22.9, 45.7, 48.6],
    'Iceland': [-24.5, -13.5, 63.4, 66.5],
    'Ireland': [-10.5, -6.0, 51.4, 55.4],
    'Italy': [6.6, 18.5, 36.6, 47.1],
    'Kosovo': [20.0, 21.8, 41.9, 43.3],
    'Latvia': [21.0, 28.2, 55.7, 58.1],
    'Liechtenstein': [9.5, 9.6, 47.0, 47.3],
    'Lithuania': [21.0, 26.8, 53.9, 56.5],
    'Luxembourg': [5.7, 6.5, 49.4, 50.2],
    'Malta': [14.2, 14.6, 35.8, 36.1],
    'Moldova': [26.6, 30.2, 45.5, 48.5],
    'Monaco': [7.4, 7.4, 43.7, 43.8],
    'Montenegro': [18.5, 20.4, 41.9, 43.6],
    'Netherlands': [3.4, 7.2, 50.8, 53.5],
    'North Macedonia': [20.5, 23.0, 40.9, 42.4],
    'Norway': [4.6, 31.1, 58.0, 71.2],
    'Poland': [14.1, 24.2, 49.0, 54.8],
    'Portugal': [-9.5, -6.2, 36.9, 42.2],
    'Romania': [20.3, 30.0, 43.6, 48.3],
    'Russia': [27.0, 180.0, 41.2, 82.0],
    'San Marino': [12.4, 12.5, 43.9, 44.0],
    'Serbia': [18.8, 23.0, 42.2, 46.2],
    'Slovakia': [16.8, 22.6, 47.7, 49.6],
    'Slovenia': [13.4, 16.6, 45.4, 46.9],
    'Spain': [-9.3, 4.3, 36.0, 43.8],
    'Sweden': [11.1, 24.2, 55.3, 69.1],
    'Switzerland': [6.0, 10.5, 45.8, 47.8],
    'Ukraine': [22.1, 40.2, 44.4, 52.4],
    'United Kingdom': [-8.6, 1.8, 49.9, 60.9],
    'Vatican City': [12.4, 12.5, 41.9, 41.9],

    # North America
    'Bahamas': [-80.5, -72.7, 20.9, 27.3],
    'Barbados': [-59.7, -59.4, 13.0, 13.3],
    'Belize': [-89.2, -87.5, 15.9, 18.5],
    'Canada': [-141.0, -52.6, 41.7, 83.1],
    'Costa Rica': [-86.0, -82.6, 8.0, 11.2],
    'Cuba': [-85.0, -74.1, 19.8, 23.3],
    'Dominican Republic': [-72.0, -68.3, 17.5, 19.9],
    'El Salvador': [-90.1, -87.7, 13.2, 14.4],
    'Guatemala': [-92.2, -88.2, 13.7, 17.8],
    'Haiti': [-74.5, -71.6, 18.0, 20.1],
    'Honduras': [-89.4, -83.1, 13.0, 16.5],
    'Jamaica': [-78.4, -76.2, 17.7, 18.5],
    'Mexico': [-117.1, -86.7, 14.5, 32.7],
    'Nicaragua': [-87.7, -83.1, 10.7, 15.0],
    'Panama': [-83.1, -77.2, 7.2, 9.6],
    'Puerto Rico': [-67.3, -65.2, 17.9, 18.5],
    'Trinidad and Tobago': [-61.9, -60.5, 10.0, 10.9],
    'United States': [-125.0, -66.0, 24.0, 50.0],

    # South America
    'Argentina': [-73.6, -53.6, -55.1, -21.8],
    'Bolivia': [-69.6, -57.5, -22.9, -9.7],
    'Brazil': [-73.9, -34.8, -33.8, 5.3],
    'Chile': [-75.6, -66.4, -55.9, -17.5],
    'Colombia': [-79.0, -66.9, -4.2, 12.5],
    'Ecuador': [-81.1, -75.2, -5.0, 1.5],
    'Guyana': [-61.4, -56.5, 1.2, 8.6],
    'Paraguay': [-62.6, -54.3, -27.6, -19.3],
    'Peru': [-81.4, -68.7, -18.4, -0.0],
    'Suriname': [-58.1, -54.0, 1.8, 6.0],
    'Uruguay': [-58.4, -53.1, -35.0, -30.1],
    'Venezuela': [-73.4, -59.8, 0.6, 12.2],

    # Oceania
    'Australia': [112.0, 154.0, -44.0, -10.0],
    'Fiji': [177.0, -179.0, -21.0, -12.5],
    'New Zealand': [166.4, 178.6, -47.3, -34.4],
    'Papua New Guinea': [141.0, 156.0, -11.7, -1.4],
    'Solomon Islands': [155.5, 170.2, -12.3, -5.0],
    'Vanuatu': [166.5, 170.2, -20.3, -13.1],
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
        Uses only core parameters that are guaranteed to be available.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dictionary with weather data or None if request fails
        """
        # Core hourly parameters - these should always be available
        hourly_params = [
            'temperature_2m',
            'relative_humidity_2m',
            'dew_point_2m',
            'precipitation_probability',
            'precipitation',
            'rain',
            'weather_code',
            'pressure_msl',
            'cloud_cover',
            'wind_speed_10m',
            'wind_direction_10m',
            'wind_gusts_10m',
        ]

        # Core daily parameters
        daily_params = [
            'weather_code',
            'temperature_2m_max',
            'temperature_2m_min',
            'precipitation_sum',
            'rain_sum',
            'precipitation_hours',
            'precipitation_probability_max',
            'wind_speed_10m_max',
            'wind_gusts_10m_max',
        ]

        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ','.join(hourly_params),
            'daily': ','.join(daily_params),
            'timezone': 'UTC',
            'forecast_days': 3
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)

            # Check for errors
            if response.status_code == 429:
                print(f"Rate limited at ({lat}, {lon}), waiting...")
                return None

            if response.status_code != 200:
                print(f"API error {response.status_code} for ({lat}, {lon}): {response.text[:200]}")
                return None

            data = response.json()

            # Check if we got error in response
            if 'error' in data:
                print(f"API returned error for ({lat}, {lon}): {data.get('reason', 'Unknown')}")
                return None

            # Normalize and add derived values
            if 'hourly' in data:
                hourly = data['hourly']
                num_hours = len(hourly.get('temperature_2m', [])) or 72

                # Map dew_point_2m to dewpoint_2m
                if 'dew_point_2m' in hourly:
                    hourly['dewpoint_2m'] = hourly['dew_point_2m']
                else:
                    # Estimate dewpoint from temp and humidity
                    temps = hourly.get('temperature_2m', [20] * num_hours)
                    rh = hourly.get('relative_humidity_2m', [50] * num_hours)
                    hourly['dewpoint_2m'] = [
                        t - ((100 - r) / 5) if t and r else 10
                        for t, r in zip(temps, rh)
                    ]

                # Add wind_speed_80m estimate (typically 1.2-1.4x surface wind)
                if 'wind_speed_10m' in hourly:
                    hourly['wind_speed_80m'] = [
                        w * 1.3 if w is not None else 15
                        for w in hourly['wind_speed_10m']
                    ]
                else:
                    hourly['wind_speed_80m'] = [15] * num_hours

                # Add CAPE estimate based on temperature and humidity
                temps = hourly.get('temperature_2m', [20] * num_hours)
                rh = hourly.get('relative_humidity_2m', [50] * num_hours)
                precip_prob = hourly.get('precipitation_probability', [0] * num_hours)

                # Estimate CAPE: higher temp + humidity + precip probability = higher CAPE
                hourly['cape'] = []
                for t, r, p in zip(temps, rh, precip_prob):
                    if t is None or r is None:
                        hourly['cape'].append(0)
                    else:
                        # Simple CAPE estimation
                        cape_estimate = max(0, (t - 15) * 50 + (r - 40) * 20 + (p or 0) * 10)
                        hourly['cape'].append(min(cape_estimate, 4000))

                # Estimate lifted index from CAPE
                hourly['lifted_index'] = [
                    max(-10, 2 - (c / 500)) if c else 2
                    for c in hourly['cape']
                ]

                # CIN estimate
                hourly['convective_inhibition'] = [-50] * num_hours

                # Add soil moisture estimates based on recent precipitation
                precip = hourly.get('precipitation', [0] * num_hours)
                recent_precip = sum([p for p in precip[:24] if p]) if precip else 0
                base_moisture = min(0.45, 0.25 + recent_precip * 0.01)

                hourly['soil_moisture_0_to_1cm'] = [base_moisture] * num_hours
                hourly['soil_moisture_1_to_3cm'] = [base_moisture * 1.05] * num_hours
                hourly['soil_moisture_3_to_9cm'] = [base_moisture * 1.1] * num_hours
                hourly['soil_moisture_9_to_27cm'] = [base_moisture * 1.15] * num_hours

                # Add evapotranspiration estimate
                hourly['evapotranspiration'] = [0.2] * num_hours

            # Add derived daily values if missing
            if 'daily' in data:
                daily = data['daily']
                num_days = len(daily.get('temperature_2m_max', [])) or 3

                if 'et0_fao_evapotranspiration' not in daily:
                    daily['et0_fao_evapotranspiration'] = [5.0] * num_days

            return data

        except requests.exceptions.Timeout:
            print(f"Timeout fetching data for ({lat}, {lon})")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for ({lat}, {lon}): {e}")
            return None
        except requests.RequestException as e:
            print(f"Error fetching data for ({lat}, {lon}): {e}")
            return None
        except Exception as e:
            print(f"Unexpected error for ({lat}, {lon}): {e}")
            return None

    def fetch_grid_data(self, bounds: List[float], resolution: float) -> Dict:
        """
        Fetch weather data for a grid of points with rate limiting.

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
        print(f"Note: Rate limiting applied - ~0.5s between requests to avoid API throttling")

        count = 0
        consecutive_failures = 0
        base_delay = 0.5  # 500ms between requests (2 req/sec - conservative)

        for lat in lats:
            for lon in lons:
                # Rate limiting - wait between requests
                if count > 0:
                    # Increase delay if we're having failures
                    current_delay = base_delay * (1 + consecutive_failures * 0.5)
                    time.sleep(min(current_delay, 3.0))

                # Retry logic with exponential backoff
                max_retries = 4
                success = False

                for attempt in range(max_retries):
                    data = self.fetch_weather_data(lat, lon)
                    if data:
                        grid_data['data'][(lat, lon)] = data
                        consecutive_failures = 0
                        success = True
                        break
                    else:
                        consecutive_failures += 1
                        if attempt < max_retries - 1:
                            # Exponential backoff on failure
                            wait_time = 2.0 * (2 ** attempt)  # 2s, 4s, 8s
                            print(f"  Retry {attempt + 1}/{max_retries} for ({lat:.1f}, {lon:.1f}) after {wait_time:.1f}s...")
                            time.sleep(wait_time)

                if not success:
                    print(f"  Failed to get data for ({lat:.1f}, {lon:.1f}) after {max_retries} attempts")

                count += 1
                if count % 5 == 0:
                    success_rate = len(grid_data['data']) / count * 100
                    print(f"  Progress: {count}/{total_points} points ({success_rate:.0f}% success)")

        success_count = len(grid_data['data'])
        print(f"Completed: {success_count}/{total_points} points fetched successfully ({success_count/total_points*100:.0f}%)")

        if success_count == 0:
            print("WARNING: No data was fetched! Check your internet connection.")

        return grid_data


# =============================================================================
# RISK CALCULATION ALGORITHMS
# =============================================================================

class RiskCalculator:
    """Calculate weather risk indices from meteorological data."""

    @staticmethod
    def _safe_max(values, default=0):
        """Safely get max from list, handling None values and invalid types."""
        if not values:
            return default
        filtered = []
        for v in values:
            if v is not None:
                try:
                    filtered.append(float(v))
                except (ValueError, TypeError):
                    pass
        return max(filtered) if filtered else default

    @staticmethod
    def _safe_min(values, default=0):
        """Safely get min from list, handling None values and invalid types."""
        if not values:
            return default
        filtered = []
        for v in values:
            if v is not None:
                try:
                    filtered.append(float(v))
                except (ValueError, TypeError):
                    pass
        return min(filtered) if filtered else default

    @staticmethod
    def _safe_mean(values, default=0):
        """Safely get mean from list, handling None values and invalid types."""
        if not values:
            return default
        filtered = []
        for v in values:
            if v is not None:
                try:
                    filtered.append(float(v))
                except (ValueError, TypeError):
                    pass
        return sum(filtered) / len(filtered) if filtered else default

    @staticmethod
    def _safe_sum(values, default=0):
        """Safely get sum from list, handling None values and invalid types."""
        if not values:
            return default
        total = 0
        for v in values:
            if v is not None:
                try:
                    total += float(v)
                except (ValueError, TypeError):
                    pass
        return total

    @staticmethod
    def _get_values(data_dict, key, length=24, default_value=0):
        """Safely extract values from a dictionary with proper defaults."""
        values = data_dict.get(key, [])
        if not values:
            return [default_value] * length
        # Ensure we have enough values
        result = []
        for i in range(length):
            if i < len(values) and values[i] is not None:
                try:
                    result.append(float(values[i]))
                except (ValueError, TypeError):
                    result.append(default_value)
            else:
                result.append(default_value)
        return result

    @staticmethod
    def calculate_severe_weather_risk(data: Dict) -> int:
        """
        Calculate severe weather risk (0-5) based on convective parameters.

        Uses advanced meteorological indices:
        - CAPE (Convective Available Potential Energy)
        - Lifted Index (atmospheric stability)
        - CIN (Convective Inhibition)
        - Wind shear (10m vs 80m winds)
        - Wind gusts
        - Precipitation intensity
        - Weather codes for active thunderstorms

        Categories:
        0 - None: No severe weather expected
        1 - Marginal: Isolated severe storms possible
        2 - Slight: Scattered severe storms possible
        3 - Enhanced: Numerous severe storms likely
        4 - Moderate: Widespread severe storms expected
        5 - High: Major severe weather outbreak expected
        """
        if not data:
            return 0

        hourly = data.get('hourly', {})
        if not hourly:
            return 0

        # Get values for next 24 hours using safe extraction
        cape_values = RiskCalculator._get_values(hourly, 'cape', 24, 0)
        li_values = RiskCalculator._get_values(hourly, 'lifted_index', 24, 2)
        cin_values = RiskCalculator._get_values(hourly, 'convective_inhibition', 24, 0)
        wind_10m = RiskCalculator._get_values(hourly, 'wind_speed_10m', 24, 10)
        wind_80m = RiskCalculator._get_values(hourly, 'wind_speed_80m', 24, 15)
        wind_gust_values = RiskCalculator._get_values(hourly, 'wind_gusts_10m', 24, 15)
        precip_values = RiskCalculator._get_values(hourly, 'precipitation', 24, 0)
        weather_codes = RiskCalculator._get_values(hourly, 'weather_code', 24, 0)
        precip_prob = RiskCalculator._get_values(hourly, 'precipitation_probability', 24, 0)

        # Calculate key parameters
        cape_max = RiskCalculator._safe_max(cape_values, 0)
        li_min = RiskCalculator._safe_min(li_values, 10)  # Lower LI = more unstable
        cin_min = RiskCalculator._safe_min(cin_values, 0)  # Less negative CIN = easier triggering
        gust_max = RiskCalculator._safe_max(wind_gust_values, 0)
        precip_max = RiskCalculator._safe_max(precip_values, 0)
        precip_prob_max = RiskCalculator._safe_max(precip_prob, 0)

        # Calculate wind shear (difference between 80m and 10m winds)
        shear_values = []
        for w10, w80 in zip(wind_10m, wind_80m):
            if w10 is not None and w80 is not None:
                shear_values.append(abs(w80 - w10))
        wind_shear = max(shear_values) if shear_values else 0

        # Check for severe weather codes
        # 95: Slight thunderstorm, 96: Thunderstorm with hail, 99: Heavy thunderstorm with hail
        has_thunderstorm = any(95 <= (c or 0) <= 99 for c in weather_codes)
        has_severe_ts = any((c or 0) in [96, 99] for c in weather_codes)

        # === CAPE Score (0-5) ===
        # Based on SPC thresholds
        cape_score = 0
        if cape_max >= 4000:
            cape_score = 5
        elif cape_max >= 2500:
            cape_score = 4
        elif cape_max >= 1500:
            cape_score = 3
        elif cape_max >= 1000:
            cape_score = 2
        elif cape_max >= 500:
            cape_score = 1

        # === Lifted Index Score (0-5) ===
        # Negative LI indicates instability
        li_score = 0
        if li_min <= -8:
            li_score = 5
        elif li_min <= -6:
            li_score = 4
        elif li_min <= -4:
            li_score = 3
        elif li_min <= -2:
            li_score = 2
        elif li_min <= 0:
            li_score = 1

        # === Wind Shear Score (0-5) ===
        shear_score = 0
        if wind_shear >= 25:  # Strong shear favors supercells
            shear_score = 5
        elif wind_shear >= 20:
            shear_score = 4
        elif wind_shear >= 15:
            shear_score = 3
        elif wind_shear >= 10:
            shear_score = 2
        elif wind_shear >= 5:
            shear_score = 1

        # === Wind Gust Score (0-5) ===
        gust_score = 0
        if gust_max >= 120:  # Hurricane force
            gust_score = 5
        elif gust_max >= 90:  # Severe
            gust_score = 4
        elif gust_max >= 70:
            gust_score = 3
        elif gust_max >= 50:
            gust_score = 2
        elif gust_max >= 35:
            gust_score = 1

        # === Precipitation Intensity Score (0-4) ===
        precip_score = 0
        if precip_max >= 40:
            precip_score = 4
        elif precip_max >= 20:
            precip_score = 3
        elif precip_max >= 10:
            precip_score = 2
        elif precip_max >= 3:
            precip_score = 1

        # === CIN Modifier ===
        # CIN between -50 and 0 is ideal for severe weather (cap that can break)
        cin_modifier = 1.0
        if -100 <= cin_min <= -25:
            cin_modifier = 1.15  # "Loaded gun" scenario
        elif cin_min > -25:
            cin_modifier = 1.0  # Too weak cap
        elif cin_min < -200:
            cin_modifier = 0.7  # Cap too strong

        # Combine scores with meteorological weighting
        combined_score = (
            cape_score * 0.30 +
            li_score * 0.20 +
            shear_score * 0.20 +
            gust_score * 0.20 +
            precip_score * 0.10
        ) * cin_modifier

        # Bonus for active thunderstorms
        if has_severe_ts:
            combined_score += 1.0
        elif has_thunderstorm:
            combined_score += 0.5

        # High precipitation probability increases confidence
        if precip_prob_max >= 80:
            combined_score *= 1.1

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
        Calculate flooding risk (0-4) based on precipitation and soil conditions.

        Uses advanced hydrological factors:
        - Precipitation totals (daily and 3-day)
        - Precipitation intensity (hourly rates)
        - Precipitation duration
        - Soil moisture at multiple depths
        - Precipitation probability
        - Antecedent conditions

        Categories:
        0 - None: No flooding expected
        1 - Marginal: Minor flooding possible
        2 - Moderate: Moderate flooding likely
        3 - High: Significant flooding expected
        4 - Extreme: Major/flash flooding likely
        """
        if not data:
            return 0

        daily = data.get('daily', {})
        hourly = data.get('hourly', {})

        if not daily and not hourly:
            return 0

        # Get precipitation data using safe extraction
        precip_sum = RiskCalculator._get_values(daily, 'precipitation_sum', 3, 0)
        precip_hours = RiskCalculator._get_values(daily, 'precipitation_hours', 3, 0)
        precip_prob = RiskCalculator._get_values(daily, 'precipitation_probability_max', 3, 0)

        # Get hourly data
        hourly_precip = RiskCalculator._get_values(hourly, 'precipitation', 72, 0)

        # Soil moisture at multiple depths
        soil_0_1 = RiskCalculator._get_values(hourly, 'soil_moisture_0_to_1cm', 24, 0.3)
        soil_1_3 = RiskCalculator._get_values(hourly, 'soil_moisture_1_to_3cm', 24, 0.3)
        soil_3_9 = RiskCalculator._get_values(hourly, 'soil_moisture_3_to_9cm', 24, 0.3)
        soil_9_27 = RiskCalculator._get_values(hourly, 'soil_moisture_9_to_27cm', 24, 0.3)

        # Calculate key metrics
        total_precip = RiskCalculator._safe_sum(precip_sum, 0)
        max_daily_precip = RiskCalculator._safe_max(precip_sum, 0)
        total_hours = RiskCalculator._safe_sum(precip_hours, 0)
        max_hourly = RiskCalculator._safe_max(hourly_precip, 0)
        precip_prob_max = RiskCalculator._safe_max(precip_prob, 0)

        # Calculate average soil moisture across depths
        avg_soil_surface = RiskCalculator._safe_mean(soil_0_1, 0.3)
        avg_soil_shallow = RiskCalculator._safe_mean(soil_1_3, 0.3)
        avg_soil_mid = RiskCalculator._safe_mean(soil_3_9, 0.3)
        avg_soil_deep = RiskCalculator._safe_mean(soil_9_27, 0.3)

        # Weighted soil moisture (surface matters most for runoff)
        weighted_soil = (avg_soil_surface * 0.4 + avg_soil_shallow * 0.3 +
                         avg_soil_mid * 0.2 + avg_soil_deep * 0.1)

        # Calculate 6-hour rainfall accumulations for flash flood potential
        six_hour_totals = []
        for i in range(0, min(48, len(hourly_precip)), 6):
            chunk = hourly_precip[i:i+6]
            total = sum([v for v in chunk if v is not None] or [0])
            six_hour_totals.append(total)
        max_6hr = max(six_hour_totals) if six_hour_totals else 0

        # === Precipitation Volume Score (0-4) ===
        precip_score = 0
        if total_precip >= 125:
            precip_score = 4
        elif total_precip >= 75:
            precip_score = 3
        elif total_precip >= 40:
            precip_score = 2
        elif total_precip >= 20:
            precip_score = 1

        # === Flash Flood Intensity Score (0-4) ===
        # Based on hourly and 6-hour rates
        intensity_score = 0
        if max_hourly >= 40 or max_6hr >= 75:
            intensity_score = 4
        elif max_hourly >= 25 or max_6hr >= 50:
            intensity_score = 3
        elif max_hourly >= 12 or max_6hr >= 30:
            intensity_score = 2
        elif max_hourly >= 5 or max_6hr >= 15:
            intensity_score = 1

        # === Duration Score (0-3) ===
        # Prolonged rain increases flood risk
        duration_score = 0
        if total_hours >= 36:
            duration_score = 3
        elif total_hours >= 24:
            duration_score = 2
        elif total_hours >= 12:
            duration_score = 1

        # === Soil Saturation Modifier ===
        # Saturated soils = more runoff
        saturation_modifier = 1.0
        if weighted_soil >= 0.45:
            saturation_modifier = 1.5  # Near saturated
        elif weighted_soil >= 0.38:
            saturation_modifier = 1.3
        elif weighted_soil >= 0.32:
            saturation_modifier = 1.15
        elif weighted_soil <= 0.15:
            saturation_modifier = 0.7  # Very dry, high infiltration

        # Combined score
        combined_score = (
            precip_score * 0.40 +
            intensity_score * 0.35 +
            duration_score * 0.25
        ) * saturation_modifier

        # Probability confidence boost
        if precip_prob_max >= 90:
            combined_score *= 1.1
        elif precip_prob_max < 40:
            combined_score *= 0.8

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
        Calculate fire weather risk (0-4) based on the Fosberg Fire Weather Index approach.

        Uses comprehensive fire weather parameters:
        - Temperature (affects fuel moisture)
        - Relative humidity (critical for fire spread)
        - Dewpoint depression (temperature - dewpoint)
        - Wind speed and gusts (fire spread rate)
        - Evapotranspiration (drying potential)
        - Soil moisture (fuel moisture proxy)
        - Recent precipitation (wetting factor)
        - Visibility (smoke/haze indicator)

        Categories:
        0 - None: No significant fire weather
        1 - Elevated: Elevated fire weather conditions
        2 - Critical: Critical fire weather conditions
        3 - Extreme: Extreme fire weather conditions
        4 - Exceptional: Exceptionally dangerous fire weather
        """
        if not data:
            return 0

        hourly = data.get('hourly', {})
        daily = data.get('daily', {})

        if not hourly:
            return 0

        # Get hourly parameters using safe extraction
        temp_values = RiskCalculator._get_values(hourly, 'temperature_2m', 24, 20)
        dewpoint_values = RiskCalculator._get_values(hourly, 'dewpoint_2m', 24, 10)
        rh_values = RiskCalculator._get_values(hourly, 'relative_humidity_2m', 24, 50)
        wind_10m = RiskCalculator._get_values(hourly, 'wind_speed_10m', 24, 10)
        wind_80m = RiskCalculator._get_values(hourly, 'wind_speed_80m', 24, 15)
        gust_values = RiskCalculator._get_values(hourly, 'wind_gusts_10m', 24, 15)

        # Soil moisture at multiple depths
        soil_0_1 = RiskCalculator._get_values(hourly, 'soil_moisture_0_to_1cm', 24, 0.3)
        soil_1_3 = RiskCalculator._get_values(hourly, 'soil_moisture_1_to_3cm', 24, 0.3)

        # Daily parameters
        precip_sum = RiskCalculator._get_values(daily, 'precipitation_sum', 3, 0)
        et0_values = RiskCalculator._get_values(daily, 'et0_fao_evapotranspiration', 3, 5)

        # Calculate key metrics
        max_temp = RiskCalculator._safe_max(temp_values, 20)
        min_rh = RiskCalculator._safe_min(rh_values, 50)
        mean_rh = RiskCalculator._safe_mean(rh_values, 50)
        max_wind = RiskCalculator._safe_max(wind_10m, 10)
        max_gust = RiskCalculator._safe_max(gust_values, 15)
        total_precip = RiskCalculator._safe_sum(precip_sum, 0)
        total_et0 = RiskCalculator._safe_sum(et0_values, 0)

        # Calculate dewpoint depression (indicates how dry air is)
        dewpoint_depressions = []
        for t, d in zip(temp_values, dewpoint_values):
            if t is not None and d is not None:
                dewpoint_depressions.append(t - d)
        max_dewpoint_depression = max(dewpoint_depressions) if dewpoint_depressions else 10

        # Surface soil moisture (most relevant for fine fuels)
        avg_surface_soil = RiskCalculator._safe_mean(soil_0_1, 0.3)
        avg_shallow_soil = RiskCalculator._safe_mean(soil_1_3, 0.3)
        fuel_moisture_proxy = (avg_surface_soil * 0.7 + avg_shallow_soil * 0.3)

        # Calculate wind transport factor (80m winds indicate mixing potential)
        wind_transport = RiskCalculator._safe_max(wind_80m, 15)

        # === Temperature Score (0-4) ===
        temp_score = 0
        if max_temp >= 42:
            temp_score = 4
        elif max_temp >= 38:
            temp_score = 3
        elif max_temp >= 32:
            temp_score = 2
        elif max_temp >= 27:
            temp_score = 1

        # === Relative Humidity Score (0-4) ===
        # Low RH is critical for fire weather
        rh_score = 0
        if min_rh <= 8:
            rh_score = 4
        elif min_rh <= 12:
            rh_score = 3
        elif min_rh <= 20:
            rh_score = 2
        elif min_rh <= 30:
            rh_score = 1

        # === Dewpoint Depression Score (0-4) ===
        # Large depression = very dry air mass
        dd_score = 0
        if max_dewpoint_depression >= 25:
            dd_score = 4
        elif max_dewpoint_depression >= 20:
            dd_score = 3
        elif max_dewpoint_depression >= 15:
            dd_score = 2
        elif max_dewpoint_depression >= 10:
            dd_score = 1

        # === Wind Score (0-4) ===
        # Use combination of sustained and gusts
        effective_wind = max_wind * 0.6 + max_gust * 0.4
        wind_score = 0
        if effective_wind >= 70:
            wind_score = 4
        elif effective_wind >= 50:
            wind_score = 3
        elif effective_wind >= 35:
            wind_score = 2
        elif effective_wind >= 20:
            wind_score = 1

        # === Mixing/Transport Score (0-3) ===
        # Strong upper winds bring dry air down
        transport_score = 0
        if wind_transport >= 50:
            transport_score = 3
        elif wind_transport >= 35:
            transport_score = 2
        elif wind_transport >= 25:
            transport_score = 1

        # === Drought/Fuel Moisture Modifier ===
        drought_modifier = 1.0
        if fuel_moisture_proxy <= 0.10:
            drought_modifier = 1.6  # Extremely dry fuels
        elif fuel_moisture_proxy <= 0.15:
            drought_modifier = 1.4
        elif fuel_moisture_proxy <= 0.20:
            drought_modifier = 1.2
        elif fuel_moisture_proxy >= 0.40:
            drought_modifier = 0.6  # Moist conditions

        # === Recent Precipitation Modifier ===
        precip_modifier = 1.0
        if total_precip >= 25:
            precip_modifier = 0.4  # Significant wetting
        elif total_precip >= 10:
            precip_modifier = 0.6
        elif total_precip >= 5:
            precip_modifier = 0.8
        elif total_precip < 1:
            precip_modifier = 1.1  # No wetting

        # === Evapotranspiration Bonus ===
        # High ET = strong drying conditions
        et_bonus = 0
        if total_et0 >= 20:
            et_bonus = 0.5
        elif total_et0 >= 15:
            et_bonus = 0.3
        elif total_et0 >= 10:
            et_bonus = 0.15

        # Combined score using fire weather index approach
        combined_score = (
            temp_score * 0.15 +
            rh_score * 0.30 +
            dd_score * 0.10 +
            wind_score * 0.30 +
            transport_score * 0.15 +
            et_bonus
        ) * drought_modifier * precip_modifier

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

        # Add Kaldock watermark
        plt.figtext(0.5, 0.5, 'Kaldock',
                    ha='center', va='center',
                    fontsize=72, fontweight='bold',
                    color='gray', alpha=0.15,
                    rotation=30,
                    transform=fig.transFigure,
                    zorder=1)

        # Add Kaldock branding with GitHub URL in corner
        plt.figtext(0.01, 0.01, 'Created by Kaldock',
                    ha='left', fontsize=9, fontweight='bold',
                    color='#333333', alpha=0.8)
        plt.figtext(0.01, 0.035, 'https://github.com/Inasjackw321/Custome-weather-maps',
                    ha='left', fontsize=7,
                    color='#555555', alpha=0.7)

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
        """Generate comprehensive sample/demo data for testing without API calls."""
        lon_min, lon_max, lat_min, lat_max = bounds

        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lats = np.arange(lat_min, lat_max + resolution, resolution)

        grid_data = {
            'lons': lons,
            'lats': lats,
            'data': {}
        }

        # Generate synthetic weather data with realistic patterns
        # Use time-based seed for variety, but keep some spatial coherence
        np.random.seed(int(abs(lon_min * 100 + lat_min * 10)) % 10000)

        for lat in lats:
            for lon in lons:
                # Create synthetic but realistic-looking weather patterns
                lat_factor = (lat - lat_min) / (lat_max - lat_min) if lat_max != lat_min else 0.5
                lon_factor = (lon - lon_min) / (lon_max - lon_min) if lon_max != lon_min else 0.5

                # Add some spatial correlation and randomness
                noise = np.random.random()
                noise2 = np.random.random()

                # Create multiple storm cells across the region
                pattern = np.sin(lat_factor * np.pi * 2) * np.cos(lon_factor * np.pi * 3)
                storm_pattern = max(0, np.sin((lat_factor * 2 + lon_factor) * np.pi * 2) + 0.3 * noise)

                # Create distinct "hot spots" for severe weather
                center_dist = np.sqrt((lat_factor - 0.5)**2 + (lon_factor - 0.5)**2)
                storm_cell1 = max(0, 1 - 3 * np.sqrt((lat_factor - 0.3)**2 + (lon_factor - 0.4)**2))
                storm_cell2 = max(0, 1 - 3 * np.sqrt((lat_factor - 0.7)**2 + (lon_factor - 0.6)**2))
                storm_cell3 = max(0, 1 - 4 * np.sqrt((lat_factor - 0.5)**2 + (lon_factor - 0.8)**2))
                combined_storms = max(storm_cell1, storm_cell2, storm_cell3) * (0.7 + 0.3 * noise)

                # Temperature based on latitude (warmer in south for northern hemisphere)
                base_temp = 20 + 20 * (1 - lat_factor) + 8 * noise
                dewpoint = base_temp - 3 - 10 * noise2

                # Humidity higher in storm areas
                base_rh = max(20, 60 + 35 * combined_storms - 20 * noise)

                # Wind patterns - stronger in storm areas
                wind_10m = 10 + 40 * combined_storms * noise + 15 * abs(pattern)
                wind_80m = wind_10m * (1.3 + 0.4 * noise2)
                wind_gust = wind_10m * (1.6 + 0.6 * combined_storms)

                # Precipitation pattern - much higher in storm zones
                precip = max(0, 5 + 25 * combined_storms + 10 * storm_pattern * noise)
                precip_prob = min(100, max(0, 20 + 80 * combined_storms))

                # Soil moisture (higher where precipitation occurs)
                soil_base = 0.25 + 0.25 * combined_storms + 0.1 * noise

                # CAPE and instability - much higher in storm zones
                cape = max(0, 200 + 4000 * combined_storms * (0.5 + 0.5 * noise))
                lifted_index = 4 - 12 * combined_storms * (0.5 + 0.5 * noise)
                cin = -30 - 80 * noise2

                grid_data['data'][(lat, lon)] = {
                    'hourly': {
                        # Temperature and moisture
                        'temperature_2m': [base_temp + 3 * np.sin(h * np.pi / 12) for h in range(24)],
                        'dewpoint_2m': [dewpoint + 1 * np.sin(h * np.pi / 12) for h in range(24)],
                        'relative_humidity_2m': [max(10, base_rh - 10 * np.sin(h * np.pi / 12)) for h in range(24)],
                        'apparent_temperature': [base_temp + 2 + 3 * np.sin(h * np.pi / 12) for h in range(24)],

                        # Precipitation
                        'precipitation': [max(0, precip * (0.5 + 0.5 * np.random.random())) for _ in range(24)],
                        'rain': [max(0, precip * 0.8 * (0.5 + 0.5 * np.random.random())) for _ in range(24)],
                        'showers': [max(0, precip * 0.2 * np.random.random()) for _ in range(24)],
                        'precipitation_probability': [precip_prob] * 24,

                        # Wind
                        'wind_speed_10m': [wind_10m * (0.8 + 0.4 * np.random.random()) for _ in range(24)],
                        'wind_speed_80m': [wind_80m * (0.8 + 0.4 * np.random.random()) for _ in range(24)],
                        'wind_gusts_10m': [wind_gust * (0.8 + 0.4 * np.random.random()) for _ in range(24)],
                        'wind_direction_10m': [int(180 + 90 * np.sin(h * np.pi / 6)) for h in range(24)],

                        # Convective parameters
                        'cape': [cape * (0.8 + 0.4 * np.random.random()) for _ in range(24)],
                        'lifted_index': [lifted_index + 2 * np.random.random() for _ in range(24)],
                        'convective_inhibition': [cin * (0.8 + 0.4 * np.random.random()) for _ in range(24)],

                        # Weather codes (thunderstorm codes 95-99)
                        'weather_code': [95 if storm_pattern > 0.5 and np.random.random() > 0.7 else int(3 * noise) for _ in range(24)],

                        # Pressure and visibility
                        'pressure_msl': [1013 + 10 * pattern] * 24,
                        'surface_pressure': [1010 + 10 * pattern] * 24,
                        'visibility': [max(1000, 20000 - 15000 * precip / 10)] * 24,
                        'cloud_cover': [min(100, max(0, 20 + 60 * storm_pattern + 20 * noise))] * 24,

                        # Soil parameters
                        'soil_moisture_0_to_1cm': [max(0.05, soil_base * (0.8 + 0.4 * np.random.random()))] * 24,
                        'soil_moisture_1_to_3cm': [max(0.05, soil_base * 1.05 * (0.9 + 0.2 * np.random.random()))] * 24,
                        'soil_moisture_3_to_9cm': [max(0.05, soil_base * 1.1 * (0.9 + 0.2 * np.random.random()))] * 24,
                        'soil_moisture_9_to_27cm': [max(0.05, soil_base * 1.15 * (0.9 + 0.2 * np.random.random()))] * 24,
                        'soil_temperature_0cm': [base_temp - 2 + 5 * np.sin(h * np.pi / 12) for h in range(24)],

                        # Evapotranspiration
                        'evapotranspiration': [max(0, 0.2 + 0.3 * lat_factor * (0.5 + 0.5 * np.random.random()))] * 24,
                        'freezing_level_height': [max(0, 3000 + 1500 * lat_factor)] * 24,
                        'snowfall': [0] * 24,
                        'snow_depth': [0] * 24,
                    },
                    'daily': {
                        'weather_code': [95 if combined_storms > 0.4 else (80 if precip > 10 else int(3 * noise))] * 3,
                        'temperature_2m_max': [base_temp + 8] * 3,
                        'temperature_2m_min': [base_temp - 5] * 3,
                        'apparent_temperature_max': [base_temp + 10] * 3,
                        'apparent_temperature_min': [base_temp - 7] * 3,
                        'precipitation_sum': [max(0, precip * 18 * (0.6 + 0.4 * np.random.random())) for _ in range(3)],
                        'rain_sum': [max(0, precip * 15 * (0.6 + 0.4 * np.random.random())) for _ in range(3)],
                        'showers_sum': [max(0, precip * 3 * np.random.random()) for _ in range(3)],
                        'snowfall_sum': [0] * 3,
                        'precipitation_hours': [int(4 + 18 * combined_storms)] * 3,
                        'precipitation_probability_max': [precip_prob] * 3,
                        'wind_speed_10m_max': [wind_gust * 0.9] * 3,
                        'wind_gusts_10m_max': [wind_gust * 1.2] * 3,
                        'wind_direction_10m_dominant': [int(180 + 90 * pattern)] * 3,
                        'et0_fao_evapotranspiration': [max(0, 3 + 5 * lat_factor * noise)] * 3,
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

        # Preview image references (keep references to avoid garbage collection)
        self.preview_pil_image = None
        self.preview_tk_image = None
        self.preview_image_label = None

        # Event type checkboxes
        self.event_severe = tk.BooleanVar(value=True)
        self.event_flood = tk.BooleanVar(value=True)
        self.event_fire = tk.BooleanVar(value=True)

        # Custom region coordinates
        self.custom_lat_min = tk.StringVar(value='25.0')
        self.custom_lat_max = tk.StringVar(value='50.0')
        self.custom_lon_min = tk.StringVar(value='-125.0')
        self.custom_lon_max = tk.StringVar(value='-65.0')

        # Selected country
        self.selected_country = None

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
            ('country', 'Select Country', 'Choose from list'),
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

        # Country selector frame (hidden by default)
        self.country_frame = ttk.Frame(step1_frame)

        ttk.Separator(self.country_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        country_label = ttk.Label(self.country_frame, text="Select a country:",
                                   font=('Helvetica', 9, 'italic'))
        country_label.pack(anchor=tk.W)

        # Country dropdown with search
        country_search_frame = ttk.Frame(self.country_frame)
        country_search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(country_search_frame, text="Search:").pack(side=tk.LEFT)
        self.country_search_var = tk.StringVar()
        self.country_search_var.trace('w', self._filter_countries)
        country_search_entry = ttk.Entry(country_search_frame, textvariable=self.country_search_var, width=20)
        country_search_entry.pack(side=tk.LEFT, padx=5)

        # Country listbox with scrollbar
        country_list_frame = ttk.Frame(self.country_frame)
        country_list_frame.pack(fill=tk.X, pady=5)

        country_scrollbar = ttk.Scrollbar(country_list_frame)
        country_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.country_listbox = tk.Listbox(country_list_frame, height=8,
                                           yscrollcommand=country_scrollbar.set,
                                           font=('Helvetica', 9),
                                           selectbackground='#4ECDC4',
                                           exportselection=False)
        self.country_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        country_scrollbar.config(command=self.country_listbox.yview)

        # Populate country list
        self.all_countries = sorted(COUNTRIES.keys())
        for country in self.all_countries:
            self.country_listbox.insert(tk.END, country)

        # Bind selection event
        self.country_listbox.bind('<<ListboxSelect>>', self._on_country_select)

        # Selected country display
        self.selected_country_label = ttk.Label(self.country_frame, text="Selected: None",
                                                 font=('Helvetica', 9, 'bold'))
        self.selected_country_label.pack(anchor=tk.W, pady=(5, 0))

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

        # Map picker button
        map_picker_frame = ttk.Frame(self.custom_coords_frame)
        map_picker_frame.pack(fill=tk.X, pady=(10, 0))

        map_picker_btn = ttk.Button(map_picker_frame, text="Pick Region on Map",
                                     command=self._open_map_picker)
        map_picker_btn.pack(fill=tk.X)

        map_picker_hint = ttk.Label(map_picker_frame,
                                     text="Click and drag on a world map to select your region",
                                     font=('Helvetica', 8), foreground='gray')
        map_picker_hint.pack(anchor=tk.W, pady=(2, 0))

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
        region = self.selected_region.get()

        # Hide both frames first
        self.country_frame.pack_forget()
        self.custom_coords_frame.pack_forget()

        # Show appropriate frame
        if region == 'country':
            self.country_frame.pack(fill=tk.X, pady=(5, 0))
        elif region == 'custom':
            self.custom_coords_frame.pack(fill=tk.X, pady=(5, 0))

    def _filter_countries(self, *args):
        """Filter country list based on search text."""
        search_text = self.country_search_var.get().lower()
        self.country_listbox.delete(0, tk.END)

        for country in self.all_countries:
            if search_text in country.lower():
                self.country_listbox.insert(tk.END, country)

    def _on_country_select(self, event):
        """Handle country selection from listbox."""
        selection = self.country_listbox.curselection()
        if selection:
            self.selected_country = self.country_listbox.get(selection[0])
            self.selected_country_label.config(text=f"Selected: {self.selected_country}")

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

    def _open_map_picker(self):
        """Open a map picker dialog for selecting a custom region using tkinter Canvas."""
        import tempfile

        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror("Error", "PIL/Pillow is required for the map picker.\nInstall with: pip install Pillow")
            return

        # Create popup window
        picker_window = tk.Toplevel(self.root)
        picker_window.title("Select Region on Map")
        picker_window.geometry("900x700")
        picker_window.transient(self.root)

        # Instructions
        instruction_frame = ttk.Frame(picker_window, padding="10")
        instruction_frame.pack(fill=tk.X)

        ttk.Label(instruction_frame,
                  text="Click and drag to select a region. Release to confirm selection.",
                  font=('Helvetica', 10)).pack(side=tk.LEFT)

        # Generate world map image using matplotlib (Agg backend)
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='#E8E8E8', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#CCE5FF')
        ax.add_feature(cfeature.COASTLINE, edgecolor='#666666', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, edgecolor='#999999', linewidth=0.3)
        ax.add_feature(cfeature.LAKES, facecolor='#CCE5FF', edgecolor='#6699CC', linewidth=0.3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title("Click and drag to select your region", fontsize=12, fontweight='bold')

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        fig.savefig(temp_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Load image with PIL
        pil_image = Image.open(temp_path)
        img_width, img_height = pil_image.size

        # Create canvas frame
        map_frame = ttk.Frame(picker_window)
        map_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create tkinter canvas
        canvas = tk.Canvas(map_frame, width=img_width, height=img_height, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True)

        # Keep reference to prevent garbage collection
        picker_window.tk_image = ImageTk.PhotoImage(pil_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=picker_window.tk_image, tags="map")

        # Map coordinate conversion (approximate for PlateCarree projection)
        # The image spans -180 to 180 longitude and -90 to 90 latitude
        # But with margins from matplotlib, we need to account for that
        # Approximate margins: left ~10%, right ~5%, top ~8%, bottom ~12%
        left_margin = int(img_width * 0.10)
        right_margin = int(img_width * 0.05)
        top_margin = int(img_height * 0.08)
        bottom_margin = int(img_height * 0.12)

        map_width = img_width - left_margin - right_margin
        map_height = img_height - top_margin - bottom_margin

        def pixel_to_coords(px, py):
            """Convert pixel coordinates to lat/lon."""
            # Normalize to 0-1 range within the map area
            norm_x = (px - left_margin) / map_width
            norm_y = (py - top_margin) / map_height

            # Convert to lat/lon
            lon = -180 + norm_x * 360
            lat = 90 - norm_y * 180  # Y is inverted

            return lon, lat

        # Selection state
        class SelectionState:
            def __init__(self):
                self.start_px = None
                self.start_py = None
                self.rect_id = None
                self.final_coords = None
                self.is_dragging = False

        state = SelectionState()

        # Status label
        bottom_frame = ttk.Frame(picker_window, padding="10")
        bottom_frame.pack(fill=tk.X)

        status_label = ttk.Label(bottom_frame, text="Click and drag on the map to select a region",
                                  font=('Helvetica', 9), foreground='gray')
        status_label.pack(side=tk.LEFT)

        def on_press(event):
            """Handle mouse press."""
            state.start_px = event.x
            state.start_py = event.y
            state.is_dragging = True

            # Remove previous rectangle
            if state.rect_id is not None:
                canvas.delete(state.rect_id)
                state.rect_id = None

        def on_motion(event):
            """Handle mouse motion while dragging."""
            if not state.is_dragging or state.start_px is None:
                return

            # Remove previous rectangle
            if state.rect_id is not None:
                canvas.delete(state.rect_id)

            # Draw new rectangle
            state.rect_id = canvas.create_rectangle(
                state.start_px, state.start_py, event.x, event.y,
                outline='red', width=2, fill='red', stipple='gray50'
            )

            # Calculate coordinates
            lon1, lat1 = pixel_to_coords(state.start_px, state.start_py)
            lon2, lat2 = pixel_to_coords(event.x, event.y)

            lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
            lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)

            # Clamp to valid ranges
            lon_min = max(-180, min(180, lon_min))
            lon_max = max(-180, min(180, lon_max))
            lat_min = max(-90, min(90, lat_min))
            lat_max = max(-90, min(90, lat_max))

            state.final_coords = (lon_min, lon_max, lat_min, lat_max)

            status_label.config(
                text=f"Selection: Lon [{lon_min:.1f} to {lon_max:.1f}], Lat [{lat_min:.1f} to {lat_max:.1f}]"
            )

        def on_release(event):
            """Handle mouse release."""
            if not state.is_dragging:
                return

            state.is_dragging = False

            if state.final_coords is not None:
                lon_min, lon_max, lat_min, lat_max = state.final_coords

                # Validate selection size
                if abs(lon_max - lon_min) < 1 or abs(lat_max - lat_min) < 1:
                    status_label.config(text="Selection too small. Please select a larger area.")
                    return

                # Update the coordinate entry fields
                self.custom_lon_min.set(f"{lon_min:.1f}")
                self.custom_lon_max.set(f"{lon_max:.1f}")
                self.custom_lat_min.set(f"{lat_min:.1f}")
                self.custom_lat_max.set(f"{lat_max:.1f}")

                # Switch to custom region
                self.selected_region.set('custom')
                self._on_region_change()

                status_label.config(
                    text=f"Selected: Lon [{lon_min:.1f} to {lon_max:.1f}], Lat [{lat_min:.1f} to {lat_max:.1f}] - Click Apply"
                )

        # Bind mouse events
        canvas.bind('<Button-1>', on_press)
        canvas.bind('<B1-Motion>', on_motion)
        canvas.bind('<ButtonRelease-1>', on_release)

        def apply_and_close():
            """Apply selection and close window."""
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            picker_window.destroy()

        def cancel():
            """Cancel and close window."""
            try:
                os.unlink(temp_path)
            except:
                pass
            picker_window.destroy()

        ttk.Button(bottom_frame, text="Apply Selection", command=apply_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT)

        # Handle window close
        def on_closing():
            try:
                os.unlink(temp_path)
            except:
                pass
            picker_window.destroy()

        picker_window.protocol("WM_DELETE_WINDOW", on_closing)

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

            # Get region config - handle country and custom regions
            if region == 'country':
                if not self.selected_country:
                    raise ValueError("Please select a country from the list")

                bounds = COUNTRIES[self.selected_country]
                lon_min, lon_max, lat_min, lat_max = bounds

                # Calculate appropriate grid resolution based on country size
                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                max_range = max(lon_range, lat_range)

                # Adjust resolution to keep API calls reasonable (max ~100 points)
                if max_range > 30:
                    grid_res = 2.0
                elif max_range > 15:
                    grid_res = 1.5
                elif max_range > 8:
                    grid_res = 1.0
                else:
                    grid_res = 0.5

                # Calculate center for projection
                center_lon = (lon_min + lon_max) / 2
                center_lat = (lat_min + lat_max) / 2

                region_config = {
                    'name': self.selected_country,
                    'bounds': bounds,
                    'grid_resolution': grid_res,
                    'projection': ccrs.LambertConformal(central_longitude=center_lon, central_latitude=center_lat)
                }
                # Add country region to REGIONS temporarily
                REGIONS['country'] = region_config
                region = 'country'

            elif region == 'custom':
                try:
                    lon_min = float(self.custom_lon_min.get())
                    lon_max = float(self.custom_lon_max.get())
                    lat_min = float(self.custom_lat_min.get())
                    lat_max = float(self.custom_lat_max.get())

                    if lon_min >= lon_max or lat_min >= lat_max:
                        raise ValueError("Invalid coordinates: min must be less than max")

                    # Calculate appropriate grid resolution
                    lon_range = lon_max - lon_min
                    lat_range = lat_max - lat_min
                    max_range = max(lon_range, lat_range)

                    if max_range > 30:
                        grid_res = 2.0
                    elif max_range > 15:
                        grid_res = 1.5
                    elif max_range > 8:
                        grid_res = 1.0
                    else:
                        grid_res = 0.5

                    region_config = {
                        'name': 'Custom Region',
                        'bounds': [lon_min, lon_max, lat_min, lat_max],
                        'grid_resolution': grid_res,
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
            try:
                self.canvas.get_tk_widget().destroy()
            except:
                pass
            self.canvas = None
        if self.current_figure:
            try:
                plt.close(self.current_figure)
            except:
                pass
            self.current_figure = None
        # Clear any preview image label
        if hasattr(self, 'preview_image_label') and self.preview_image_label:
            self.preview_image_label.destroy()
            self.preview_image_label = None
        # Clear PIL image reference
        if hasattr(self, 'preview_pil_image'):
            self.preview_pil_image = None
        if hasattr(self, 'preview_tk_image'):
            self.preview_tk_image = None

    def _show_map_preview(self, filepath: str):
        """Display a map in the preview panel using tkinter native widgets."""
        self._clear_preview()

        try:
            from PIL import Image, ImageTk

            # Load image with PIL
            self.preview_pil_image = Image.open(filepath)

            # Get the preview frame size
            self.preview_frame.update_idletasks()
            frame_width = self.preview_frame.winfo_width() - 20
            frame_height = self.preview_frame.winfo_height() - 20

            # Resize image to fit frame while maintaining aspect ratio
            img_width, img_height = self.preview_pil_image.size
            ratio = min(frame_width / img_width, frame_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            if new_width > 0 and new_height > 0:
                resized = self.preview_pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized = self.preview_pil_image

            # Convert to PhotoImage for tkinter
            self.preview_tk_image = ImageTk.PhotoImage(resized)

            # Create label to display image
            self.preview_image_label = ttk.Label(self.preview_frame, image=self.preview_tk_image)
            self.preview_image_label.pack(expand=True)

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
