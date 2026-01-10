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
            'hourly': ','.join([
                'temperature_2m',
                'relative_humidity_2m',
                'dewpoint_2m',
                'apparent_temperature',
                'precipitation_probability',
                'precipitation',
                'rain',
                'showers',
                'snowfall',
                'snow_depth',
                'weather_code',
                'pressure_msl',
                'surface_pressure',
                'cloud_cover',
                'visibility',
                'evapotranspiration',
                'wind_speed_10m',
                'wind_speed_80m',
                'wind_direction_10m',
                'wind_gusts_10m',
                'cape',
                'lifted_index',
                'convective_inhibition',
                'freezing_level_height',
                'soil_temperature_0cm',
                'soil_moisture_0_to_1cm',
                'soil_moisture_1_to_3cm',
                'soil_moisture_3_to_9cm',
                'soil_moisture_9_to_27cm'
            ]),
            'daily': ','.join([
                'weather_code',
                'temperature_2m_max',
                'temperature_2m_min',
                'apparent_temperature_max',
                'apparent_temperature_min',
                'precipitation_sum',
                'rain_sum',
                'showers_sum',
                'snowfall_sum',
                'precipitation_hours',
                'precipitation_probability_max',
                'wind_speed_10m_max',
                'wind_gusts_10m_max',
                'wind_direction_10m_dominant',
                'et0_fao_evapotranspiration'
            ]),
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
    def _safe_max(values, default=0):
        """Safely get max from list, handling None values."""
        filtered = [v for v in values if v is not None]
        return max(filtered) if filtered else default

    @staticmethod
    def _safe_min(values, default=0):
        """Safely get min from list, handling None values."""
        filtered = [v for v in values if v is not None]
        return min(filtered) if filtered else default

    @staticmethod
    def _safe_mean(values, default=0):
        """Safely get mean from list, handling None values."""
        filtered = [v for v in values if v is not None]
        return sum(filtered) / len(filtered) if filtered else default

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
        if not data or 'hourly' not in data:
            return 0

        hourly = data['hourly']

        # Get values for next 24 hours
        cape_values = hourly.get('cape', [0])[:24]
        li_values = hourly.get('lifted_index', [0])[:24]
        cin_values = hourly.get('convective_inhibition', [0])[:24]
        wind_10m = hourly.get('wind_speed_10m', [0])[:24]
        wind_80m = hourly.get('wind_speed_80m', [0])[:24]
        wind_gust_values = hourly.get('wind_gusts_10m', [0])[:24]
        precip_values = hourly.get('precipitation', [0])[:24]
        weather_codes = hourly.get('weather_code', [0])[:24]
        precip_prob = hourly.get('precipitation_probability', [0])[:24]

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
        if not data or 'daily' not in data:
            return 0

        daily = data['daily']
        hourly = data.get('hourly', {})

        # Get precipitation data
        precip_sum = daily.get('precipitation_sum', [0])[:3]
        precip_hours = daily.get('precipitation_hours', [0])[:3]
        rain_sum = daily.get('rain_sum', [0])[:3]
        showers_sum = daily.get('showers_sum', [0])[:3]
        precip_prob = daily.get('precipitation_probability_max', [0])[:3]

        # Get hourly data
        hourly_precip = hourly.get('precipitation', [0])[:72]
        hourly_rain = hourly.get('rain', [0])[:72]
        hourly_showers = hourly.get('showers', [0])[:72]

        # Soil moisture at multiple depths
        soil_0_1 = hourly.get('soil_moisture_0_to_1cm', [0.3])[:24]
        soil_1_3 = hourly.get('soil_moisture_1_to_3cm', [0.3])[:24]
        soil_3_9 = hourly.get('soil_moisture_3_to_9cm', [0.3])[:24]
        soil_9_27 = hourly.get('soil_moisture_9_to_27cm', [0.3])[:24]

        # Calculate key metrics
        total_precip = sum([v for v in precip_sum if v is not None] or [0])
        max_daily_precip = RiskCalculator._safe_max(precip_sum, 0)
        total_hours = sum([v for v in precip_hours if v is not None] or [0])
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
        if not data or 'hourly' not in data:
            return 0

        hourly = data['hourly']
        daily = data.get('daily', {})

        # Get hourly parameters
        temp_values = hourly.get('temperature_2m', [20])[:24]
        dewpoint_values = hourly.get('dewpoint_2m', [10])[:24]
        rh_values = hourly.get('relative_humidity_2m', [50])[:24]
        wind_10m = hourly.get('wind_speed_10m', [10])[:24]
        wind_80m = hourly.get('wind_speed_80m', [15])[:24]
        gust_values = hourly.get('wind_gusts_10m', [15])[:24]
        evap_values = hourly.get('evapotranspiration', [0])[:24]
        soil_temp = hourly.get('soil_temperature_0cm', [20])[:24]

        # Soil moisture at multiple depths
        soil_0_1 = hourly.get('soil_moisture_0_to_1cm', [0.3])[:24]
        soil_1_3 = hourly.get('soil_moisture_1_to_3cm', [0.3])[:24]
        soil_3_9 = hourly.get('soil_moisture_3_to_9cm', [0.3])[:24]

        # Daily parameters
        precip_sum = daily.get('precipitation_sum', [0])[:3]
        et0_values = daily.get('et0_fao_evapotranspiration', [0])[:3]

        # Calculate key metrics
        max_temp = RiskCalculator._safe_max(temp_values, 20)
        min_rh = RiskCalculator._safe_min(rh_values, 50)
        mean_rh = RiskCalculator._safe_mean(rh_values, 50)
        max_wind = RiskCalculator._safe_max(wind_10m, 10)
        max_gust = RiskCalculator._safe_max(gust_values, 15)
        total_precip = sum([v for v in precip_sum if v is not None] or [0])
        total_et0 = sum([v for v in et0_values if v is not None] or [0])

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

        # Add Kaldock branding in corner
        plt.figtext(0.01, 0.01, 'Created by Kaldock',
                    ha='left', fontsize=9, fontweight='bold',
                    color='#333333', alpha=0.8)

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
        np.random.seed(42)  # For reproducibility

        for lat in lats:
            for lon in lons:
                # Create synthetic but realistic-looking weather patterns
                lat_factor = (lat - lat_min) / (lat_max - lat_min) if lat_max != lat_min else 0.5
                lon_factor = (lon - lon_min) / (lon_max - lon_min) if lon_max != lon_min else 0.5

                # Add some spatial correlation and randomness
                noise = np.random.random()
                noise2 = np.random.random()
                pattern = np.sin(lat_factor * np.pi) * np.cos(lon_factor * np.pi * 2)
                storm_pattern = np.sin((lat_factor + lon_factor) * np.pi * 1.5) * noise

                # Temperature based on latitude (warmer in south for northern hemisphere)
                base_temp = 15 + 25 * (1 - lat_factor) + 5 * noise
                dewpoint = base_temp - 5 - 15 * noise2

                # Humidity inversely related to temperature
                base_rh = max(10, 80 - 50 * lat_factor - 20 * noise)

                # Wind patterns
                wind_10m = 8 + 25 * noise * abs(pattern)
                wind_80m = wind_10m * (1.3 + 0.4 * noise2)
                wind_gust = wind_10m * (1.5 + 0.5 * noise)

                # Precipitation pattern
                precip = max(0, 8 * (noise + pattern * 0.3))
                precip_prob = min(100, max(0, 30 + 60 * storm_pattern))

                # Soil moisture (drier in warm areas)
                soil_base = 0.35 - 0.2 * lat_factor + 0.1 * noise

                # CAPE and instability
                cape = max(0, 500 + 3000 * storm_pattern * noise)
                lifted_index = 2 - 8 * storm_pattern * noise
                cin = -50 - 100 * noise2

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
                        'weather_code': [95 if storm_pattern > 0.5 else int(3 * noise)] * 3,
                        'temperature_2m_max': [base_temp + 8] * 3,
                        'temperature_2m_min': [base_temp - 5] * 3,
                        'apparent_temperature_max': [base_temp + 10] * 3,
                        'apparent_temperature_min': [base_temp - 7] * 3,
                        'precipitation_sum': [max(0, precip * 12 * (0.5 + 0.5 * np.random.random())) for _ in range(3)],
                        'rain_sum': [max(0, precip * 10 * (0.5 + 0.5 * np.random.random())) for _ in range(3)],
                        'showers_sum': [max(0, precip * 2 * np.random.random()) for _ in range(3)],
                        'snowfall_sum': [0] * 3,
                        'precipitation_hours': [int(6 + 12 * storm_pattern * noise)] * 3,
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
