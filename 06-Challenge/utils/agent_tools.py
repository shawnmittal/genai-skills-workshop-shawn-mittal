import os
import requests
from typing import Dict, Any, Optional, List, Tuple

GOOGLE_MAP_KEY = os.getenv("GOOGLE_MAP_KEY")


def get_lat_lon_from_address(
    address: str, api_key: str
) -> Optional[Tuple[float, float]]:
    """Converts a physical address to latitude and longitude.

    Uses Google's Geocoding API.

    Args:
        address: The street address or place name to geocode
                 (e.g., "1600 Amphitheatre Parkway, Mountain View, CA").
        api_key: Your Google Maps Platform API key.

    Returns:
        A tuple containing the latitude and longitude (lat, lon), or None if
        the address could not be geocoded.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': address,
        'key': api_key
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return (location['lat'], location['lng'])
        else:
            error_message = data.get('error_message', '')
            print(
                f"Geocoding API Error: {data.get('status')} - {error_message}"
            )
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the network request: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing the Geocoding API response: {e}")
        return None


def get_weather_forecast(
    latitude: float, longitude: float
) -> Optional[List[Dict[str, Any]]]:
    """Retrieves the weather forecast from the National Weather Service API.

    This function takes a latitude and longitude, first finds the
    corresponding NWS grid forecast endpoint, and then fetches the detailed
    weather forecast for that location.

    Args:
        latitude: The latitude of the location (e.g., 38.8977).
        longitude: The longitude of the location (e.g., -77.0365).

    Returns:
        A list of dictionaries, where each dictionary represents a forecast
        period (e.g., 'Tonight', 'Thursday'). Returns None if an error
        occurs or the data cannot be fetched.
    """
    headers = {
        'User-Agent': 'MyWeatherApp/1.0 (contact@example.com)'
    }
    points_url = f"https://api.weather.gov/points/{latitude},{longitude}"

    try:
        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_response.raise_for_status()
        points_data = points_response.json()
        forecast_url = points_data.get('properties', {}).get('forecast')

        if not forecast_url:
            print("Error: Could not find forecast URL in the API response.")
            return None

        forecast_response = requests.get(
            forecast_url, headers=headers, timeout=10
        )
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        periods = forecast_data.get('properties', {}).get('periods')
        return periods

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the network request: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing the NWS API response: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_weather(address: str) -> Optional[List[Dict[str, Any]]]:
    """
    Takes an address and returns a weather forecast from the National Weather
    Service.

    Args:
        address: The street address or place name to geocode
                 (e.g., "1600 Amphitheatre Parkway, Mountain View, CA").

    Returns:
        A list of dictionaries, where each dictionary represents a forecast
        period (e.g., 'Tonight', 'Thursday'). Returns None if an error occurs.
    """
    try:
        lat, lon = get_lat_lon_from_address(
            address=address, api_key=GOOGLE_MAP_KEY
        )
        forecast = get_weather_forecast(lat, lon)
        return forecast
    except Exception as e:
        print(f"Something broke. Good luck fixing:\n{e}")
        return None
