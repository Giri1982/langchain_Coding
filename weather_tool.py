from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather details for a city."""
    weather_by_city = {
        "chennai": "31C, humid, partly cloudy",
        "london": "18C, light rain",
        "new york": "24C, sunny",
        "tokyo": "27C, clear skies",
        "paris": "22C, breezy",
    }
    return weather_by_city.get(
        city.strip().lower(), "Weather data unavailable for this city."
    )