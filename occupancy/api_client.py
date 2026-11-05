"""Get readings: from the real API (platform repository) or from the simulated data.

Contract (platform/docs/api/openapi.yaml):
    GET /api/v1/readings/{building}/{roomNumber}
        ?start_time=...&end_time=...&sensors=...&sensors=...
- Times as 'YYYY-MM-DD HH:mm:ss.SSS' in Europe/Amsterdam local time.
- Range of at most 24 hours.
- 404 = none of the requested sensors has data in that range (not a fatal error).
"""
import json
from datetime import datetime
from urllib.parse import quote, urlencode

import requests

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def format_time(moment):
    """Datetime -> text in the API format (milliseconds)."""
    return moment.strftime(TIME_FORMAT)[:-3]


class ApiSource:
    """Reads from the real API."""

    def __init__(self, config, timeout=10):
        api = config["api"]
        self.url = f"{api['base_url'].rstrip('/')}/api/v1/readings/{api['building']}/{api['room']}"
        self.sensors = config["sensors"]
        self.timeout = timeout

    def fetch(self, start, end):
        """Return the 'data' field of the response, or None if there is no data (404)."""
        params = {"start_time": format_time(start),
                  "end_time": format_time(end),
                  "sensors": self.sensors}
        # doseq repeats the key for each sensor; quote encodes the space as %20
        query = urlencode(params, doseq=True, quote_via=quote)
        response = requests.get(f"{self.url}?{query}", timeout=self.timeout)
        if response.status_code == 404:
            return None
        body = response.json()
        if response.status_code != 200:
            raise RuntimeError(f"The API answered {response.status_code}: {body.get('message')}")
        return body["data"]

    def patch_prediction(self, start, end, level):
        """Apply one predicted level to all matching room readings."""
        body = {
            "start_time": format_time(start),
            "end_time": format_time(end),
            "busy_level": int(level),
        }
        response = requests.patch(self.url, json=body, timeout=self.timeout)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"The prediction API returned invalid JSON ({response.status_code})"
            ) from exc
        if response.status_code != 200:
            message = payload.get("message", "unknown error")
            raise RuntimeError(
                f"The prediction API answered {response.status_code}: {message}"
            )
        return payload["data"]


class SimulatedSource:
    """Reads data/simulated/readings.json as if it were the API.

    Used to test the live prediction without the API: returns only the
    readings between start and end, in the same format as the real API.
    """

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)["data"]

    def time_span(self):
        """First and last timestamp of the simulated data."""
        stamps = [datetime.strptime(r["timestamp"], TIME_FORMAT)
                  for readings in self.data.values() for r in readings]
        return min(stamps), max(stamps)

    def fetch(self, start, end):
        result = {}
        for sensor, readings in self.data.items():
            result[sensor] = [r for r in readings
                              if start <= datetime.strptime(r["timestamp"], TIME_FORMAT) <= end]
        if not any(result.values()):
            return None
        return result
