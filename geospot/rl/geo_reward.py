"""
Geodesic reward computation for geolocation.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import NamedTuple

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


class GeoLocation(NamedTuple):
    """A geographic location with optional hierarchical info."""

    lat: float
    lon: float
    city: str | None = None
    region: str | None = None
    country: str | None = None


class ParsedGeoResponse(NamedTuple):
    """Result of parsing a model's geo prediction."""

    location: GeoLocation | None
    format_valid: bool
    raw_text: str


def parse_geo_response(response: str) -> ParsedGeoResponse:
    """Parse model response to extract location prediction."""
    response = response.strip()

    # Try structured format
    loc = _parse_structured_format(response)
    if loc:
        return ParsedGeoResponse(location=loc, format_valid=True, raw_text=response)

    # Try coordinate-only
    loc = _parse_coordinate_format(response)
    if loc:
        return ParsedGeoResponse(location=loc, format_valid=True, raw_text=response)

    # Try JSON
    loc = _parse_json_format(response)
    if loc:
        return ParsedGeoResponse(location=loc, format_valid=True, raw_text=response)

    return ParsedGeoResponse(location=None, format_valid=False, raw_text=response)


def _parse_structured_format(text: str) -> GeoLocation | None:
    lat = lon = None
    city = region = country = None

    patterns = {
        "lat": r"(?:latitude|lat)[:\s]+(-?\d+\.?\d*)",
        "lon": r"(?:longitude|lon|lng)[:\s]+(-?\d+\.?\d*)",
        "city": r"city[:\s]+([^\n]+)",
        "region": r"(?:region|state|province)[:\s]+([^\n]+)",
        "country": r"country[:\s]+([^\n]+)",
    }

    text_lower = text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == "lat":
                lat = _safe_float(value)
            elif key == "lon":
                lon = _safe_float(value)
            elif key == "city":
                city = value.title()
            elif key == "region":
                region = value.title()
            elif key == "country":
                country = value.title()

    if lat is not None and lon is not None and _valid_coords(lat, lon):
        return GeoLocation(lat=lat, lon=lon, city=city, region=region, country=country)
    return None


def _parse_coordinate_format(text: str) -> GeoLocation | None:
    pattern = r"(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        lat = _safe_float(match.group(1))
        lon = _safe_float(match.group(2))
        if lat is not None and lon is not None and _valid_coords(lat, lon):
            return GeoLocation(lat=lat, lon=lon)
    return None


def _parse_json_format(text: str) -> GeoLocation | None:
    try:
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            data = json.loads(json_match.group())
            lat = data.get("lat") or data.get("latitude")
            lon = data.get("lon") or data.get("lng") or data.get("longitude")
            if lat is not None and lon is not None and _valid_coords(lat, lon):
                return GeoLocation(
                    lat=float(lat),
                    lon=float(lon),
                    city=data.get("city"),
                    region=data.get("region") or data.get("state"),
                    country=data.get("country"),
                )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _valid_coords(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


@dataclass
class GeoRewardConfig:
    """Config for geo reward. Default tau=25km gives city-level sensitivity."""

    coord_tau: float = 25.0
    coord_weight: float = 0.7
    city_weight: float = 0.1
    region_weight: float = 0.1
    country_weight: float = 0.1
    format_penalty: float = 0.1


@dataclass
class GeoRewardResult:
    """Result of computing geo reward."""

    total_reward: float
    coord_reward: float
    city_reward: float
    region_reward: float
    country_reward: float
    distance_km: float | None
    format_valid: bool

    def to_metrics(self) -> dict[str, float]:
        metrics = {
            "reward/total": self.total_reward,
            "reward/coord": self.coord_reward,
            "reward/city": self.city_reward,
            "reward/region": self.region_reward,
            "reward/country": self.country_reward,
            "format_valid": float(self.format_valid),
        }
        if self.distance_km is not None:
            metrics["distance_km"] = self.distance_km
        return metrics


def compute_geo_reward(
    prediction: ParsedGeoResponse,
    ground_truth: GeoLocation,
    config: GeoRewardConfig | None = None,
) -> GeoRewardResult:
    """Compute hierarchical geo reward. reward = exp(-distance / tau)."""
    if config is None:
        config = GeoRewardConfig()

    # Normalize weights
    total_weight = config.coord_weight + config.city_weight + config.region_weight + config.country_weight
    w_coord = config.coord_weight / total_weight
    w_city = config.city_weight / total_weight
    w_region = config.region_weight / total_weight
    w_country = config.country_weight / total_weight

    if prediction.location is None:
        return GeoRewardResult(
            total_reward=-config.format_penalty,
            coord_reward=0.0,
            city_reward=0.0,
            region_reward=0.0,
            country_reward=0.0,
            distance_km=None,
            format_valid=False,
        )

    pred, gt = prediction.location, ground_truth

    distance_km = haversine_km(pred.lat, pred.lon, gt.lat, gt.lon)
    coord_reward = math.exp(-distance_km / config.coord_tau)

    city_reward = _text_match(pred.city, gt.city)
    region_reward = _text_match(pred.region, gt.region)
    country_reward = _text_match(pred.country, gt.country)

    total_reward = (
        w_coord * coord_reward
        + w_city * city_reward
        + w_region * region_reward
        + w_country * country_reward
    )

    return GeoRewardResult(
        total_reward=total_reward,
        coord_reward=coord_reward,
        city_reward=city_reward,
        region_reward=region_reward,
        country_reward=country_reward,
        distance_km=distance_km,
        format_valid=True,
    )


def _text_match(pred: str | None, gt: str | None) -> float:
    """1.0 for exact match (case-insensitive), 0.0 otherwise."""
    if gt is None or pred is None:
        return 0.0
    return 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0


def distance_bucket(distance_km: float) -> str:
    """GeoGuessr-style distance buckets."""
    if distance_km < 1:
        return "<1km"
    elif distance_km < 25:
        return "1-25km"
    elif distance_km < 200:
        return "25-200km"
    elif distance_km < 750:
        return "200-750km"
    elif distance_km < 2500:
        return "750-2500km"
    return ">2500km"


def geoguessr_score(distance_km: float) -> int:
    """GeoGuessr-style score (0-5000 points)."""
    if distance_km < 0.05:
        return 5000
    return max(0, int(5000 * math.exp(-distance_km / 2000)))
