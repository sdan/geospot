"""
Geodesic reward computation for geolocation RL.

Single-step reward with:
- Distance-based reward (exp(-d/tau) or geoguessr score)
- Geocell hierarchy via geohash prefix matching
- Format penalty for unparseable responses
"""

import math
import re
from dataclasses import dataclass
from typing import NamedTuple

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


class GeoLocation(NamedTuple):
    """Geographic location with optional hierarchy."""

    lat: float
    lon: float
    city: str | None = None
    region: str | None = None
    country: str | None = None


# -----------------------------------------------------------------------------
# Reverse Geocoding (coords → country/region)
# -----------------------------------------------------------------------------

class ReverseGeocoder:
    """
    Offline reverse geocoder using reverse_geocoder library.

    Converts (lat, lon) → {country, region, city}.
    Uses K-D tree for fast lookups, no API calls needed.

    Install: pip install reverse_geocoder
    """

    _instance: "ReverseGeocoder | None" = None
    _rg = None  # Lazy load reverse_geocoder module

    def __new__(cls) -> "ReverseGeocoder":
        # Singleton pattern - reverse_geocoder loads a large dataset on first use
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self):
        if self._rg is None:
            try:
                import reverse_geocoder as rg
                # Pre-load the dataset (happens once)
                rg.search([(0, 0)])
                ReverseGeocoder._rg = rg
            except ImportError:
                raise ImportError(
                    "reverse_geocoder not installed. Run: pip install reverse_geocoder"
                )

    def __call__(self, lat: float, lon: float) -> dict[str, str | None]:
        """
        Reverse geocode coordinates to location info.

        Returns:
            {"country": "US", "region": "California", "city": "San Francisco"}
        """
        self._ensure_loaded()

        if not _valid_coords(lat, lon):
            return {"country": None, "region": None, "city": None}

        results = self._rg.search([(lat, lon)])
        if not results:
            return {"country": None, "region": None, "city": None}

        result = results[0]
        return {
            "country": result.get("cc"),       # ISO-2 country code
            "region": result.get("admin1"),    # State/province
            "city": result.get("name"),        # Nearest city
        }

    def batch(self, coords: list[tuple[float, float]]) -> list[dict[str, str | None]]:
        """Batch reverse geocode for efficiency."""
        self._ensure_loaded()

        out = [{"country": None, "region": None, "city": None} for _ in coords]

        valid_pairs: list[tuple[float, float]] = []
        valid_indices: list[int] = []
        for i, (lat, lon) in enumerate(coords):
            if _valid_coords(lat, lon):
                valid_pairs.append((lat, lon))
                valid_indices.append(i)

        if not valid_pairs:
            return out

        results = self._rg.search(valid_pairs)
        for i, r in zip(valid_indices, results, strict=True):
            out[i] = {
                "country": r.get("cc"),
                "region": r.get("admin1"),
                "city": r.get("name"),
            }

        return out


def geo_location_from_coords(
    lat: float,
    lon: float,
    reverse_geocode: bool = True,
) -> GeoLocation:
    """
    Create a GeoLocation from coordinates, optionally reverse geocoding.

    Args:
        lat: Latitude
        lon: Longitude
        reverse_geocode: If True, derive country/region/city from coords

    Returns:
        GeoLocation with populated fields
    """
    if not reverse_geocode:
        return GeoLocation(lat=lat, lon=lon)

    geocoder = ReverseGeocoder()
    info = geocoder(lat, lon)

    return GeoLocation(
        lat=lat,
        lon=lon,
        city=info.get("city"),
        region=info.get("region"),
        country=info.get("country"),
    )


class ParsedGeoResponse(NamedTuple):
    """Result of parsing model output."""

    location: GeoLocation | None
    format_valid: bool
    raw_text: str


# -----------------------------------------------------------------------------
# Response parsing (strict format from SFT)
# -----------------------------------------------------------------------------


def parse_geo_response(response: str) -> ParsedGeoResponse:
    """
    Parse model response. Accepts:

        Latitude: <degrees>
        Longitude: <degrees>

    Or bare coordinates:

        <lat>, <lon>

    Strips <think> blocks first.
    """
    response = response.strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Try structured format first
    lat_match = re.search(r"Latitude:\s*(-?\d+\.?\d*)", response)
    lon_match = re.search(r"Longitude:\s*(-?\d+\.?\d*)", response)

    if lat_match and lon_match:
        try:
            lat, lon = float(lat_match.group(1)), float(lon_match.group(1))
            if _valid_coords(lat, lon):
                return ParsedGeoResponse(GeoLocation(lat, lon), True, response)
        except ValueError:
            pass

    # Try bare coords: "lat, lon"
    bare_match = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", response)
    if bare_match:
        try:
            lat, lon = float(bare_match.group(1)), float(bare_match.group(2))
            if _valid_coords(lat, lon):
                return ParsedGeoResponse(GeoLocation(lat, lon), True, response)
        except ValueError:
            pass

    return ParsedGeoResponse(None, False, response)


def _valid_coords(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


# -----------------------------------------------------------------------------
# Geohash (for geocell hierarchy)
# -----------------------------------------------------------------------------

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int = 6) -> str:
    """Encode (lat, lon) into geohash. Each char ~= one level of hierarchy."""
    if precision <= 0 or not _valid_coords(lat, lon):
        return ""
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    chars: list[str] = []
    bit, ch, even = 0, 0, True
    while len(chars) < precision:
        if even:
            mid = (lon_min + lon_max) / 2
            if lon > mid:
                ch |= 1 << (4 - bit)
                lon_min = mid
            else:
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2
            if lat > mid:
                ch |= 1 << (4 - bit)
                lat_min = mid
            else:
                lat_max = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            chars.append(_GEOHASH_BASE32[ch])
            bit, ch = 0, 0
    return "".join(chars)


def common_prefix_len(a: str, b: str) -> int:
    """Return length of common prefix between two strings."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


# -----------------------------------------------------------------------------
# Reward computation
# -----------------------------------------------------------------------------


@dataclass
class GeoRewardConfig:
    """
    Reward configuration.

    - coord_tau: decay constant for exp(-d/tau). Start high (coarse), anneal low (fine).
    - coord_reward_kind: "exp" for exp(-d/tau), "geoguessr" for geoguessr scoring / 5000.
    - geohash_precision: length of geohash for geocell reward (0 to disable).
    - geocell_weight: weight for geocell prefix match reward.
    - format_penalty: penalty when response can't be parsed.
    """

    coord_tau: float = 25.0
    coord_reward_kind: str = "exp"  # "exp" or "geoguessr"
    coord_weight: float = 0.7
    geocell_weight: float = 0.3
    geohash_precision: int = 5
    format_penalty: float = 0.1


@dataclass
class GeoRewardResult:
    """Reward breakdown for logging."""

    total_reward: float
    coord_reward: float
    geocell_reward: float
    distance_km: float | None
    format_valid: bool
    geohash_prefix_len: int = 0

    def to_metrics(self) -> dict[str, float]:
        m = {
            "reward/total": self.total_reward,
            "reward/coord": self.coord_reward,
            "reward/geocell": self.geocell_reward,
            "format_valid": float(self.format_valid),
            "geohash_prefix_len": float(self.geohash_prefix_len),
        }
        if self.distance_km is not None:
            m["distance_km"] = self.distance_km
        return m


def compute_geo_reward(
    prediction: ParsedGeoResponse,
    ground_truth: GeoLocation,
    config: GeoRewardConfig | None = None,
) -> GeoRewardResult:
    """
    Compute single-step geo reward.

    reward = w_coord * coord_reward + w_geocell * geocell_reward
    """
    cfg = config or GeoRewardConfig()

    # Invalid parse -> format penalty
    if prediction.location is None:
        return GeoRewardResult(
            total_reward=-cfg.format_penalty,
            coord_reward=0.0,
            geocell_reward=0.0,
            distance_km=None,
            format_valid=False,
        )

    pred, gt = prediction.location, ground_truth

    # Distance reward
    distance_km = haversine_km(pred.lat, pred.lon, gt.lat, gt.lon)
    if cfg.coord_reward_kind == "exp":
        coord_reward = math.exp(-distance_km / cfg.coord_tau)
    elif cfg.coord_reward_kind == "geoguessr":
        coord_reward = geoguessr_score(distance_km) / 5000.0
    else:
        raise ValueError(f"Unknown coord_reward_kind: {cfg.coord_reward_kind}")

    # Geocell reward (prefix match)
    geocell_reward = 0.0
    geohash_prefix_len = 0
    if cfg.geocell_weight > 0 and cfg.geohash_precision > 0:
        pred_hash = geohash_encode(pred.lat, pred.lon, cfg.geohash_precision)
        gt_hash = geohash_encode(gt.lat, gt.lon, cfg.geohash_precision)
        geohash_prefix_len = common_prefix_len(pred_hash, gt_hash)
        geocell_reward = geohash_prefix_len / cfg.geohash_precision

    # Weighted sum
    w_total = cfg.coord_weight + cfg.geocell_weight
    if w_total <= 0:
        w_total = 1.0
    total_reward = (cfg.coord_weight * coord_reward + cfg.geocell_weight * geocell_reward) / w_total

    return GeoRewardResult(
        total_reward=total_reward,
        coord_reward=coord_reward,
        geocell_reward=geocell_reward,
        distance_km=distance_km,
        format_valid=True,
        geohash_prefix_len=geohash_prefix_len,
    )


def geoguessr_score(distance_km: float) -> int:
    """GeoGuessr-style score (0-5000). Non-saturating compared to exp(-d/25)."""
    if distance_km < 0.05:
        return 5000
    return max(0, int(5000 * math.exp(-distance_km / 2000)))


def distance_bucket(distance_km: float) -> str:
    """For stratified eval metrics."""
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
