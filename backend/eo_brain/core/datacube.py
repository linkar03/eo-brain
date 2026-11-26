"""
eo_brain/core/datacube.py - DataCube Engine v4

Changes from v3:
- STAC search retry with exponential backoff (network resilience)
- Configurable HTTP timeout for STAC client
- UTM zone boundary warning preserved
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import time
import warnings

import xarray as xr
import numpy as np
from pystac_client import Client as STACClient
from pystac import Item

logger = logging.getLogger(__name__)

# Retry config for STAC catalog queries
_STAC_MAX_RETRIES = 3
_STAC_BACKOFF_BASE = 1.5  # seconds, doubles each retry
_STAC_TIMEOUT_SEC = 30


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        if self.west >= self.east:
            raise ValueError("west must be < east")
        if self.south >= self.north:
            raise ValueError("south must be < north")

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def center(self) -> Tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)

    def width_deg(self) -> float:
        return self.east - self.west

    def _lon_to_utm_zone(self, lon: float) -> int:
        return int((lon + 180) / 6) + 1

    def crosses_utm_zones(self) -> bool:
        zone_west = self._lon_to_utm_zone(self.west)
        zone_east = self._lon_to_utm_zone(self.east)
        return zone_west != zone_east

    def estimate_utm_crs(self) -> str:
        lon_center, lat_center = self.center()
        zone = self._lon_to_utm_zone(lon_center)
        hemisphere = "N" if lat_center >= 0 else "S"
        epsg = 32600 + zone if hemisphere == "N" else 32700 + zone

        if self.crosses_utm_zones():
            zone_west = self._lon_to_utm_zone(self.west)
            zone_east = self._lon_to_utm_zone(self.east)
            warnings.warn(
                f"BBox crosses UTM zones {zone_west}-{zone_east}. "
                f"Using zone {zone} (centroid). Edge distortion may occur.",
                UserWarning
            )

        return f"EPSG:{epsg}"

    @classmethod
    def from_tuple(cls, t: tuple) -> BBox:
        return cls(west=t[0], south=t[1], east=t[2], north=t[3])


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")

    def to_stac_string(self) -> str:
        return f"{self.start.isoformat()}/{self.end.isoformat()}"


@dataclass
class SearchParams:
    bbox: BBox
    time_range: TimeRange
    collections: List[str] = field(default_factory=list)
    query: Dict[str, Any] = field(default_factory=dict)
    limit: int = 500

    def with_query(self, **filters) -> SearchParams:
        return SearchParams(
            bbox=self.bbox,
            time_range=self.time_range,
            collections=self.collections,
            query={**self.query, **filters},
            limit=self.limit
        )


class STACProvider(ABC):
    @property
    @abstractmethod
    def catalog_url(self) -> str: ...

    @property
    @abstractmethod
    def default_collections(self) -> List[str]: ...

    @property
    @abstractmethod
    def cloud_mask_band(self) -> Optional[str]: ...

    @abstractmethod
    def get_asset_mapping(self) -> Dict[str, str]: ...

    @abstractmethod
    def compute_cloud_mask(self, cloud_band: xr.DataArray) -> xr.DataArray: ...

    def get_client(self) -> STACClient:
        from requests import Session
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        # Transport-level retry for transient HTTP errors (502, 503, 504)
        retry_strategy = Retry(
            total=_STAC_MAX_RETRIES,
            backoff_factor=_STAC_BACKOFF_BASE,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session = Session()
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))

        return STACClient.open(
            self.catalog_url,
            stac_io=None,
            headers={"Accept": "application/geo+json"},
        )

    def sign_item(self, item: Item) -> Item:
        return item


class Element84Sentinel2(STACProvider):
    catalog_url = "https://earth-search.aws.element84.com/v1"
    default_collections = ["sentinel-2-l2a"]
    cloud_mask_band = "scl"

    _ASSET_MAP = {
        "red": "red", "green": "green", "blue": "blue",
        "nir": "nir", "nir08": "nir08", "nir09": "nir09",
        "swir16": "swir16", "swir22": "swir22",
        "scl": "scl", "coastal": "coastal",
    }

    def get_asset_mapping(self) -> Dict[str, str]:
        return self._ASSET_MAP

    def compute_cloud_mask(self, scl: xr.DataArray) -> xr.DataArray:
        # SCL: 4=vegetation, 5=bare soil, 6=water, 7=unclassified (sometimes useful)
        return scl.isin([4, 5, 6])


class Element84Landsat(STACProvider):
    catalog_url = "https://earth-search.aws.element84.com/v1"
    default_collections = ["landsat-c2-l2"]
    cloud_mask_band = "qa"

    _ASSET_MAP = {
        "red": "red", "green": "green", "blue": "blue",
        "nir": "nir08", "swir16": "swir16", "swir22": "swir22",
        "qa": "qa_pixel",
    }

    def get_asset_mapping(self) -> Dict[str, str]:
        return self._ASSET_MAP

    def compute_cloud_mask(self, qa: xr.DataArray) -> xr.DataArray:
        qa_int = qa.astype(np.uint16)
        cloud_bit = 1 << 3
        shadow_bit = 1 << 4
        return ((qa_int & cloud_bit) == 0) & ((qa_int & shadow_bit) == 0)


class CopernicusSentinel2(STACProvider):
    catalog_url = "https://catalogue.dataspace.copernicus.eu/stac"
    default_collections = ["sentinel-2-l2a"]
    cloud_mask_band = "scl"

    _ASSET_MAP = {
        "red": "B04", "green": "B03", "blue": "B02",
        "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12",
        "scl": "SCL",
    }

    def get_asset_mapping(self) -> Dict[str, str]:
        return self._ASSET_MAP

    def compute_cloud_mask(self, scl: xr.DataArray) -> xr.DataArray:
        return scl.isin([4, 5, 6])


class PlanetaryComputerSentinel2(STACProvider):
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    default_collections = ["sentinel-2-l2a"]
    cloud_mask_band = "scl"

    _ASSET_MAP = {
        "red": "B04", "green": "B03", "blue": "B02",
        "nir": "B08", "nir08": "B8A", "swir16": "B11",
        "scl": "SCL",
    }

    def __init__(self):
        self._pc = None

    def get_asset_mapping(self) -> Dict[str, str]:
        return self._ASSET_MAP

    def compute_cloud_mask(self, scl: xr.DataArray) -> xr.DataArray:
        return scl.isin([4, 5])

    def sign_item(self, item: Item) -> Item:
        if self._pc is None:
            try:
                import planetary_computer as pc
                self._pc = pc
            except ImportError:
                return item
        return self._pc.sign(item)


PROVIDERS: Dict[str, type] = {
    "element84-s2": Element84Sentinel2,
    "element84-landsat": Element84Landsat,
    "copernicus-s2": CopernicusSentinel2,
    "planetary-s2": PlanetaryComputerSentinel2,
}


def get_provider(name: str) -> STACProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]()


@dataclass
class LoadConfig:
    resolution_m: int = 10
    crs: Optional[str] = None
    chunks: Dict[str, int] = field(default_factory=lambda: {"x": 1024, "y": 1024, "time": 1})
    groupby: str = "solar_day"

    def resolve_crs(self, bbox: BBox) -> str:
        if self.crs:
            return self.crs
        return bbox.estimate_utm_crs()


@dataclass
class LoadResult:
    data: xr.Dataset
    provider: STACProvider
    crs: str
    bbox: BBox

    def apply_cloud_mask(self) -> xr.Dataset:
        mask_band = self.provider.cloud_mask_band
        if mask_band is None:
            logger.warning(f"Provider {type(self.provider).__name__} has no cloud mask band")
            return self.data
        if mask_band not in self.data:
            logger.warning(f"Cloud mask band '{mask_band}' not in dataset. Load it explicitly.")
            return self.data

        mask = self.provider.compute_cloud_mask(self.data[mask_band])
        return self.data.where(mask)


def _search_with_retry(client, collections, bbox, datetime_str, query, limit) -> List[Item]:
    """
    STAC search with application-level retry for catalog-specific errors
    that aren't caught by the transport retry (e.g. timeout, partial response).
    """
    last_error = None

    for attempt in range(_STAC_MAX_RETRIES):
        try:
            search = client.search(
                collections=collections,
                bbox=bbox,
                datetime=datetime_str,
                query=query if query else None,
                limit=limit,
            )
            items = list(search.items())
            return items

        except Exception as e:
            last_error = e
            wait = _STAC_BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                f"STAC search attempt {attempt + 1}/{_STAC_MAX_RETRIES} failed: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"STAC search failed after {_STAC_MAX_RETRIES} attempts: {last_error}")


class DataCube:
    def __init__(self, provider: Union[str, STACProvider] = "element84-s2"):
        if isinstance(provider, str):
            self.provider = get_provider(provider)
        else:
            self.provider = provider
        self._client = None

    @property
    def client(self) -> STACClient:
        if self._client is None:
            self._client = self.provider.get_client()
        return self._client

    def search(self, params: SearchParams) -> List[Item]:
        collections = params.collections or self.provider.default_collections

        items_raw = _search_with_retry(
            self.client,
            collections=collections,
            bbox=params.bbox.to_tuple(),
            datetime_str=params.time_range.to_stac_string(),
            query=params.query,
            limit=params.limit,
        )

        items = [self.provider.sign_item(item) for item in items_raw]
        logger.info(f"Found {len(items)} items from {self.provider.catalog_url}")
        return items

    def load(
        self,
        params: SearchParams,
        bands: List[str],
        config: LoadConfig = None
    ) -> LoadResult:
        config = config or LoadConfig()

        items = self.search(params)
        if not items:
            logger.warning("No items found for query")
            return LoadResult(
                data=xr.Dataset(),
                provider=self.provider,
                crs="",
                bbox=params.bbox
            )

        target_crs = config.resolve_crs(params.bbox)
        logger.info(f"CRS: {target_crs}, resolution: {config.resolution_m}m, items: {len(items)}")

        asset_map = self.provider.get_asset_mapping()
        provider_bands = [asset_map.get(b, b) for b in bands]

        from odc.stac import load as odc_load

        ds = odc_load(
            items,
            bands=provider_bands,
            bbox=params.bbox.to_tuple(),
            resolution=config.resolution_m,
            crs=target_crs,
            groupby=config.groupby,
            chunks=config.chunks,
            stac_cfg={"*": {"warnings": "ignore"}}
        )

        # Rename provider-specific band names to generic names
        reverse_map = {v: k for k, v in asset_map.items() if k in bands}
        rename_dict = {b: reverse_map.get(b, b) for b in ds.data_vars}
        ds = ds.rename(rename_dict)

        logger.info(f"Loaded: dims={dict(ds.dims)}, vars={list(ds.data_vars)}")

        return LoadResult(
            data=ds,
            provider=self.provider,
            crs=target_crs,
            bbox=params.bbox
        )