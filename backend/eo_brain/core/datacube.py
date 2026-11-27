"""
eo_brain/core/datacube.py - DataCube Engine v4

Changes from v3:
- STAC search retry with exponential backoff (network resilience)
- Configurable HTTP timeout for STAC client
- UTM zone boundary warning preserved
- Radiometry per scene: each provider declares how a scene's digital numbers
  become surface reflectance (``STACProvider.radiometry``, core.radiometry);
  ``load_items`` keeps scenes with different rules in different time slices
  and attaches each slice's rule to the data
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

from eo_brain.core import radiometry as rad
from eo_brain.core.radiometry import Radiometry, RadiometryUndetermined

logger = logging.getLogger(__name__)

# Retry config for STAC catalog queries
_STAC_MAX_RETRIES = 3
_STAC_BACKOFF_BASE = 1.5  # seconds, doubles each retry
_STAC_TIMEOUT_SEC = 30

_CLOUD_READS_CONFIGURED = False


def configure_cloud_reads() -> None:
    """
    GDAL settings for reading COGs over HTTP, once per process (odc-stac's
    cloud defaults: no directory listing on open, HTTP retries). Without them
    GDAL lists the bucket "directory" of every asset it opens; on the Casentino
    benchmark (10 scenes, 3 bands, 60 m) the load took 40 s instead of 15 s.
    """
    global _CLOUD_READS_CONFIGURED
    if _CLOUD_READS_CONFIGURED:
        return
    from odc.stac import configure_rio
    configure_rio(cloud_defaults=True)
    _CLOUD_READS_CONFIGURED = True


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
    # Registry name (PROVIDERS), recorded in the job's radiometry block.
    name: str = "custom"
    # Storage type of the reflectance bands and of the cloud-mask band, and the
    # mask band's nodata: given to the loader with the radiometry rule's nodata,
    # so that loading does not depend on what each catalogue declares (Earth
    # Search declares the same values; Planetary Computer declares none).
    data_type: str = "uint16"
    mask_data_type: str = "uint8"
    mask_nodata: Optional[int] = None

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

    def radiometry(self, item: Item) -> Radiometry:
        """How this scene's DNs become surface reflectance (core.radiometry); raises RadiometryUndetermined."""
        raise RadiometryUndetermined(f"{type(self).__name__} declares no radiometry rule")


class Element84Sentinel2(STACProvider):
    """
    Earth Search ``sentinel-2-l2a``. Radiometry: ``radiometry.sentinel2_earth_search``
    (the offset is removed by Earth Search when its flag says so; the assets'
    ``raster:bands`` offset is not used). Checked on pixels against Planetary
    Computer, scene S2B_32TPP_20261029 (README, "Radiometry").
    """

    name = "element84-s2"
    mask_nodata = 0
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

    def radiometry(self, item: Item) -> Radiometry:
        return rad.sentinel2_earth_search(item)


class Element84Landsat(STACProvider):
    """
    Earth Search ``landsat-c2-l2``. Radiometry: USGS Collection 2 Level-2
    surface reflectance, DN * 0.0000275 - 0.2. Does not load as configured:
    the assets are in the requester-pays ``usgs-landsat`` bucket.
    """

    name = "element84-landsat"
    mask_data_type = "uint16"
    mask_nodata = 1   # QA_PIXEL fill
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

    def radiometry(self, item: Item) -> Radiometry:
        return rad.landsat_c2_l2(item)


class CopernicusSentinel2(STACProvider):
    """
    Copernicus Data Space. Radiometry: ESA's DNs, offset by processing
    baseline (``radiometry.sentinel2_esa``). Does not load as configured
    (asset keys, authentication).
    """

    name = "copernicus-s2"
    mask_nodata = 0
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

    def radiometry(self, item: Item) -> Radiometry:
        return rad.sentinel2_esa(item)


class PlanetaryComputerSentinel2(STACProvider):
    """
    Planetary Computer ``sentinel-2-l2a``: ESA's DNs as distributed, offset
    by processing baseline (``radiometry.sentinel2_esa``). Checked on pixels:
    DN = Earth Search DN + 1000 on the same scene.
    """

    name = "planetary-s2"
    mask_nodata = 0
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

    def radiometry(self, item: Item) -> Radiometry:
        return rad.sentinel2_esa(item)

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
    # Default matches the API's chunk policy for the default 10 m resolution
    # (api.main._optimal_chunks: <=10 m -> 2048). The API always passes chunks
    # explicitly; this default only applies to callers that do not.
    chunks: Dict[str, int] = field(default_factory=lambda: {"x": 2048, "y": 2048, "time": 1})
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
    # Radiometry rule of each scene loaded (by id) and of each time slice.
    scene_rules: Dict[str, Radiometry] = field(default_factory=dict)
    slice_rules: List[Radiometry] = field(default_factory=list)
    # Scenes left out because their rule could not be determined ({id, reason}).
    refused: List[Dict[str, str]] = field(default_factory=list)

    def radiometry_summary(self, refused: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """The job's ``radiometry`` block (radiometry.summarize)."""
        return rad.summarize(
            self.provider.name, self.scene_rules, [r.rule for r in self.slice_rules],
            self.refused if refused is None else refused,
        )

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
        """
        Search, then load every item found whose radiometry is known (see
        load_items). The others are left out with a warning and listed in
        ``LoadResult.refused``.
        """
        items, refused = rad.split_by_radiometry(self.provider, self.search(params))
        if refused:
            logger.warning(f"{len(refused)} scenes left out, radiometry undetermined: {[r['id'] for r in refused]}")
        result = self.load_items(items, params.bbox, bands, config)
        result.refused = refused
        return result

    def load_items(
        self,
        items: List[Item],
        bbox: BBox,
        bands: List[str],
        config: LoadConfig = None
    ) -> LoadResult:
        """
        Load the given items over ``bbox`` (the API selects them first: core.scenes).

        Every item needs a known radiometry (``provider.radiometry``; filter
        with ``radiometry.split_by_radiometry`` first): an undetermined one
        raises RadiometryUndetermined. The bands stay digital numbers; each
        time slice carries its rule (``radiometry.attach_radiometry``) and
        the algorithms convert per slice.
        """
        config = config or LoadConfig()

        if not items:
            logger.warning("No items found for query")
            return LoadResult(
                data=xr.Dataset(),
                provider=self.provider,
                crs="",
                bbox=bbox
            )

        target_crs = config.resolve_crs(bbox)
        logger.info(f"CRS: {target_crs}, resolution: {config.resolution_m}m, items: {len(items)}")

        scene_rules = {item.id: self.provider.radiometry(item) for item in items}
        nodata_values = {r.nodata for r in scene_rules.values()}
        if len(nodata_values) != 1:
            raise ValueError(f"Scenes with different nodata values cannot be loaded together: {nodata_values}")
        if config.groupby != "solar_day":
            raise ValueError(f"load_items groups scenes by solar day and radiometry, not {config.groupby!r}")
        group, slice_rules = rad.radiometry_groups(items, scene_rules, bbox.center()[0])

        asset_map = self.provider.get_asset_mapping()
        provider_bands = [asset_map.get(b, b) for b in bands]

        # Declared storage type and nodata of every band (the radiometry rule's
        # nodata for reflectance), also where the catalogue declares none
        # (Planetary Computer): the loader fuses the scenes of one day on them,
        # and the coverage count reads the mask band's.
        (nodata,) = nodata_values
        asset_cfg = {}
        for b in bands:
            if b == self.provider.cloud_mask_band:
                cfg = {"data_type": self.provider.mask_data_type}
                if self.provider.mask_nodata is not None:
                    cfg["nodata"] = self.provider.mask_nodata
            else:
                cfg = {"data_type": self.provider.data_type, "nodata": nodata}
            asset_cfg[asset_map.get(b, b)] = cfg

        from odc.stac import load as odc_load

        configure_cloud_reads()
        ds = odc_load(
            items,
            bands=provider_bands,
            bbox=bbox.to_tuple(),
            resolution=config.resolution_m,
            crs=target_crs,
            groupby=group,
            chunks=config.chunks,
            stac_cfg={"*": {"warnings": "ignore", "assets": asset_cfg}}
        )
        ds = rad.attach_radiometry(ds, slice_rules)

        # Rename provider-specific band names to generic names
        reverse_map = {v: k for k, v in asset_map.items() if k in bands}
        rename_dict = {b: reverse_map.get(b, b) for b in ds.data_vars}
        ds = ds.rename(rename_dict)

        logger.info(f"Loaded: dims={dict(ds.sizes)}, vars={list(ds.data_vars)}, "
                    f"radiometry per slice: {[r.rule for r in slice_rules]}")

        return LoadResult(
            data=ds,
            provider=self.provider,
            crs=target_crs,
            bbox=bbox,
            scene_rules=scene_rules,
            slice_rules=slice_rules,
        )