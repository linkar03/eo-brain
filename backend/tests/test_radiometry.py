"""
Radiometry: digital numbers to surface reflectance, scene by scene (core.radiometry).

Offline and pure numpy / xarray: no catalogue, no raster reads. The item
metadata below copies the fields of real items (Earth Search and Planetary
Computer scene S2B 32TPP 2026-10-29, Earth Search Landsat
LC09_L2SP_192030_20261119_02_T2); the pixel checks against the real archives
are in the README, "Radiometry".

Every expected NDVI is computed in float64 from the reflectance the DNs were
built from, so each test checks the whole chain: rule from the metadata,
nodata, scale, offset, floor, per-acquisition index, composite.
"""

from datetime import date, datetime, timezone
from itertools import groupby
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from eo_brain.algorithms.base import (
    FloodDetectionAlgorithm, MNDWIAlgorithm, NDVIAlgorithm, NDWIAlgorithm,
)
from eo_brain.core import radiometry as rad
from eo_brain.core.coverage import summarize_coverage
from eo_brain.core.radiometry import (
    LANDSAT_C2_L2_SR, REFLECTANCE_UNITS, S2_L2A_BOA_OFFSET, S2_L2A_DN, S2_L2A_OFFSET_REMOVED,
    RadiometryUndetermined, attach_radiometry, band_to_reflectance, landsat_c2_l2,
    no_usable_scene_message, radiometry_groups, refused_note, s2_processing_baseline,
    sentinel2_earth_search, sentinel2_esa, split_by_radiometry, summarize, to_reflectance,
)

TOL = 1e-6           # float32 arithmetic on values of order 1
LON = 11.78          # Casentino


# ---------------------------------------------------------------- item metadata

def _dt(day="2026-10-29", hour=10):
    return datetime.fromisoformat(f"{day}T{hour:02d}:16:00").replace(tzinfo=timezone.utc)


def es_item(id="S2B_32TPP_20261029_0_L2A", pb="05.13", flag=True, day="2026-10-29"):
    """Earth Search sentinel-2-l2a. Its assets always say offset -0.1 (the trap)."""
    props = {"s2:processing_baseline": pb, "earthsearch:boa_offset_applied": flag}
    if pb is None:
        del props["s2:processing_baseline"]
    if flag is None:
        del props["earthsearch:boa_offset_applied"]
    band = {"raster:bands": [{"nodata": 0, "data_type": "uint16", "scale": 0.0001, "offset": -0.1}]}
    return SimpleNamespace(id=id, properties=props, datetime=_dt(day),
                           assets={"red": band, "nir": band, "green": band})


def pc_item(id="S2B_MSIL2A_20261029T101049_R022_T32TPP_20261029T125212", pb="05.13", day="2026-10-29"):
    """Planetary Computer sentinel-2-l2a: ESA's DNs, baseline in s2:processing_baseline."""
    props = {"s2:processing_baseline": pb} if pb is not None else {}
    return SimpleNamespace(id=id, properties=props, datetime=_dt(day), assets={})


def landsat_item(number="02", correction="L2SP", id="LC09_L2SP_192030_20261119_20261120_02_T2"):
    props = {"landsat:collection_number": number, "landsat:correction": correction}
    return SimpleNamespace(id=id, properties=props, datetime=_dt("2026-11-19"), assets={})


# ---------------------------------------------------------------- synthetic bands

def dn_for(rule, refl):
    """Integer DNs that hold reflectance ``refl`` under ``rule`` (exactly)."""
    dn = (np.asarray(refl, dtype=np.float64) - rule.offset) / rule.scale
    out = np.rint(dn)
    assert np.allclose(out, dn, atol=1e-6), "pick reflectances that are whole DNs"
    return out.astype(np.uint16)


def cube(slices, rules, names=("red", "nir")):
    """Dataset (time, y, x) of DN bands, one slice per entry, with the rules attached."""
    data = {
        name: xr.DataArray(np.stack([s[name] for s in slices]), dims=("time", "y", "x"))
        for name in names
    }
    ds = xr.Dataset(data, coords={"time": np.arange(len(slices))})
    return attach_radiometry(ds, rules)


def ndvi64(red, nir):
    red, nir = np.asarray(red, np.float64), np.asarray(nir, np.float64)
    return (nir - red) / (nir + red)


# Reflectances that are whole DNs for every Sentinel-2 rule (steps of 1e-4).
S2_RED = np.array([[0.0300, 0.0500, 0.1200]])
S2_NIR = np.array([[0.2500, 0.4500, 0.3000]])
# Landsat: DN 8000/10000/12000 and 16000/20000/24000.
LS_RED = np.array([[0.02, 0.075, 0.13]])
LS_NIR = np.array([[0.24, 0.35, 0.46]])


# ---------------------------------------------------------------- rules per scene

class TestProcessingBaseline:
    def test_from_property(self):
        assert s2_processing_baseline(pc_item(pb="05.13")) == (5, 13)
        assert s2_processing_baseline(pc_item(pb="04.00")) == (4, 0)

    def test_from_processing_version(self):
        """Copernicus Data Space items carry processing:version."""
        item = SimpleNamespace(id="x", properties={"processing:version": "05.11"})
        assert s2_processing_baseline(item) == (5, 11)

    def test_from_product_name(self):
        uri = "S2B_MSIL2A_20210707T101029_N0301_R022_T32TPP_20210707T131521.SAFE"
        assert s2_processing_baseline(SimpleNamespace(id="x", properties={"s2:product_uri": uri})) == (3, 1)
        assert s2_processing_baseline(SimpleNamespace(id=uri, properties={})) == (3, 1)

    def test_unparseable_value_falls_back_to_the_name(self):
        item = SimpleNamespace(id="S2A_MSIL2A_20220301T101021_N0400_R022_T32TPP_X",
                               properties={"s2:processing_baseline": "n/a"})
        assert s2_processing_baseline(item) == (4, 0)

    def test_absent(self):
        assert s2_processing_baseline(SimpleNamespace(id="S2B_32TPP_20261029_0_L2A", properties={})) is None


class TestProviderRules:
    def test_earth_search_flag_true_offset_removed(self):
        assert sentinel2_earth_search(es_item()) is S2_L2A_OFFSET_REMOVED
        assert S2_L2A_OFFSET_REMOVED.offset == 0.0 and S2_L2A_OFFSET_REMOVED.scale == 1e-4

    def test_earth_search_pre_pb04_plain_dn(self):
        """The 2021 product of 32TPP on Earth Search: baseline 03.01, flag false, no offset in the DNs."""
        item = es_item(id="S2B_32TPP_20210707_0_L2A", pb="03.01", flag=False, day="2021-07-07")
        assert sentinel2_earth_search(item) is S2_L2A_DN

    @pytest.mark.parametrize("flag", [False, None])
    def test_earth_search_pb04_without_flag_true_is_undetermined(self, flag):
        with pytest.raises(RadiometryUndetermined, match="#66, #71"):
            sentinel2_earth_search(es_item(pb="05.00", flag=flag))

    def test_earth_search_flag_true_but_no_baseline_is_undetermined(self):
        with pytest.raises(RadiometryUndetermined, match="processing baseline"):
            sentinel2_earth_search(es_item(pb=None, flag=True))

    @pytest.mark.parametrize("pb,rule", [
        ("02.14", S2_L2A_DN), ("03.01", S2_L2A_DN), ("04.00", S2_L2A_BOA_OFFSET),
        ("05.10", S2_L2A_BOA_OFFSET), ("05.13", S2_L2A_BOA_OFFSET),
    ])
    def test_esa_offset_from_baseline_04_00(self, pb, rule):
        assert sentinel2_esa(pc_item(pb=pb)) is rule

    def test_esa_baseline_not_date(self):
        """A 2018 acquisition reprocessed with baseline 05.00 carries the offset."""
        item = pc_item(id="S2A_MSIL2A_20180701T101031_N0500_R022_T32TPP_20230712T000000", pb=None, day="2018-07-01")
        assert sentinel2_esa(item) is S2_L2A_BOA_OFFSET

    def test_esa_no_baseline_is_undetermined(self):
        with pytest.raises(RadiometryUndetermined, match="processing baseline"):
            sentinel2_esa(pc_item(id="S2B_32TPP_20261029", pb=None))

    @pytest.mark.parametrize("correction", ["L2SP", "L2SR"])
    def test_landsat_collection2_level2(self, correction):
        rule = landsat_c2_l2(landsat_item(correction=correction))
        assert rule is LANDSAT_C2_L2_SR
        assert (rule.scale, rule.offset, rule.nodata) == (2.75e-5, -0.2, 0)

    @pytest.mark.parametrize("number,correction", [("01", "L2SP"), ("02", "L1TP"), ("", "")])
    def test_landsat_other_products_are_undetermined(self, number, correction):
        with pytest.raises(RadiometryUndetermined, match="Collection 2 Level-2"):
            landsat_c2_l2(landsat_item(number=number, correction=correction))


# ---------------------------------------------------------------- one conversion

class TestExactNDVI:
    """Synthetic DNs of known reflectance give the exact NDVI, for each rule."""

    @pytest.mark.parametrize("rule,red,nir", [
        (S2_L2A_DN, S2_RED, S2_NIR),
        (S2_L2A_BOA_OFFSET, S2_RED, S2_NIR),
        (S2_L2A_OFFSET_REMOVED, S2_RED, S2_NIR),
        (LANDSAT_C2_L2_SR, LS_RED, LS_NIR),
    ], ids=lambda v: getattr(v, "rule", None))
    def test_exact_ndvi(self, rule, red, nir):
        ds = cube([{"red": dn_for(rule, red), "nir": dn_for(rule, nir)}], [rule])
        out = NDVIAlgorithm().run(ds).data["ndvi"].values
        np.testing.assert_allclose(out, ndvi64(red, nir), atol=TOL, rtol=0)

    def test_landsat_dns(self):
        """The Landsat values above are the USGS formula on whole DNs."""
        assert dn_for(LANDSAT_C2_L2_SR, LS_RED).tolist() == [[8000, 10000, 12000]]
        np.testing.assert_allclose(
            to_reflectance(np.array([7273, 43636, 0]), 2.75e-5, -0.2, 0),
            [2.75e-5, 1.0, np.nan], atol=1e-4,   # USGS valid range 7273-43636 -> ~0 .. ~1
        )

    @pytest.mark.parametrize("algo,var,bands", [
        (NDWIAlgorithm, "ndwi", ("green", "nir")),
        (MNDWIAlgorithm, "mndwi", ("green", "swir16")),
    ])
    def test_water_indices_use_the_same_conversion(self, algo, var, bands):
        green, other = np.array([[0.0600, 0.0450]]), np.array([[0.0200, 0.2000]])
        ds = cube([{bands[0]: dn_for(S2_L2A_BOA_OFFSET, green), bands[1]: dn_for(S2_L2A_BOA_OFFSET, other)}],
                  [S2_L2A_BOA_OFFSET], names=bands)
        out = algo().run(ds).data[var].values
        np.testing.assert_allclose(out, (green - other) / (green + other), atol=TOL, rtol=0)

    def test_flood_index_on_offset_scene(self):
        green, nir, swir = np.array([[0.0600]]), np.array([[0.0300]]), np.array([[0.0200]])
        r = S2_L2A_BOA_OFFSET
        ds = cube([{"green": dn_for(r, green), "nir": dn_for(r, nir), "swir16": dn_for(r, swir)}],
                  [r], names=("green", "nir", "swir16"))
        res = FloodDetectionAlgorithm().run(ds)
        assert res.metadata["index_used"] == "mndwi"
        np.testing.assert_allclose(res.data["water_index"].values, (green - swir) / (green + swir), atol=TOL)


class TestMixedComposite:
    """One composite from scenes with and without the offset applied."""

    def test_median_over_three_rules(self):
        # Three acquisitions of the same pixels; their true NDVI is 0.5, 0.6, 0.8.
        # The middle one is an ESA product with the offset in its DNs: read with
        # the wrong rule it would move, and the median with it.
        red = np.array([[0.1000, 0.0500]])
        nir_for = lambda v: red * (1 + v) / (1 - v)
        truths = [0.5, 0.6, 0.8]
        rules = [S2_L2A_OFFSET_REMOVED, S2_L2A_BOA_OFFSET, S2_L2A_DN]
        slices = [{"red": dn_for(r, red), "nir": dn_for(r, nir_for(v))} for r, v in zip(rules, truths)]
        out = NDVIAlgorithm().run(cube(slices, rules)).data["ndvi"].values
        np.testing.assert_allclose(out, 0.6, atol=TOL, rtol=0)

        # The same DNs with one rule for all (what a per-provider constant would do)
        wrong = cube(slices, [S2_L2A_DN] * 3)
        assert not np.allclose(NDVIAlgorithm().run(wrong).data["ndvi"].values, 0.6, atol=1e-3)

    def test_same_scene_from_two_providers_agrees(self):
        """Earth Search (offset removed) and Planetary Computer (offset in DNs) DNs of one scene."""
        es = cube([{"red": dn_for(S2_L2A_OFFSET_REMOVED, S2_RED), "nir": dn_for(S2_L2A_OFFSET_REMOVED, S2_NIR)}],
                  [S2_L2A_OFFSET_REMOVED])
        pc = cube([{"red": dn_for(S2_L2A_BOA_OFFSET, S2_RED), "nir": dn_for(S2_L2A_BOA_OFFSET, S2_NIR)}],
                  [S2_L2A_BOA_OFFSET])
        assert (pc["red"].values.astype(int) - es["red"].values.astype(int) == 1000).all()
        np.testing.assert_array_equal(NDVIAlgorithm().run(es).data["ndvi"].values,
                                      NDVIAlgorithm().run(pc).data["ndvi"].values)


class TestDoubleSubtractionTrap:
    """Earth Search assets say offset -0.1 even where the flag says the offset is already removed."""

    def test_rule_ignores_asset_offset(self):
        item = es_item()
        assert item.assets["red"]["raster:bands"][0]["offset"] == -0.1
        rule = sentinel2_earth_search(item)
        assert rule.offset == 0.0

    def test_ndvi_is_the_true_one(self):
        item = es_item()
        rule = sentinel2_earth_search(item)
        red_dn, nir_dn = dn_for(rule, S2_RED), dn_for(rule, S2_NIR)
        out = NDVIAlgorithm().run(cube([{"red": red_dn, "nir": nir_dn}], [rule])).data["ndvi"].values
        np.testing.assert_allclose(out, ndvi64(S2_RED, S2_NIR), atol=TOL)

        # What trusting the asset metadata gives: every reflectance 0.1 lower.
        meta = item.assets["red"]["raster:bands"][0]
        trapped = cube([{"red": red_dn, "nir": nir_dn}],
                       [rad.Radiometry("asset-metadata", meta["scale"], meta["offset"], 0, "")])
        bad = NDVIAlgorithm().run(trapped).data["ndvi"].values
        assert np.abs(bad - ndvi64(S2_RED, S2_NIR)).max() > 0.1


class TestNodata:
    def test_nodata_masked_before_scaling(self):
        """DN 0 is NO_DATA: NaN, not (0 - 1000) / 10000 raised to the floor."""
        out = to_reflectance(np.array([0, 1000, 1500], dtype=np.uint16), 1e-4, -0.1, 0)
        assert np.isnan(out[0])
        assert out[1] == np.float32(1e-4) and out[2] == pytest.approx(0.05, abs=TOL)

    def test_nodata_pixel_left_to_other_acquisitions(self):
        r = S2_L2A_BOA_OFFSET
        red, nir = dn_for(r, S2_RED), dn_for(r, S2_NIR)
        hole = red.copy()
        hole[0, 1] = 0
        both_empty = nir.copy()
        both_empty[0, 2] = 0
        ds = cube([{"red": hole, "nir": nir}, {"red": red, "nir": both_empty}], [r, r])
        per_slice = NDVIAlgorithm._compute_ndvi(ds).values
        assert np.isnan(per_slice[0, 0, 1]) and np.isnan(per_slice[1, 0, 2])
        out = NDVIAlgorithm().run(ds).data["ndvi"].values
        np.testing.assert_allclose(out, ndvi64(S2_RED, S2_NIR), atol=TOL)

    def test_pixel_with_no_data_anywhere_stays_empty(self):
        r = S2_L2A_OFFSET_REMOVED
        red = dn_for(r, S2_RED)
        red[0, 0] = 0
        out = NDVIAlgorithm().run(cube([{"red": red, "nir": dn_for(r, S2_NIR)}], [r])).data["ndvi"].values
        assert np.isnan(out[0, 0]) and np.isfinite(out[0, 1:]).all()

    def test_cloud_masked_float_band_is_still_converted(self):
        """After the cloud mask the DNs are float with NaN: same conversion (the old float shortcut is gone)."""
        r = S2_L2A_BOA_OFFSET
        ds = cube([{"red": dn_for(r, S2_RED), "nir": dn_for(r, S2_NIR)}], [r])
        masked = ds.where(xr.DataArray([[[True, True, False]]], dims=("time", "y", "x")))
        assert masked["red"].dtype.kind == "f"
        red = band_to_reflectance(masked["red"]).values
        np.testing.assert_allclose(red[0, 0, :2], S2_RED[0, :2], atol=TOL)
        assert np.isnan(red[0, 0, 2])
        out = NDVIAlgorithm().run(masked).data["ndvi"].values
        np.testing.assert_allclose(out[0, :2], ndvi64(S2_RED, S2_NIR)[0, :2], atol=TOL)


class TestNonPositiveReflectance:
    def test_floor_is_one_dn_step(self):
        out = to_reflectance(np.array([500, 999, 1000, 1001], dtype=np.uint16), 1e-4, -0.1, 0)
        np.testing.assert_array_equal(out, np.float32([1e-4, 1e-4, 1e-4, 1e-4]))
        assert out.dtype == np.float32

    def test_matches_earth_search_floor(self):
        """Where ESA's DN is below 1000 Earth Search stores DN 1: both give the same reflectance."""
        pc = to_reflectance(np.array([747, 987, 999]), 1e-4, -0.1, 0)
        es = to_reflectance(np.array([1, 1, 1]), 1e-4, 0.0, 0)
        np.testing.assert_array_equal(pc, es)

    def test_dark_water_stays_water(self):
        """A water pixel with green just above and NIR below the offset: kept, strongly positive NDWI."""
        r = S2_L2A_BOA_OFFSET
        ds = cube([{"green": np.array([[1300]], np.uint16), "nir": np.array([[940]], np.uint16)}],
                  [r], names=("green", "nir"))
        out = NDWIAlgorithm().run(ds).data["ndwi"].values
        np.testing.assert_allclose(out, (0.03 - 1e-4) / (0.03 + 1e-4), atol=TOL)

    def test_above_one_is_kept(self):
        assert to_reflectance(np.array([12000]), 1e-4, 0.0, 0)[0] == pytest.approx(1.2, abs=TOL)


class TestDeclaredRadiometryOnly:
    def test_band_without_rule_is_refused(self):
        ds = xr.Dataset({"red": xr.DataArray(np.full((2, 2), 500, np.uint16), dims=("y", "x")),
                         "nir": xr.DataArray(np.full((2, 2), 4500, np.uint16), dims=("y", "x"))})
        with pytest.raises(RadiometryUndetermined, match="no declared radiometry"):
            NDVIAlgorithm().run(ds)
        assert issubclass(RadiometryUndetermined, ValueError)   # the API reports it as a compute error

    def test_band_marked_as_reflectance_is_used_as_is(self):
        attrs = {"units": REFLECTANCE_UNITS}
        ds = xr.Dataset({"red": xr.DataArray(S2_RED, dims=("y", "x"), attrs=attrs),
                         "nir": xr.DataArray(S2_NIR, dims=("y", "x"), attrs=attrs)})
        np.testing.assert_allclose(NDVIAlgorithm().run(ds).data["ndvi"].values, ndvi64(S2_RED, S2_NIR), atol=TOL)

    def test_lazy_input_stays_lazy(self):
        r = S2_L2A_BOA_OFFSET
        ds = cube([{"red": dn_for(r, S2_RED), "nir": dn_for(r, S2_NIR)}], [r])
        ds = ds.chunk({"time": 1})
        refl = band_to_reflectance(ds["red"])
        assert isinstance(refl.data, da.Array)
        assert refl.attrs["units"] == REFLECTANCE_UNITS and "nodata" not in refl.attrs
        np.testing.assert_allclose(refl.values[0], S2_RED, atol=TOL)


# ---------------------------------------------------------------- slices and audit

def _odc_groups(items, group):
    """odc-stac's grouping: sort by (key, time, id), then group equal keys."""
    order = sorted(items, key=lambda it: (group(it, None, 0), it.datetime, it.id))
    return [list(g) for _, g in groupby(order, key=lambda it: group(it, None, 0))]


class TestSlices:
    def test_homogeneous_days_are_solar_days(self):
        items = [es_item(id="a", day="2026-10-29"), es_item(id="b", day="2026-10-29"),
                 es_item(id="c", day="2026-11-03")]
        rules = {it.id: sentinel2_earth_search(it) for it in items}
        group, slice_rules = radiometry_groups(items, rules, LON)
        groups = _odc_groups(items, group)
        assert [[it.id for it in g] for g in groups] == [["a", "b"], ["c"]]
        assert slice_rules == [S2_L2A_OFFSET_REMOVED, S2_L2A_OFFSET_REMOVED]

    def test_mixed_rule_day_is_split_in_key_order(self):
        items = [pc_item(id="new", pb="05.00", day="2021-07-07"), pc_item(id="old", pb="03.01", day="2021-07-07"),
                 pc_item(id="later", pb="05.13", day="2021-07-12")]
        rules = {it.id: sentinel2_esa(it) for it in items}
        group, slice_rules = radiometry_groups(items, rules, LON)
        groups = _odc_groups(items, group)
        assert len(groups) == len(slice_rules) == 3
        for g, r in zip(groups, slice_rules):
            assert {rules[it.id] for it in g} == {r}

    def test_attach_checks_slice_count(self):
        ds = cube([{"red": dn_for(S2_L2A_DN, S2_RED), "nir": dn_for(S2_L2A_DN, S2_NIR)}], [S2_L2A_DN])
        with pytest.raises(ValueError, match="2 radiometry slices"):
            attach_radiometry(ds, [S2_L2A_DN, S2_L2A_DN])
        assert ds[rad.OFFSET].dtype == np.float64 and ds[rad.RULE].values.tolist() == ["s2-l2a-pre-pb04"]


class TestAudit:
    def test_undetermined_scenes_are_left_out_and_listed(self):
        provider = SimpleNamespace(radiometry=sentinel2_earth_search)
        items = [es_item(id="ok"), es_item(id="flag-false", flag=False), es_item(id="no-flag", flag=None)]
        usable, refused = split_by_radiometry(provider, items)
        assert [it.id for it in usable] == ["ok"]
        assert [r["id"] for r in refused] == ["flag-false", "no-flag"]
        assert "boa_offset_applied False" in refused[0]["reason"]
        assert "boa_offset_applied missing" in refused[1]["reason"]
        assert refused_note(refused).startswith("2 scenes were left out")
        assert refused_note(refused[:1]).startswith("1 scene was left out")
        assert refused_note([]) is None
        assert no_usable_scene_message(refused).startswith("No usable scene: the radiometry of all 2 scenes")

    def test_summary(self):
        scene_rules = {"a": S2_L2A_OFFSET_REMOVED, "b": S2_L2A_OFFSET_REMOVED, "c": S2_L2A_DN}
        refused = [{"id": f"r{i}", "reason": "x"} for i in range(25)]
        block = summarize("element84-s2", scene_rules,
                          ["s2-l2a-offset-removed", "s2-l2a-pre-pb04", "s2-l2a-offset-removed"], refused)
        assert block["provider"] == "element84-s2"
        assert block["asset_scale_offset_used"] is False
        assert block["non_positive"] == rad.NON_POSITIVE_POLICY
        by_rule = {r["rule"]: r for r in block["rules"]}
        assert by_rule["s2-l2a-offset-removed"]["scenes"] == 2
        assert by_rule["s2-l2a-offset-removed"]["acquisition_slices"] == 2
        assert by_rule["s2-l2a-offset-removed"]["offset"] == 0.0
        assert by_rule["s2-l2a-pre-pb04"] == {**S2_L2A_DN.as_dict(), "scenes": 1, "acquisition_slices": 1}
        assert block["scenes_refused"] == 25 and len(block["refused"]) == 20

    def test_coverage_counts_a_split_day_once(self):
        d1, d2 = date(2021, 7, 7), date(2021, 7, 12)
        valid = np.ones((2, 2), bool)
        cov = summarize_coverage(valid, valid, valid, [d1, d1, d2], [3, 1, 0])
        assert cov["acquisition_days_loaded"] == 2
        assert cov["acquisition_days_with_data"] == 1
        assert cov["first_date_with_data"] == cov["last_date_with_data"] == "2021-07-07"
