-- SYNTHETIC TEST FIXTURE - NOT REAL DATA.
-- Invented place names, hand-picked numbers. Used by the database-backed
-- tests and the text-to-SQL eval. Loaded into a throwaway database after
-- schema.sql, schema_vectors.sql, agent_views.sql and agent_role.sql.
--
-- Known answers (area_m2 values are stored as given, not derived from the
-- toy geometries):
--   analysis ...001 flood   completed  Testville North  alert 8 ha (2 polys), suspect 4 ha, watch 1 ha;
--                                                       run before zone sizes and class areas were
--                                                       recorded: old-style vector_stats, no
--                                                       severity_class_areas rows
--   analysis ...002 flood   completed  Testville North  alert 12 ha (1 poly), suspect 4.5 ha (2 polys)
--   analysis ...003 lisa    completed  Sample Plain     alert 6 ha (HL), suspect 1.5 ha (LH), watch 0.5 ha (HL)
--   analysis ...004 ndvi    completed  Testville North  no polygons (continuous index)
--   analysis ...005 ndwi    failed     Nowhere Lake     no polygons
--   analysis ...006 flood   pending    Sample Plain     no polygons
--   analysis ...007 ndvi    completed  Sample Plain     window June 2025
--   analysis ...008 ndvi    completed  Sample Plain     window December 2025
--   analysis ...009 ndwi    completed  Nowhere Lake
--   analysis ...010 mndwi   completed  Nowhere Lake
--   analysis ...011 ndwi    completed  Old Harbour      run before value distributions were
--                                                       recorded: no intervals, no value_*
--   latest (highest created_at): overall 006; completed 010; ndvi 004; Sample Plain ndvi 008;
--   ndwi 009 (005, failed, is older); mndwi 010.
--   footprints: 001 overlaps 002 and 004; 003 overlaps 006, 007 and 008 (007 and 008 have
--   the same box as 003); 005 overlaps 009 and 010; 011 overlaps nothing.
--   coverage (result.coverage, as api.main writes it; agent.analyses coverage_* columns):
--     001 97.8 %   002 100 %   003 91.2 %   004 41.5 % low coverage (cloudy 7-day window,
--     58.5 % masked by clouds, 6 of 9 scenes at least 95 % cloudy, 2 of the other 3 with
--     clear pixels, 1 acquisition day)
--     005 / 006 none (failed / pending)
--     007 99.4 %   008 96.3 %   009 97.1 %   010 88.2 %   011 93.4 % (2.5 % outside the scenes)
--   value distribution (index_value_bins; generated with core.distribution, so the pixel
--   counts add up to result.coverage.covered_px; ha = hectares of covered area):
--     004 ndvi  >= 0.65: 554.32 ha (15.0 %)    >= 0.4: 2401.99 ha    total 3695.36 ha
--     007 ndvi  >= 0.65: 4786.44 ha (55.0 %)   >= 0.4: 6787.88 ha    total 8702.32 ha
--               the 0.64 interval holds 435.08 ha: it is below 0.65
--     008 ndvi  >= 0.65: 1517.64 ha (18.0 %)   0.2 to 0.5: 3962.52 ha (0.50 interval excluded)
--     009 ndwi  >= 0:    2506.00 ha (30.0 %)   the -0.01 interval (167.04 ha) is below 0
--     010 mndwi >= 0:    2579.76 ha (34.0 %)
--   result.value_distribution (agent.analyses value_mean / value_min / value_max) for 004
--   and 007-010; 007 and 011 also carry an older-style result.summary with other numbers,
--   which the views must not read.
--   class areas (severity_class_areas, agent.severity_areas; 10 m pixels of 100 m2, pixel
--   counts adding up to result.coverage.covered_px) and zone sizes (result.vector_stats,
--   agent.analyses min_zone_* / small_patches_*) for 002 and 003 only:
--     002 flood  water (severity >= 1) 1802 px = 18.02 ha, 0.2029 % of 888120 covered px:
--                watch 0.52 ha, suspect 4.86 ha, alert 12.64 ha; the zones hold 16.5 ha;
--                zones of at least 4 px = 400 m2; 76 smaller patches (1.52 ha) left out
--     003 lisa   anomalies (severity >= 1) 1062 px = 10.62 ha, 0.1332 % of 797400 covered
--                px, 0.1893 % of 561062 vegetated px: HL 8.10 ha (watch 1.70, suspect 0,
--                alert 6.40), LH 2.52 ha (watch 0.60, suspect 1.80, alert 0.12); alert
--                6.52 ha; zones of at least 4 px = 400 m2; 131 smaller patches (2.62 ha)
--     001 (flood, older) has neither: its areas and zone sizes are not available.

INSERT INTO jobs (id, status, algorithm, provider, bbox, start_date, end_date, params,
                  progress, message, created_at, completed_at, result)
VALUES
  ('00000000-0000-4000-a000-000000000001', 'completed', 'flood', 'element84-s2',
   ST_MakeEnvelope(11.00, 44.00, 11.10, 44.10, 4326),
   '2025-12-11 09:00:00+00', '2026-01-10 09:00:00+00',
   '{"geocoded_name": "Testville North (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 97.8% of the requested area (4 acquisition days, 2025-12-14 to 2026-01-08).',
   '2026-01-10 09:00:00+00', '2026-01-10 09:01:35+00',
   '{"coverage": {"coverage_pct": 97.8, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 2.2, "uncovered_no_data_pct": 0.0, "scenes_found": 14, "scenes_too_cloudy": 3, "scenes_checked": 11, "scenes_with_clear_pixels": 9, "scenes_used": 4, "acquisition_days_loaded": 4, "acquisition_days_with_data": 4, "first_date_with_data": "2025-12-14", "last_date_with_data": "2026-01-08"}, "vector_stats": {"feature_count": 4, "total_area_m2": 130000.0, "by_severity": {"1": 1, "2": 1, "3": 2}}}'),
  ('00000000-0000-4000-a000-000000000002', 'completed', 'flood', 'element84-s2',
   ST_MakeEnvelope(11.02, 44.02, 11.12, 44.12, 4326),
   '2025-10-14 10:00:00+00', '2026-01-12 10:00:00+00',
   '{"geocoded_name": "Testville North (synthetic)", "resolution_m": 10, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 100% of the requested area (5 acquisition days, 2025-10-20 to 2026-01-04).',
   '2026-01-12 10:00:00+00', '2026-01-12 10:02:00+00',
   '{"coverage": {"coverage_pct": 100.0, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 0.0, "uncovered_no_data_pct": 0.0, "area_px": 888120, "covered_px": 888120, "scenes_found": 41, "scenes_too_cloudy": 6, "scenes_checked": 35, "scenes_with_clear_pixels": 30, "scenes_used": 5, "acquisition_days_loaded": 5, "acquisition_days_with_data": 5, "first_date_with_data": "2025-10-20", "last_date_with_data": "2026-01-04"}, "vector_stats": {"feature_count": 3, "total_area_m2": 165000.0, "by_severity": {"1": 0, "2": 2, "3": 1}, "min_zone_pixels": 4, "min_zone_area_m2": 400.0, "skipped_small": 76, "skipped_small_area_m2": 15200.0, "skipped_invalid": 0, "max_features_reached": false}, "severity_areas": {"algorithm": "flood", "pixels": 888120, "pixel_area_m2": 100.0, "area_m2": 88812000.0, "pixels_by_severity": {"0": 886318, "1": 52, "2": 486, "3": 1264}}}'),
  ('00000000-0000-4000-a000-000000000003', 'completed', 'lisa', 'element84-s2',
   ST_MakeEnvelope(12.00, 45.00, 12.10, 45.10, 4326),
   '2025-01-11 08:00:00+00', '2026-01-11 08:00:00+00',
   '{"geocoded_name": "Sample Plain (synthetic)", "resolution_m": 10, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 91.2% of the requested area (6 acquisition days, 2025-02-17 to 2025-12-28).',
   '2026-01-11 08:00:00+00', '2026-01-11 08:03:20+00',
   '{"coverage": {"coverage_pct": 91.2, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 8.8, "uncovered_no_data_pct": 0.0, "area_px": 874200, "covered_px": 797400, "scenes_found": 120, "scenes_too_cloudy": 22, "scenes_checked": 64, "scenes_with_clear_pixels": 51, "scenes_used": 6, "acquisition_days_loaded": 6, "acquisition_days_with_data": 6, "first_date_with_data": "2025-02-17", "last_date_with_data": "2025-12-28"}, "vector_stats": {"feature_count": 3, "total_area_m2": 80000.0, "by_severity": {"1": 1, "2": 1, "3": 1}, "min_zone_pixels": 4, "min_zone_area_m2": 400.0, "skipped_small": 131, "skipped_small_area_m2": 26200.0, "skipped_invalid": 0, "max_features_reached": false}, "severity_areas": {"algorithm": "lisa", "pixels": 797400, "pixel_area_m2": 100.0, "area_m2": 79740000.0, "pixels_by_severity": {"0": 796338, "1": 230, "2": 180, "3": 652}, "vegetated_pixels": 561062}}'),
  ('00000000-0000-4000-a000-000000000004', 'completed', 'ndvi', 'element84-s2',
   ST_MakeEnvelope(11.00, 44.00, 11.10, 44.10, 4326),
   '2026-01-06 07:30:00+00', '2026-01-13 07:30:00+00',
   '{"geocoded_name": "Testville North (synthetic)", "resolution_m": 10, "apply_cloud_mask": true}',
   100,
   'Completed with partial coverage: cloud-free data covers only 41.5% of the requested area (1 acquisition day, 2026-01-09). Reason: 58.5% of the area was masked out (cloud, shadow, snow) on every day loaded (of the 9 scenes in the window, 6 were at least 95% cloudy and 2 of the other 3 had cloud-free pixels over the area). A longer time window may cover more of it.',
   '2026-01-13 07:30:00+00', '2026-01-13 07:30:45+00',
   '{"coverage": {"coverage_pct": 41.5, "low_coverage": true, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 58.5, "uncovered_no_data_pct": 0.0, "area_px": 890448, "covered_px": 369536, "scenes_found": 9, "scenes_too_cloudy": 6, "scenes_checked": 3, "scenes_with_clear_pixels": 2, "scenes_used": 2, "acquisition_days_loaded": 1, "acquisition_days_with_data": 1, "first_date_with_data": "2026-01-09", "last_date_with_data": "2026-01-09"}, "value_distribution": {"index": "ndvi", "pixels": 369536, "pixel_area_m2": 100.0, "area_m2": 36953600.0, "bin_width": 0.01, "min": 0.083, "max": 0.717, "mean": 0.4085}}'),
  ('00000000-0000-4000-a000-000000000005', 'failed', 'ndwi', 'element84-s2',
   ST_MakeEnvelope(9.50, 46.00, 9.60, 46.10, 4326),
   '2025-12-15 12:00:00+00', '2026-01-14 12:00:00+00',
   '{"geocoded_name": "Nowhere Lake (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   30, 'No satellite data found. Try a wider date range or different area.',
   '2026-01-14 12:00:00+00', '2026-01-14 12:00:30+00', NULL),
  ('00000000-0000-4000-a000-000000000006', 'pending', 'flood', 'element84-s2',
   ST_MakeEnvelope(12.05, 45.05, 12.15, 45.15, 4326),
   '2025-12-16 16:00:00+00', '2026-01-15 16:00:00+00',
   '{"geocoded_name": "Sample Plain (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   0, NULL, '2026-01-15 16:00:00+00', NULL, NULL),
  ('00000000-0000-4000-a000-000000000007', 'completed', 'ndvi', 'element84-s2',
   ST_MakeEnvelope(12.00, 45.00, 12.10, 45.10, 4326),
   '2025-06-01 00:00:00+00', '2025-06-30 23:59:59+00',
   '{"geocoded_name": "Sample Plain (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 99.4% of the requested area (3 acquisition days, 2025-06-04 to 2025-06-24).',
   '2026-01-11 10:00:00+00', '2026-01-11 10:01:10+00',
   '{"coverage": {"coverage_pct": 99.4, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 0.6, "uncovered_no_data_pct": 0.0, "area_px": 218871, "covered_px": 217558, "scenes_found": 12, "scenes_too_cloudy": 2, "scenes_checked": 10, "scenes_with_clear_pixels": 9, "scenes_used": 3, "acquisition_days_loaded": 3, "acquisition_days_with_data": 3, "first_date_with_data": "2025-06-04", "last_date_with_data": "2025-06-24"}, "value_distribution": {"index": "ndvi", "pixels": 217558, "pixel_area_m2": 400.0, "area_m2": 87023200.0, "bin_width": 0.01, "min": 0.123, "max": 0.837, "mean": 0.6107}, "summary": {"band_1": {"min": -0.214, "max": 0.893, "mean": 0.5874}}}'),
  ('00000000-0000-4000-a000-000000000008', 'completed', 'ndvi', 'element84-s2',
   ST_MakeEnvelope(12.00, 45.00, 12.10, 45.10, 4326),
   '2025-12-01 00:00:00+00', '2025-12-31 23:59:59+00',
   '{"geocoded_name": "Sample Plain (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 96.3% of the requested area (4 acquisition days, 2025-12-03 to 2025-12-28).',
   '2026-01-11 11:00:00+00', '2026-01-11 11:01:25+00',
   '{"coverage": {"coverage_pct": 96.3, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 3.7, "uncovered_no_data_pct": 0.0, "area_px": 218871, "covered_px": 210773, "scenes_found": 12, "scenes_too_cloudy": 5, "scenes_checked": 7, "scenes_with_clear_pixels": 5, "scenes_used": 4, "acquisition_days_loaded": 4, "acquisition_days_with_data": 4, "first_date_with_data": "2025-12-03", "last_date_with_data": "2025-12-28"}, "value_distribution": {"index": "ndvi", "pixels": 210773, "pixel_area_m2": 400.0, "area_m2": 84309200.0, "bin_width": 0.01, "min": 0.053, "max": 0.727, "mean": 0.4119}}'),
  ('00000000-0000-4000-a000-000000000009', 'completed', 'ndwi', 'element84-s2',
   ST_MakeEnvelope(9.50, 46.00, 9.60, 46.10, 4326),
   '2025-11-15 13:00:00+00', '2026-01-14 13:00:00+00',
   '{"geocoded_name": "Nowhere Lake (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 97.1% of the requested area (4 acquisition days, 2025-11-21 to 2026-01-10).',
   '2026-01-14 13:00:00+00', '2026-01-14 13:01:05+00',
   '{"coverage": {"coverage_pct": 97.1, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 2.9, "uncovered_no_data_pct": 0.0, "area_px": 215062, "covered_px": 208826, "scenes_found": 24, "scenes_too_cloudy": 9, "scenes_checked": 15, "scenes_with_clear_pixels": 11, "scenes_used": 4, "acquisition_days_loaded": 4, "acquisition_days_with_data": 4, "first_date_with_data": "2025-11-21", "last_date_with_data": "2026-01-10"}, "value_distribution": {"index": "ndwi", "pixels": 208826, "pixel_area_m2": 400.0, "area_m2": 83530400.0, "bin_width": 0.01, "min": -0.617, "max": 0.377, "mean": -0.1952}}'),
  ('00000000-0000-4000-a000-000000000010', 'completed', 'mndwi', 'element84-s2',
   ST_MakeEnvelope(9.50, 46.00, 9.60, 46.10, 4326),
   '2025-11-15 13:00:00+00', '2026-01-14 13:00:00+00',
   '{"geocoded_name": "Nowhere Lake (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 88.2% of the requested area (3 acquisition days, 2025-11-21 to 2026-01-10).',
   '2026-01-14 13:30:00+00', '2026-01-14 13:31:00+00',
   '{"coverage": {"coverage_pct": 88.2, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 11.8, "uncovered_no_data_pct": 0.0, "area_px": 215062, "covered_px": 189685, "scenes_found": 24, "scenes_too_cloudy": 9, "scenes_checked": 15, "scenes_with_clear_pixels": 11, "scenes_used": 3, "acquisition_days_loaded": 3, "acquisition_days_with_data": 3, "first_date_with_data": "2025-11-21", "last_date_with_data": "2026-01-10"}, "value_distribution": {"index": "mndwi", "pixels": 189685, "pixel_area_m2": 400.0, "area_m2": 75874000.0, "bin_width": 0.01, "min": -0.547, "max": 0.617, "mean": -0.1098}}'),
  ('00000000-0000-4000-a000-000000000011', 'completed', 'ndwi', 'element84-s2',
   ST_MakeEnvelope(13.50, 45.60, 13.60, 45.70, 4326),
   '2025-11-13 15:00:00+00', '2026-01-12 15:00:00+00',
   '{"geocoded_name": "Old Harbour (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100,
   'Completed: cloud-free data covers 93.4% of the requested area (5 acquisition days, 2025-11-14 to 2026-01-08).',
   '2026-01-12 15:00:00+00', '2026-01-12 15:01:20+00',
   '{"coverage": {"coverage_pct": 93.4, "low_coverage": false, "low_coverage_below_pct": 80.0, "uncovered_masked_pct": 4.1, "uncovered_no_data_pct": 2.5, "area_px": 216593, "covered_px": 202298, "scenes_found": 26, "scenes_too_cloudy": 8, "scenes_checked": 18, "scenes_with_clear_pixels": 14, "scenes_used": 5, "acquisition_days_loaded": 5, "acquisition_days_with_data": 5, "first_date_with_data": "2025-11-14", "last_date_with_data": "2026-01-08"}, "summary": {"band_1": {"min": -0.782, "max": 0.641, "mean": -0.2391}}}');

INSERT INTO alert_features (id, job_id, algorithm, severity, quadrant, geometry, centroid,
                            area_m2, perimeter_m, created_at)
SELECT v.id, v.job_id::uuid, v.algorithm, v.severity, v.quadrant,
       ST_MakeEnvelope(v.x, v.y, v.x + 0.002, v.y + 0.002, 4326),
       ST_Centroid(ST_MakeEnvelope(v.x, v.y, v.x + 0.002, v.y + 0.002, 4326)),
       v.area_m2, v.perimeter_m, v.created_at::timestamptz
FROM (VALUES
  ( 1, '00000000-0000-4000-a000-000000000001', 'flood', 3, NULL::smallint, 11.010, 44.010,  50000.0,  900.0, '2026-01-10 09:01:30+00'),
  ( 2, '00000000-0000-4000-a000-000000000001', 'flood', 3, NULL,          11.020, 44.020,  30000.0,  700.0, '2026-01-10 09:01:30+00'),
  ( 3, '00000000-0000-4000-a000-000000000001', 'flood', 2, NULL,          11.030, 44.030,  40000.0,  800.0, '2026-01-10 09:01:30+00'),
  ( 4, '00000000-0000-4000-a000-000000000001', 'flood', 1, NULL,          11.040, 44.040,  10000.0,  400.0, '2026-01-10 09:01:30+00'),
  ( 5, '00000000-0000-4000-a000-000000000002', 'flood', 3, NULL,          11.050, 44.050, 120000.0, 1400.0, '2026-01-12 10:01:50+00'),
  ( 6, '00000000-0000-4000-a000-000000000002', 'flood', 2, NULL,          11.060, 44.060,  20000.0,  560.0, '2026-01-12 10:01:50+00'),
  ( 7, '00000000-0000-4000-a000-000000000002', 'flood', 2, NULL,          11.070, 44.070,  25000.0,  620.0, '2026-01-12 10:01:50+00'),
  ( 8, '00000000-0000-4000-a000-000000000003', 'lisa',  3, 3,             12.010, 45.010,  60000.0, 1000.0, '2026-01-11 08:03:10+00'),
  ( 9, '00000000-0000-4000-a000-000000000003', 'lisa',  2, 4,             12.020, 45.020,  15000.0,  500.0, '2026-01-11 08:03:10+00'),
  (10, '00000000-0000-4000-a000-000000000003', 'lisa',  1, 3,             12.030, 45.030,   5000.0,  300.0, '2026-01-11 08:03:10+00')
) AS v(id, job_id, algorithm, severity, quadrant, x, y, area_m2, perimeter_m, created_at);

SELECT setval(pg_get_serial_sequence('alert_features', 'id'), 10);

-- Value intervals of the continuous-index analyses 004 and 007-010 (011 has none:
-- run before distributions were recorded). The pixel counts add up to each job's
-- result.coverage.covered_px; area_m2 = pixel_count * resolution_m ^ 2.
INSERT INTO index_value_bins (job_id, bin_index, value_from, value_to, pixel_count, area_m2)
VALUES
  ('00000000-0000-4000-a000-000000000004', 108,  0.08,  0.09,  36953,  3695300.0),
  ('00000000-0000-4000-a000-000000000004', 122,  0.22,  0.23,  92384,  9238400.0),
  ('00000000-0000-4000-a000-000000000004', 141,  0.41,  0.42, 110860, 11086000.0),
  ('00000000-0000-4000-a000-000000000004', 158,  0.58,  0.59,  73907,  7390700.0),
  ('00000000-0000-4000-a000-000000000004', 166,  0.66,  0.67,  36953,  3695300.0),
  ('00000000-0000-4000-a000-000000000004', 171,  0.71,  0.72,  18479,  1847900.0),
  ('00000000-0000-4000-a000-000000000007', 112,  0.12,  0.13,   8702,  3480800.0),
  ('00000000-0000-4000-a000-000000000007', 127,  0.27,  0.28,  17404,  6961600.0),
  ('00000000-0000-4000-a000-000000000007', 138,  0.38,  0.39,  21755,  8702000.0),
  ('00000000-0000-4000-a000-000000000007', 149,  0.49,  0.50,  26106, 10442400.0),
  ('00000000-0000-4000-a000-000000000007', 150,  0.50,  0.51,  13053,  5221200.0),
  ('00000000-0000-4000-a000-000000000007', 164,  0.64,  0.65,  10877,  4350800.0),
  ('00000000-0000-4000-a000-000000000007', 165,  0.65,  0.66,  21755,  8702000.0),
  ('00000000-0000-4000-a000-000000000007', 174,  0.74,  0.75,  43511, 17404400.0),
  ('00000000-0000-4000-a000-000000000007', 183,  0.83,  0.84,  54395, 21758000.0),
  ('00000000-0000-4000-a000-000000000008', 105,  0.05,  0.06,  12646,  5058400.0),
  ('00000000-0000-4000-a000-000000000008', 118,  0.18,  0.19,  29508, 11803200.0),
  ('00000000-0000-4000-a000-000000000008', 131,  0.31,  0.32,  52693, 21077200.0),
  ('00000000-0000-4000-a000-000000000008', 144,  0.44,  0.45,  46370, 18548000.0),
  ('00000000-0000-4000-a000-000000000008', 150,  0.50,  0.51,  10538,  4215200.0),
  ('00000000-0000-4000-a000-000000000008', 157,  0.57,  0.58,  21077,  8430800.0),
  ('00000000-0000-4000-a000-000000000008', 166,  0.66,  0.67,  25292, 10116800.0),
  ('00000000-0000-4000-a000-000000000008', 172,  0.72,  0.73,  12649,  5059600.0),
  ('00000000-0000-4000-a000-000000000009',  38, -0.62, -0.61,  31323, 12529200.0),
  ('00000000-0000-4000-a000-000000000009',  55, -0.45, -0.44,  52206, 20882400.0),
  ('00000000-0000-4000-a000-000000000009',  69, -0.31, -0.30,  41765, 16706000.0),
  ('00000000-0000-4000-a000-000000000009',  88, -0.12, -0.11,  16706,  6682400.0),
  ('00000000-0000-4000-a000-000000000009',  99, -0.01,  0.00,   4176,  1670400.0),
  ('00000000-0000-4000-a000-000000000009', 100,  0.00,  0.01,   6264,  2505600.0),
  ('00000000-0000-4000-a000-000000000009', 118,  0.18,  0.19,  25059, 10023600.0),
  ('00000000-0000-4000-a000-000000000009', 137,  0.37,  0.38,  31327, 12530800.0),
  ('00000000-0000-4000-a000-000000000010',  45, -0.55, -0.54,  37937, 15174800.0),
  ('00000000-0000-4000-a000-000000000010',  62, -0.38, -0.37,  47421, 18968400.0),
  ('00000000-0000-4000-a000-000000000010',  80, -0.20, -0.19,  28452, 11380800.0),
  ('00000000-0000-4000-a000-000000000010',  95, -0.05, -0.04,  11381,  4552400.0),
  ('00000000-0000-4000-a000-000000000010', 104,  0.04,  0.05,   7587,  3034800.0),
  ('00000000-0000-4000-a000-000000000010', 122,  0.22,  0.23,  18968,  7587200.0),
  ('00000000-0000-4000-a000-000000000010', 145,  0.45,  0.46,  26555, 10622000.0),
  ('00000000-0000-4000-a000-000000000010', 161,  0.61,  0.62,  11384,  4553600.0);

-- Class areas of the flood and lisa analyses 002 and 003 (001 has none: run before
-- they were recorded). Every class has a row, empty ones included; the pixel counts
-- add up to each job's result.coverage.covered_px; area_m2 = pixel_count * 100.
INSERT INTO severity_class_areas (job_id, algorithm, severity, quadrant, pixel_count, area_m2)
VALUES
  ('00000000-0000-4000-a000-000000000002', 'flood', 0, 0, 886318, 88631800.0),
  ('00000000-0000-4000-a000-000000000002', 'flood', 1, 0,     52,     5200.0),
  ('00000000-0000-4000-a000-000000000002', 'flood', 2, 0,    486,    48600.0),
  ('00000000-0000-4000-a000-000000000002', 'flood', 3, 0,   1264,   126400.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  0, 0, 236338, 23633800.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  0, 1, 250000, 25000000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  0, 2, 230000, 23000000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  0, 3,  45000,  4500000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  0, 4,  35000,  3500000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  1, 3,    170,    17000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  1, 4,     60,     6000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  2, 3,      0,        0.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  2, 4,    180,    18000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  3, 3,    640,    64000.0),
  ('00000000-0000-4000-a000-000000000003', 'lisa',  3, 4,     12,     1200.0);
