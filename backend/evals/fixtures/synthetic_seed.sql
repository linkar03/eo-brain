-- SYNTHETIC TEST FIXTURE - NOT REAL DATA.
-- Invented place names, hand-picked numbers. Used by the database-backed
-- tests and the text-to-SQL eval. Loaded into a throwaway database after
-- schema.sql, schema_vectors.sql, agent_views.sql and agent_role.sql.
--
-- Known answers (area_m2 values are stored as given, not derived from the
-- toy geometries):
--   analysis ...001 flood   completed  Testville North  alert 8 ha (2 polys), suspect 4 ha, watch 1 ha
--   analysis ...002 flood   completed  Testville North  alert 12 ha (1 poly), suspect 4.5 ha (2 polys)
--   analysis ...003 lisa    completed  Sample Plain     alert 6 ha (HL), suspect 1.5 ha (LH), watch 0.5 ha (HL)
--   analysis ...004 ndvi    completed  Testville North  no polygons (continuous index)
--   analysis ...005 ndwi    failed     Nowhere Lake     no polygons
--   analysis ...006 flood   pending    Sample Plain     no polygons
--   footprints: 001 overlaps 002 and 004; 003 overlaps 006; 005 overlaps nothing.

INSERT INTO jobs (id, status, algorithm, provider, bbox, start_date, end_date, params,
                  progress, message, created_at, completed_at)
VALUES
  ('00000000-0000-4000-a000-000000000001', 'completed', 'flood', 'element84-s2',
   ST_MakeEnvelope(11.00, 44.00, 11.10, 44.10, 4326),
   '2025-12-11 09:00:00+00', '2026-01-10 09:00:00+00',
   '{"geocoded_name": "Testville North (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100, 'done', '2026-01-10 09:00:00+00', '2026-01-10 09:01:35+00'),
  ('00000000-0000-4000-a000-000000000002', 'completed', 'flood', 'element84-s2',
   ST_MakeEnvelope(11.02, 44.02, 11.12, 44.12, 4326),
   '2025-10-14 10:00:00+00', '2026-01-12 10:00:00+00',
   '{"geocoded_name": "Testville North (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   100, 'done', '2026-01-12 10:00:00+00', '2026-01-12 10:02:00+00'),
  ('00000000-0000-4000-a000-000000000003', 'completed', 'lisa', 'element84-s2',
   ST_MakeEnvelope(12.00, 45.00, 12.10, 45.10, 4326),
   '2025-01-11 08:00:00+00', '2026-01-11 08:00:00+00',
   '{"geocoded_name": "Sample Plain (synthetic)", "resolution_m": 30, "apply_cloud_mask": true}',
   100, 'done', '2026-01-11 08:00:00+00', '2026-01-11 08:03:20+00'),
  ('00000000-0000-4000-a000-000000000004', 'completed', 'ndvi', 'element84-s2',
   ST_MakeEnvelope(11.00, 44.00, 11.10, 44.10, 4326),
   '2026-01-06 07:30:00+00', '2026-01-13 07:30:00+00',
   '{"geocoded_name": "Testville North (synthetic)", "resolution_m": 10, "apply_cloud_mask": true}',
   100, 'done', '2026-01-13 07:30:00+00', '2026-01-13 07:30:45+00'),
  ('00000000-0000-4000-a000-000000000005', 'failed', 'ndwi', 'element84-s2',
   ST_MakeEnvelope(9.50, 46.00, 9.60, 46.10, 4326),
   '2025-12-15 12:00:00+00', '2026-01-14 12:00:00+00',
   '{"geocoded_name": "Nowhere Lake (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   30, 'No satellite data found. Try a wider date range or different area.',
   '2026-01-14 12:00:00+00', '2026-01-14 12:00:30+00'),
  ('00000000-0000-4000-a000-000000000006', 'pending', 'flood', 'element84-s2',
   ST_MakeEnvelope(12.05, 45.05, 12.15, 45.15, 4326),
   '2025-12-16 16:00:00+00', '2026-01-15 16:00:00+00',
   '{"geocoded_name": "Sample Plain (synthetic)", "resolution_m": 20, "apply_cloud_mask": true}',
   0, NULL, '2026-01-15 16:00:00+00', NULL);

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
