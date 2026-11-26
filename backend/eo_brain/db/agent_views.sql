-- Curated, read-only view layer for the data agent.
-- Run AFTER schema.sql and schema_vectors.sql. Idempotent.
--
-- The data agent never sees the application tables. It sees these views,
-- with model-friendly names, derived units and labels. The COMMENT ON text
-- below is the schema documentation: eo_brain/agent/sql_exec.py reads it back
-- from the catalog and puts it in the data agent's prompt, so there is one
-- source of truth for both the database and the model.
--
-- Geometry columns are exposed only so spatial predicates can use them; the
-- executor never returns geometry values to the model.
--
-- Closed-set columns (a fixed list of possible values) list them in their
-- comment as "Values: <list>": text values in single quotes, exactly as the
-- view returns them (case-sensitive, for = and IN), integer codes bare. Any
-- explanation comes after the list and contains no quotes. The comments use
-- dollar quoting ($$...$$) so the single quotes need no escaping.
-- tests/test_db_view_values.py checks every list against what the view can
-- produce: the view's own CASE mappings, the job_status enum, and the Python
-- constants the writers use (algorithm registry, Severity, MoranQuadrant,
-- DISCRETE_OUTPUT_ALGORITHMS).

CREATE SCHEMA IF NOT EXISTS agent;
COMMENT ON SCHEMA agent IS 'Read-only views over analysis jobs and their alert polygons, for the data agent.';

-- ---------------------------------------------------------------- analyses

CREATE OR REPLACE VIEW agent.analyses AS
SELECT
    j.id                                        AS analysis_id,
    j.algorithm                                 AS algorithm,
    CASE j.algorithm
        WHEN 'ndvi'  THEN 'vegetation index (NDVI)'
        WHEN 'ndwi'  THEN 'water index (NDWI)'
        WHEN 'mndwi' THEN 'modified water index (MNDWI)'
        WHEN 'flood' THEN 'flood extent'
        WHEN 'lisa'  THEN 'vegetation anomalies (LISA)'
        ELSE j.algorithm
    END                                         AS algorithm_label,
    j.status::text                              AS status,
    j.params ->> 'geocoded_name'                AS location_name,
    CASE WHEN (j.params ->> 'resolution_m') ~ '^[0-9]+$'
         THEN (j.params ->> 'resolution_m')::integer END AS resolution_m,
    j.start_date                                AS window_start,
    j.end_date                                  AS window_end,
    (j.end_date::date - j.start_date::date)     AS window_days,
    j.created_at                                AS created_at,
    j.completed_at                              AS completed_at,
    EXTRACT(EPOCH FROM (j.completed_at - j.created_at))::double precision AS processing_seconds,
    ST_Area(j.bbox::geography) / 1e4            AS footprint_area_ha,
    ST_Area(j.bbox::geography) / 1e6            AS footprint_area_km2,
    ST_XMin(j.bbox)                             AS west,
    ST_YMin(j.bbox)                             AS south,
    ST_XMax(j.bbox)                             AS east,
    ST_YMax(j.bbox)                             AS north,
    j.message                                   AS status_message,
    j.bbox                                      AS footprint
FROM public.jobs AS j;

COMMENT ON VIEW agent.analyses IS
    'One row per satellite analysis job (requested area, algorithm, time window, status).';
COMMENT ON COLUMN agent.analyses.analysis_id IS 'Unique id of the analysis (uuid).';
COMMENT ON COLUMN agent.analyses.algorithm IS
    $$Algorithm code. Values: 'ndvi', 'ndwi', 'mndwi', 'flood', 'lisa'. Filter on this column, not on algorithm_label.$$;
COMMENT ON COLUMN agent.analyses.algorithm_label IS
    $$Display name of the algorithm. Values: 'vegetation index (NDVI)', 'water index (NDWI)', 'modified water index (MNDWI)', 'flood extent', 'vegetation anomalies (LISA)' (for ndvi, ndwi, mndwi, flood, lisa in that order). For display only.$$;
COMMENT ON COLUMN agent.analyses.status IS
    $$Processing state. Values: 'pending', 'processing', 'completed', 'failed'.$$;
COMMENT ON COLUMN agent.analyses.location_name IS 'Place name the request was geocoded to; may be NULL. Match with ILIKE.';
COMMENT ON COLUMN agent.analyses.resolution_m IS 'Processing resolution in metres.';
COMMENT ON COLUMN agent.analyses.window_start IS 'Start of the satellite imagery time window.';
COMMENT ON COLUMN agent.analyses.window_end IS 'End of the satellite imagery time window.';
COMMENT ON COLUMN agent.analyses.window_days IS 'Length of the time window in days.';
COMMENT ON COLUMN agent.analyses.created_at IS 'When the analysis was requested. "Latest analysis" = highest created_at.';
COMMENT ON COLUMN agent.analyses.completed_at IS 'When processing finished (completed or failed); NULL while running.';
COMMENT ON COLUMN agent.analyses.processing_seconds IS 'Seconds from request to completion; NULL while running.';
COMMENT ON COLUMN agent.analyses.footprint_area_ha IS 'Area of the requested bounding box in hectares (not the alert area).';
COMMENT ON COLUMN agent.analyses.footprint_area_km2 IS 'Area of the requested bounding box in square kilometres.';
COMMENT ON COLUMN agent.analyses.west IS 'Bounding box west longitude (WGS84 degrees).';
COMMENT ON COLUMN agent.analyses.south IS 'Bounding box south latitude (WGS84 degrees).';
COMMENT ON COLUMN agent.analyses.east IS 'Bounding box east longitude (WGS84 degrees).';
COMMENT ON COLUMN agent.analyses.north IS 'Bounding box north latitude (WGS84 degrees).';
COMMENT ON COLUMN agent.analyses.status_message IS 'Last progress or error message.';
COMMENT ON COLUMN agent.analyses.footprint IS
    'Bounding box polygon, EPSG:4326. Use only inside spatial functions (ST_Intersects, ST_Area(footprint::geography)); never returned.';

-- ---------------------------------------------------------------- alert areas

CREATE OR REPLACE VIEW agent.alert_areas AS
SELECT
    f.id                                        AS feature_id,
    f.job_id                                    AS analysis_id,
    f.algorithm                                 AS algorithm,
    f.severity                                  AS severity,
    CASE f.severity WHEN 1 THEN 'watch' WHEN 2 THEN 'suspect' WHEN 3 THEN 'alert' END AS severity_label,
    f.quadrant                                  AS quadrant,
    CASE f.quadrant WHEN 1 THEN 'HH' WHEN 2 THEN 'LL' WHEN 3 THEN 'HL' WHEN 4 THEN 'LH' END AS quadrant_label,
    f.area_m2                                   AS area_m2,
    f.area_m2 / 1e4                             AS area_ha,
    f.area_m2 / 1e6                             AS area_km2,
    f.perimeter_m                               AS perimeter_m,
    ST_X(f.centroid)                            AS centroid_lon,
    ST_Y(f.centroid)                            AS centroid_lat,
    f.created_at                                AS detected_at,
    j.params ->> 'geocoded_name'                AS location_name,
    j.created_at                                AS analysis_created_at,
    f.geometry                                  AS geometry
FROM public.alert_features AS f
JOIN public.jobs AS j ON j.id = f.job_id;

COMMENT ON VIEW agent.alert_areas IS
    'One row per alert polygon produced by a flood or lisa analysis (continuous indices like ndvi produce none).';
COMMENT ON COLUMN agent.alert_areas.feature_id IS 'Unique id of the polygon.';
COMMENT ON COLUMN agent.alert_areas.analysis_id IS 'Analysis that produced the polygon (agent.analyses.analysis_id).';
COMMENT ON COLUMN agent.alert_areas.algorithm IS
    $$Algorithm that produced the polygon. Values: 'flood', 'lisa'.$$;
COMMENT ON COLUMN agent.alert_areas.severity IS
    $$Severity code. Values: 1, 2, 3 (1 = watch, 2 = suspect, 3 = alert; higher is more severe).$$;
COMMENT ON COLUMN agent.alert_areas.severity_label IS
    $$Severity name. Values: 'watch', 'suspect', 'alert' (severity 1, 2, 3).$$;
COMMENT ON COLUMN agent.alert_areas.quadrant IS
    $$LISA Moran quadrant code. Values: 1, 2, 3, 4 (1 = HH, 2 = LL, 3 = HL, 4 = LH, see quadrant_label). NULL for flood polygons and whenever no quadrant was recorded.$$;
COMMENT ON COLUMN agent.alert_areas.quadrant_label IS
    $$LISA Moran quadrant as a two-letter code. Values: 'HH', 'LL', 'HL', 'LH'. Meaning: HH = high cluster (high value among high neighbours), LL = low cluster, HL = high outlier (high value among low neighbours), LH = low outlier. NULL for flood polygons and whenever no quadrant was recorded.$$;
COMMENT ON COLUMN agent.alert_areas.area_m2 IS 'Polygon area in square metres (computed in UTM).';
COMMENT ON COLUMN agent.alert_areas.area_ha IS 'Polygon area in hectares.';
COMMENT ON COLUMN agent.alert_areas.area_km2 IS 'Polygon area in square kilometres.';
COMMENT ON COLUMN agent.alert_areas.perimeter_m IS 'Polygon perimeter in metres.';
COMMENT ON COLUMN agent.alert_areas.centroid_lon IS 'Centroid longitude (WGS84 degrees).';
COMMENT ON COLUMN agent.alert_areas.centroid_lat IS 'Centroid latitude (WGS84 degrees).';
COMMENT ON COLUMN agent.alert_areas.detected_at IS 'When the polygon was stored.';
COMMENT ON COLUMN agent.alert_areas.location_name IS 'Place name of the parent analysis.';
COMMENT ON COLUMN agent.alert_areas.analysis_created_at IS 'created_at of the parent analysis.';
COMMENT ON COLUMN agent.alert_areas.geometry IS
    'Polygon, EPSG:4326. Use only inside spatial functions; never returned.';

-- ---------------------------------------------------------------- per-severity summary

CREATE OR REPLACE VIEW agent.analysis_severity_summary AS
SELECT
    f.job_id                                    AS analysis_id,
    j.algorithm                                 AS algorithm,
    j.params ->> 'geocoded_name'                AS location_name,
    j.created_at                                AS analysis_created_at,
    f.severity                                  AS severity,
    CASE f.severity WHEN 1 THEN 'watch' WHEN 2 THEN 'suspect' WHEN 3 THEN 'alert' END AS severity_label,
    count(*)                                    AS feature_count,
    sum(f.area_m2)                              AS total_area_m2,
    sum(f.area_m2) / 1e4                        AS total_area_ha,
    sum(f.area_m2) / 1e6                        AS total_area_km2,
    avg(f.area_m2)                              AS avg_area_m2
FROM public.alert_features AS f
JOIN public.jobs AS j ON j.id = f.job_id
GROUP BY f.job_id, j.algorithm, j.params ->> 'geocoded_name', j.created_at, f.severity;

COMMENT ON VIEW agent.analysis_severity_summary IS
    'Alert polygons aggregated per analysis and severity. Severities with no polygons have no row (treat as 0).';
COMMENT ON COLUMN agent.analysis_severity_summary.analysis_id IS 'Analysis id (agent.analyses.analysis_id).';
COMMENT ON COLUMN agent.analysis_severity_summary.algorithm IS
    $$Algorithm of the analysis. Values: 'flood', 'lisa'.$$;
COMMENT ON COLUMN agent.analysis_severity_summary.location_name IS 'Place name of the analysis.';
COMMENT ON COLUMN agent.analysis_severity_summary.analysis_created_at IS 'created_at of the analysis.';
COMMENT ON COLUMN agent.analysis_severity_summary.severity IS
    $$Severity code. Values: 1, 2, 3 (1 = watch, 2 = suspect, 3 = alert; higher is more severe).$$;
COMMENT ON COLUMN agent.analysis_severity_summary.severity_label IS
    $$Severity name. Values: 'watch', 'suspect', 'alert' (severity 1, 2, 3).$$;
COMMENT ON COLUMN agent.analysis_severity_summary.feature_count IS 'Number of polygons.';
COMMENT ON COLUMN agent.analysis_severity_summary.total_area_m2 IS 'Total polygon area in square metres.';
COMMENT ON COLUMN agent.analysis_severity_summary.total_area_ha IS 'Total polygon area in hectares.';
COMMENT ON COLUMN agent.analysis_severity_summary.total_area_km2 IS 'Total polygon area in square kilometres.';
COMMENT ON COLUMN agent.analysis_severity_summary.avg_area_m2 IS 'Mean polygon area in square metres.';
