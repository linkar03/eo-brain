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
-- OUTLIER_QUADRANTS, DISCRETE_OUTPUT_ALGORITHMS, CONTINUOUS_OUTPUT_ALGORITHMS).

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
    j.bbox                                      AS footprint,
    -- Coverage (api.main writes result.coverage, core.coverage computes it).
    -- Appended after footprint: CREATE OR REPLACE VIEW can only add columns at the end.
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'coverage_pct') = 'number'
         THEN (j.result -> 'coverage' ->> 'coverage_pct')::double precision END AS coverage_pct,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'low_coverage') = 'boolean'
         THEN (j.result -> 'coverage' ->> 'low_coverage')::boolean END AS low_coverage,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'uncovered_masked_pct') = 'number'
         THEN (j.result -> 'coverage' ->> 'uncovered_masked_pct')::double precision END AS uncovered_masked_pct,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'uncovered_no_data_pct') = 'number'
         THEN (j.result -> 'coverage' ->> 'uncovered_no_data_pct')::double precision END AS uncovered_no_data_pct,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'scenes_found') = 'number'
         THEN (j.result -> 'coverage' ->> 'scenes_found')::integer END AS scenes_found,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'scenes_too_cloudy') = 'number'
         THEN (j.result -> 'coverage' ->> 'scenes_too_cloudy')::integer END AS scenes_too_cloudy,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'scenes_checked') = 'number'
         THEN (j.result -> 'coverage' ->> 'scenes_checked')::integer END AS scenes_checked,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'scenes_with_clear_pixels') = 'number'
         THEN (j.result -> 'coverage' ->> 'scenes_with_clear_pixels')::integer END AS scenes_with_clear_pixels,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'scenes_used') = 'number'
         THEN (j.result -> 'coverage' ->> 'scenes_used')::integer END AS scenes_used,
    CASE WHEN jsonb_typeof(j.result -> 'coverage' -> 'acquisition_days_with_data') = 'number'
         THEN (j.result -> 'coverage' ->> 'acquisition_days_with_data')::integer END AS acquisition_days,
    CASE WHEN (j.result -> 'coverage' ->> 'first_date_with_data') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
         THEN (j.result -> 'coverage' ->> 'first_date_with_data')::date END AS first_acquisition,
    CASE WHEN (j.result -> 'coverage' ->> 'last_date_with_data') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
         THEN (j.result -> 'coverage' ->> 'last_date_with_data')::date END AS last_acquisition,
    -- Index statistics of continuous results over the covered pixels (api.main
    -- writes result.value_distribution, core.distribution computes it; not
    -- result.summary, which covers the whole result grid, a margin outside the
    -- requested box included).
    CASE WHEN jsonb_typeof(j.result -> 'value_distribution' -> 'mean') = 'number'
         THEN (j.result -> 'value_distribution' ->> 'mean')::double precision END AS value_mean,
    CASE WHEN jsonb_typeof(j.result -> 'value_distribution' -> 'min') = 'number'
         THEN (j.result -> 'value_distribution' ->> 'min')::double precision END AS value_min,
    CASE WHEN jsonb_typeof(j.result -> 'value_distribution' -> 'max') = 'number'
         THEN (j.result -> 'value_distribution' ->> 'max')::double precision END AS value_max,
    -- Mapped zones of flood and lisa results (api.main writes result.vector_stats,
    -- algorithms.vectorize computes it). Analyses run before these keys were
    -- written have a vector_stats block without them: NULL here.
    CASE WHEN jsonb_typeof(j.result -> 'vector_stats' -> 'min_zone_pixels') = 'number'
         THEN (j.result -> 'vector_stats' ->> 'min_zone_pixels')::integer END AS min_zone_pixels,
    CASE WHEN jsonb_typeof(j.result -> 'vector_stats' -> 'min_zone_area_m2') = 'number'
         THEN (j.result -> 'vector_stats' ->> 'min_zone_area_m2')::double precision END AS min_zone_area_m2,
    CASE WHEN jsonb_typeof(j.result -> 'vector_stats' -> 'skipped_small') = 'number'
         THEN (j.result -> 'vector_stats' ->> 'skipped_small')::integer END AS small_patches_left_out,
    CASE WHEN jsonb_typeof(j.result -> 'vector_stats' -> 'skipped_small_area_m2') = 'number'
         THEN (j.result -> 'vector_stats' ->> 'skipped_small_area_m2')::double precision / 1e4 END AS small_patches_area_ha,
    CASE WHEN jsonb_typeof(j.result -> 'vector_stats' -> 'max_features_reached') = 'boolean'
         THEN (j.result -> 'vector_stats' ->> 'max_features_reached')::boolean END AS zone_limit_reached
FROM public.jobs AS j;

COMMENT ON VIEW agent.analyses IS
    'One row per satellite analysis job (requested area, algorithm, time window, status; for completed jobs, how much of the area has data, and why the rest has none; for ndvi, ndwi and mndwi, the mean, lowest and highest index value; for flood and lisa, the minimum size of a mapped zone and how many smaller patches were left out).';
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
COMMENT ON COLUMN agent.analyses.coverage_pct IS
    'Percent (0 to 100) of the requested area with a cloud-free result value, rounded down. NULL unless completed (and for analyses run before coverage was recorded). The rest of the area is blank on the map: see uncovered_masked_pct and uncovered_no_data_pct for why.';
COMMENT ON COLUMN agent.analyses.low_coverage IS
    'TRUE when coverage_pct is below 80: the analysis completed, but a fifth or more of the area has no data and its figures describe only the covered part; status_message then states the reason. FALSE otherwise; NULL when coverage_pct is NULL.';
COMMENT ON COLUMN agent.analyses.uncovered_masked_pct IS
    'Percent of the requested area that the loaded scenes saw but the cloud mask removed on every acquisition day (clouds, cloud shadows, snow). Large values mean a cloudy time window; a longer window usually helps.';
COMMENT ON COLUMN agent.analyses.uncovered_no_data_pct IS
    'Percent of the requested area outside the footprint of every loaded scene (no satellite pass imaged it in the scenes used).';
COMMENT ON COLUMN agent.analyses.scenes_found IS 'Satellite scenes in the catalogue for the area and time window, at any cloud cover.';
COMMENT ON COLUMN agent.analyses.scenes_too_cloudy IS 'Scenes found whose whole-tile cloud cover was 95 percent or more; skipped without checking their pixels.';
COMMENT ON COLUMN agent.analyses.scenes_checked IS 'Scenes whose cloud mask was checked over the area before choosing; NULL if the check was not possible.';
COMMENT ON COLUMN agent.analyses.scenes_with_clear_pixels IS 'Scenes, among those checked, with at least one cloud-free pixel over the area.';
COMMENT ON COLUMN agent.analyses.scenes_used IS 'Scenes actually loaded to compute the result (the best combination for coverage, not all scenes found).';
COMMENT ON COLUMN agent.analyses.acquisition_days IS 'Acquisition days that contributed cloud-free pixels to the result.';
COMMENT ON COLUMN agent.analyses.first_acquisition IS 'First acquisition day that contributed cloud-free pixels.';
COMMENT ON COLUMN agent.analyses.last_acquisition IS 'Last acquisition day that contributed cloud-free pixels.';
COMMENT ON COLUMN agent.analyses.value_mean IS
    'Mean index value over the covered pixels of the requested area: the pixels counted by coverage_pct and by agent.index_value_distribution. Completed ndvi, ndwi and mndwi analyses only; NULL for flood and lisa, and for analyses run before it was recorded.';
COMMENT ON COLUMN agent.analyses.value_min IS
    'Lowest index value over the same covered pixels as value_mean; NULL in the same cases.';
COMMENT ON COLUMN agent.analyses.value_max IS
    'Highest index value over the same covered pixels as value_mean; NULL in the same cases.';
COMMENT ON COLUMN agent.analyses.min_zone_pixels IS
    'flood and lisa: the smallest patch mapped as a zone, in pixels of the result grid. A zone is a connected patch of pixels of one class (lisa: one severity and one quadrant) inside the requested area; smaller patches are not mapped (agent.alert_areas has no row for them), but their pixels are counted in agent.severity_areas. NULL for ndvi, ndwi and mndwi, and for flood and lisa analyses run before it was recorded: then the zone size is not available.';
COMMENT ON COLUMN agent.analyses.min_zone_area_m2 IS
    'The same minimum in square metres (min_zone_pixels times the pixel area). Return it with any number of zones: N zones of at least this size. NULL in the same cases as min_zone_pixels.';
COMMENT ON COLUMN agent.analyses.small_patches_left_out IS
    'Number of patches, of any severity, smaller than min_zone_pixels and therefore not mapped as zones. Return it with any number of zones. NULL in the same cases as min_zone_pixels.';
COMMENT ON COLUMN agent.analyses.small_patches_area_ha IS
    'Total area in hectares of those patches. NULL in the same cases as min_zone_pixels.';
COMMENT ON COLUMN agent.analyses.zone_limit_reached IS
    'TRUE when mapping stopped at its limit of 50000 zones: the analysis has more zones than agent.alert_areas lists, and small_patches_left_out counts only the patches met before the stop. NULL in the same cases as min_zone_pixels.';

-- ---------------------------------------------------------------- alert areas

CREATE OR REPLACE VIEW agent.alert_areas AS
SELECT
    f.id                                        AS feature_id,
    f.job_id                                    AS analysis_id,
    f.algorithm                                 AS algorithm,
    f.severity                                  AS severity,
    CASE f.severity WHEN 1 THEN 'watch' WHEN 2 THEN 'suspect' WHEN 3 THEN 'alert' END AS severity_label,
    f.quadrant                                  AS quadrant,
    -- LISA zones are spatial outliers only (algorithms.base.OUTLIER_QUADRANTS).
    CASE f.quadrant WHEN 3 THEN 'HL' WHEN 4 THEN 'LH' END AS quadrant_label,
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
    'One row per mapped zone (alert polygon) of a flood or lisa analysis: a connected patch of pixels of one severity (lisa: and one quadrant) inside the requested area, of at least agent.analyses.min_zone_area_m2. Smaller patches are not mapped: agent.analyses.small_patches_left_out counts them. Use this view to count zones and to locate or measure single zones; for the area or the share of a severity use agent.severity_areas, which counts every pixel, small patches included, so a sum of area_ha here is less than the area of the severity. Zones of analyses run before the minimum zone size was recorded (min_zone_area_m2 NULL) have no quadrant and may reach a little beyond the requested area. Continuous indices like ndvi produce no zones: their values are in agent.index_value_distribution.';
COMMENT ON COLUMN agent.alert_areas.feature_id IS 'Unique id of the zone.';
COMMENT ON COLUMN agent.alert_areas.analysis_id IS 'Analysis that produced the zone (agent.analyses.analysis_id).';
COMMENT ON COLUMN agent.alert_areas.algorithm IS
    $$Algorithm that produced the zone. Values: 'flood', 'lisa'.$$;
COMMENT ON COLUMN agent.alert_areas.severity IS
    $$Severity code. Values: 1, 2, 3 (1 = watch, 2 = suspect, 3 = alert; higher is more severe).$$;
COMMENT ON COLUMN agent.alert_areas.severity_label IS
    $$Severity name. Values: 'watch', 'suspect', 'alert' (severity 1, 2, 3).$$;
COMMENT ON COLUMN agent.alert_areas.quadrant IS
    $$LISA Moran quadrant code of the zone. Values: 3, 4 (3 = HL, 4 = LH, see quadrant_label). Every pixel of a lisa zone has this quadrant. NULL for flood zones, and for lisa zones of analyses run before the quadrant was recorded.$$;
COMMENT ON COLUMN agent.alert_areas.quadrant_label IS
    $$LISA Moran quadrant as a two-letter code. Values: 'HL', 'LH'. Meaning: HL = high outlier (vegetation index higher than the vegetation around it), LH = low outlier (lower than the vegetation around it). LISA marks only outliers, so every lisa zone is one or the other. NULL for flood zones, and for lisa zones of analyses run before the quadrant was recorded.$$;
COMMENT ON COLUMN agent.alert_areas.area_m2 IS 'Zone area in square metres (computed in UTM).';
COMMENT ON COLUMN agent.alert_areas.area_ha IS 'Zone area in hectares.';
COMMENT ON COLUMN agent.alert_areas.area_km2 IS 'Zone area in square kilometres.';
COMMENT ON COLUMN agent.alert_areas.perimeter_m IS 'Zone perimeter in metres.';
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
    'Mapped zones (agent.alert_areas) aggregated per analysis and severity: how many and their total area. A severity with no zone has no row (0 zones). The total leaves out the patches smaller than agent.analyses.min_zone_area_m2, so it is less than the area of the severity: for areas and shares use agent.severity_areas.';
COMMENT ON COLUMN agent.analysis_severity_summary.analysis_id IS 'Analysis id (agent.analyses.analysis_id).';
COMMENT ON COLUMN agent.analysis_severity_summary.algorithm IS
    $$Algorithm of the analysis. Values: 'flood', 'lisa'.$$;
COMMENT ON COLUMN agent.analysis_severity_summary.location_name IS 'Place name of the analysis.';
COMMENT ON COLUMN agent.analysis_severity_summary.analysis_created_at IS 'created_at of the analysis.';
COMMENT ON COLUMN agent.analysis_severity_summary.severity IS
    $$Severity code. Values: 1, 2, 3 (1 = watch, 2 = suspect, 3 = alert; higher is more severe).$$;
COMMENT ON COLUMN agent.analysis_severity_summary.severity_label IS
    $$Severity name. Values: 'watch', 'suspect', 'alert' (severity 1, 2, 3).$$;
COMMENT ON COLUMN agent.analysis_severity_summary.feature_count IS
    'Number of mapped zones (each of at least agent.analyses.min_zone_area_m2).';
COMMENT ON COLUMN agent.analysis_severity_summary.total_area_m2 IS 'Total area of the mapped zones in square metres.';
COMMENT ON COLUMN agent.analysis_severity_summary.total_area_ha IS 'Total area of the mapped zones in hectares.';
COMMENT ON COLUMN agent.analysis_severity_summary.total_area_km2 IS 'Total area of the mapped zones in square kilometres.';
COMMENT ON COLUMN agent.analysis_severity_summary.avg_area_m2 IS 'Mean zone area in square metres.';

-- ---------------------------------------------------------------- index value distribution

-- api.main writes index_value_bins when a continuous-index job completes
-- (core.distribution computes them over the covered pixels). The share is
-- taken over all intervals of the job, before any filter of the caller.
CREATE OR REPLACE VIEW agent.index_value_distribution AS
SELECT
    b.job_id                                    AS analysis_id,
    j.algorithm                                 AS algorithm,
    j.params ->> 'geocoded_name'                AS location_name,
    j.created_at                                AS analysis_created_at,
    b.value_from                                AS value_from,
    b.value_to                                  AS value_to,
    b.area_m2 / 1e4                             AS area_ha,
    b.area_m2 / 1e6                             AS area_km2,
    (100.0 * b.pixel_count / sum(b.pixel_count) OVER (PARTITION BY b.job_id))::double precision
                                                AS pct_of_covered_area
FROM public.index_value_bins AS b
JOIN public.jobs AS j ON j.id = b.job_id;

COMMENT ON VIEW agent.index_value_distribution IS
    $$How the index values of completed ndvi, ndwi and mndwi analyses are spread over their covered area: one row per value interval 0.01 wide (value_from <= value < value_to) that holds at least one pixel; an interval with no pixel has no row. Area with an index value of at least t: SELECT t AS threshold, SUM(area_ha) FROM agent.index_value_distribution WHERE analysis_id = ... AND value_from >= t. Below t: value_to <= t. From a to b: value_from >= a AND value_to <= b. Share of the covered area: SUM(pct_of_covered_area) with the same filter. Thresholds resolve to 0.01 (round t to two decimals) and the lower edge is inclusive, so above t and at least t are the same filter. When the user names no threshold, use the app defaults: ndvi 0.65 for dense green vegetation (a choice of this app, not a standard); ndwi 0 and mndwi 0 for open water (the usual convention, water above 0; the interval from 0 also holds values of exactly 0). Thresholds are conventions, not measurements: always return the threshold used as a column. Areas describe only the covered part of the requested area, never the rest: when agent.analyses.low_coverage is TRUE, return its coverage_pct with the area. An analysis with no row here has no recorded distribution: analyses run before distributions were recorded, and every flood and lisa analysis (their areas are in agent.severity_areas). With the filter in WHERE, SUM is NULL both then and when no pixel reaches t; to tell the two apart, filter inside the sum: SUM(CASE WHEN value_from >= t THEN area_ha ELSE 0 END) is 0 when no pixel reaches t and NULL only when there is no distribution.$$;
COMMENT ON COLUMN agent.index_value_distribution.analysis_id IS 'Analysis id (agent.analyses.analysis_id).';
COMMENT ON COLUMN agent.index_value_distribution.algorithm IS
    $$Index of the analysis. Values: 'ndvi', 'ndwi', 'mndwi'.$$;
COMMENT ON COLUMN agent.index_value_distribution.location_name IS 'Place name of the analysis.';
COMMENT ON COLUMN agent.index_value_distribution.analysis_created_at IS 'created_at of the analysis.';
COMMENT ON COLUMN agent.index_value_distribution.value_from IS
    'Lower edge of the interval, inclusive: a number with two decimals from -1.00 to 0.99. Filter at least t, and above t, with value_from >= t.';
COMMENT ON COLUMN agent.index_value_distribution.value_to IS
    'Upper edge of the interval, exclusive: value_from + 0.01; the last interval, 0.99 to 1.00, also holds 1. Filter below t with value_to <= t.';
COMMENT ON COLUMN agent.index_value_distribution.area_ha IS
    'Area in hectares of the covered pixels with a value in this interval (pixel count times the pixel area of the result grid). Add it up over the intervals selected.';
COMMENT ON COLUMN agent.index_value_distribution.area_km2 IS 'The same area in square kilometres.';
COMMENT ON COLUMN agent.index_value_distribution.pct_of_covered_area IS
    'Percent (0 to 100) of the covered pixels of the analysis that fall in this interval; over all intervals of an analysis it adds up to 100. A share of the covered area, not of the requested area: times coverage_pct / 100 it becomes a share of the requested area.';

-- ---------------------------------------------------------------- severity areas

-- api.main writes severity_class_areas when a flood or lisa job completes
-- (core.severity_areas counts every class over the covered pixels). LISA pixels
-- of severity 0 are grouped into vegetated (quadrant 1 to 4: they took part in
-- the statistics) and not; the quadrant is shown for the anomaly classes only,
-- the ones a zone can have. The shares are taken over all classes of the job,
-- before any filter of the caller.
CREATE OR REPLACE VIEW agent.severity_areas AS
WITH classes AS (
    SELECT
        c.job_id,
        c.algorithm,
        c.severity,
        CASE WHEN c.algorithm = 'lisa' AND c.severity > 0 THEN c.quadrant END AS quadrant,
        CASE WHEN c.algorithm = 'lisa' THEN c.quadrant <> 0 END AS vegetated,
        sum(c.pixel_count) AS pixel_count,
        sum(c.area_m2) AS area_m2
    FROM public.severity_class_areas AS c
    GROUP BY 1, 2, 3, 4, 5
)
SELECT
    k.job_id                                    AS analysis_id,
    k.algorithm                                 AS algorithm,
    j.params ->> 'geocoded_name'                AS location_name,
    j.created_at                                AS analysis_created_at,
    k.severity                                  AS severity,
    CASE k.severity WHEN 0 THEN 'none' WHEN 1 THEN 'watch' WHEN 2 THEN 'suspect' WHEN 3 THEN 'alert' END AS severity_label,
    k.quadrant                                  AS quadrant,
    CASE k.quadrant WHEN 3 THEN 'HL' WHEN 4 THEN 'LH' END AS quadrant_label,
    k.vegetated                                 AS vegetated,
    k.area_m2 / 1e4                             AS area_ha,
    k.area_m2 / 1e6                             AS area_km2,
    (100.0 * k.pixel_count / nullif(sum(k.pixel_count) OVER (PARTITION BY k.job_id), 0))::double precision
                                                AS pct_of_covered_area,
    CASE WHEN k.vegetated THEN
        (100.0 * k.pixel_count
         / nullif(sum(k.pixel_count) FILTER (WHERE k.vegetated) OVER (PARTITION BY k.job_id), 0))::double precision
    END                                         AS pct_of_vegetated_area
FROM classes AS k
JOIN public.jobs AS j ON j.id = k.job_id;

COMMENT ON VIEW agent.severity_areas IS
    $$Area of every severity class of completed flood and lisa analyses over the covered part of the requested area: every pixel counted once, also the patches too small to be mapped as zones. Use this view for hectares, square kilometres and shares (how much area, what share, how much water); use agent.alert_areas and agent.analysis_severity_summary to count the mapped zones (how many zones, anomalies or areas) and to locate them, never for an area or a share. Rows: flood, one per severity 0 to 3; lisa, one per anomaly class (severity 1 to 3, quadrant HL or LH) and two of severity 0 (vegetated TRUE and FALSE). Every class has its row, with area 0 when it is empty. flood: water seen in the time window, from the highest water index of each pixel over the window (mndwi, or ndwi when the scenes have no SWIR band); severity 1 above 0, 2 above 0.2, 3 above 0.35 (app defaults); water = severity >= 1. It includes permanent water (sea, lakes, rivers): it is water present in the period, not a change from a normal state, so report it as water detected, not as flooded area. lisa: anomalies = severity >= 1, vegetation whose index is a spatial outlier among the vegetation around it. Area of a set of classes: SELECT SUM(area_ha) FROM agent.severity_areas WHERE analysis_id = ... AND severity >= 1. Share of the covered area: SUM(pct_of_covered_area) with the same filter; lisa share of the vegetated area: SUM(pct_of_vegetated_area). Areas describe only the covered part of the requested area, never the rest: when agent.analyses.low_coverage is TRUE, return its coverage_pct with the area. An analysis with no row here has no recorded areas: analyses run before areas were recorded, and every ndvi, ndwi and mndwi analysis. Then SUM is NULL: the area is not available; say so, and do not report 0 or the zone total of agent.analysis_severity_summary instead. A SUM of 0 means the analysis has areas and the classes selected are empty.$$;
COMMENT ON COLUMN agent.severity_areas.analysis_id IS 'Analysis id (agent.analyses.analysis_id).';
COMMENT ON COLUMN agent.severity_areas.algorithm IS
    $$Algorithm of the analysis. Values: 'flood', 'lisa'.$$;
COMMENT ON COLUMN agent.severity_areas.location_name IS 'Place name of the analysis.';
COMMENT ON COLUMN agent.severity_areas.analysis_created_at IS 'created_at of the analysis.';
COMMENT ON COLUMN agent.severity_areas.severity IS
    $$Severity code. Values: 0, 1, 2, 3 (0 = none, 1 = watch, 2 = suspect, 3 = alert; higher is more severe). flood: 0 = no water, 1 to 3 = water; lisa: 0 = no anomaly, 1 to 3 = anomaly.$$;
COMMENT ON COLUMN agent.severity_areas.severity_label IS
    $$Severity name. Values: 'none', 'watch', 'suspect', 'alert' (severity 0, 1, 2, 3).$$;
COMMENT ON COLUMN agent.severity_areas.quadrant IS
    $$LISA Moran quadrant code of an anomaly class. Values: 3, 4 (3 = HL, 4 = LH, see quadrant_label). NULL for severity 0 and for flood.$$;
COMMENT ON COLUMN agent.severity_areas.quadrant_label IS
    $$LISA Moran quadrant as a two-letter code. Values: 'HL', 'LH'. Meaning: HL = high outlier (vegetation index higher than the vegetation around it), LH = low outlier (lower than the vegetation around it). NULL for severity 0 and for flood.$$;
COMMENT ON COLUMN agent.severity_areas.vegetated IS
    'lisa only: TRUE for the pixels LISA classified, vegetated (NDVI of at least 0.3) with vegetated pixels around them; every anomaly row is TRUE. FALSE for the rest of the covered area (bare soil, water, built-up, isolated vegetated pixels). NULL for flood.';
COMMENT ON COLUMN agent.severity_areas.area_ha IS
    'Area in hectares of the covered pixels of this class (pixel count times the pixel area of the result grid); 0 when the class is empty. Add it up over the classes selected.';
COMMENT ON COLUMN agent.severity_areas.area_km2 IS 'The same area in square kilometres.';
COMMENT ON COLUMN agent.severity_areas.pct_of_covered_area IS
    'Percent (0 to 100) of the covered pixels of the analysis in this class; over all rows of an analysis it adds up to 100. A share of the covered area, not of the requested area: times coverage_pct / 100 it becomes a share of the requested area.';
COMMENT ON COLUMN agent.severity_areas.pct_of_vegetated_area IS
    'lisa only: percent (0 to 100) of the vegetated pixels (vegetated TRUE) of the analysis in this class; over the vegetated rows it adds up to 100. NULL for rows with vegetated FALSE, for flood, and when the analysis has no vegetated pixel.';
