-- EO Brain Vector Features Schema
-- Stores polygonized anomaly/flood regions for MVT serving, and the value
-- distribution of continuous-index results (index_value_bins, at the end)
-- Run AFTER schema.sql (requires jobs table)

CREATE TABLE IF NOT EXISTS alert_features (
    id SERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Classification
    algorithm VARCHAR(20) NOT NULL,      -- 'lisa', 'flood'
    severity SMALLINT NOT NULL,          -- 1=watch, 2=suspect, 3=alert
    quadrant SMALLINT,                   -- LISA only: 3=HL, 4=LH (nullable for flood)
    
    -- Geometry stored in WGS84 for MVT compatibility
    geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    centroid GEOMETRY(POINT, 4326),
    
    -- Pre-calculated metrics (computed in UTM before reprojection)
    area_m2 DOUBLE PRECISION NOT NULL,
    perimeter_m DOUBLE PRECISION,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index for tile queries (critical for MVT performance)
CREATE INDEX IF NOT EXISTS idx_alert_features_geom 
    ON alert_features USING GIST(geometry);

-- Job lookup for cleanup and filtering
CREATE INDEX IF NOT EXISTS idx_alert_features_job 
    ON alert_features(job_id);

-- Severity filtering (common query pattern)
CREATE INDEX IF NOT EXISTS idx_alert_features_severity 
    ON alert_features(job_id, severity);

-- Statistics view for job results
CREATE OR REPLACE VIEW alert_feature_stats AS
SELECT 
    job_id,
    algorithm,
    severity,
    COUNT(*) as feature_count,
    SUM(area_m2) as total_area_m2,
    AVG(area_m2) as avg_area_m2
FROM alert_features
GROUP BY job_id, algorithm, severity;

COMMENT ON TABLE alert_features IS 'Polygonized anomaly regions extracted from raster analysis';
COMMENT ON COLUMN alert_features.quadrant IS 'Moran quadrant for LISA: 3=HL (high outlier), 4=LH (low outlier)';
COMMENT ON COLUMN alert_features.area_m2 IS 'Area in square meters, calculated in UTM before reprojection';

-- Distribution of a continuous index (ndvi, ndwi, mndwi) over the covered part
-- of the requested area: one row per non-empty value interval 0.01 wide over
-- [-1, 1] (eo_brain/core/distribution.py). Interval k holds
-- value_from <= value < value_to; the last one (0.99 to 1.00) also holds 1.
-- The pixels counted are those result.coverage counts as covered, so a job's
-- pixel_count adds up to its coverage.covered_px. Written in the transaction
-- that completes the job (JobRepository.complete); flood, lisa and analyses run
-- before this table existed have no rows. Read through agent.index_value_distribution.
-- Idempotent: re-running this file on an existing database adds the table.
CREATE TABLE IF NOT EXISTS index_value_bins (
    job_id      UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    bin_index   SMALLINT NOT NULL CHECK (bin_index BETWEEN 0 AND 199),
    value_from  NUMERIC(4,2) NOT NULL,
    value_to    NUMERIC(4,2) NOT NULL,
    pixel_count BIGINT NOT NULL CHECK (pixel_count > 0),
    area_m2     DOUBLE PRECISION NOT NULL CHECK (area_m2 > 0),
    PRIMARY KEY (job_id, bin_index),
    CHECK (value_from = -1 + bin_index * 0.01 AND value_to = value_from + 0.01)
);

COMMENT ON TABLE index_value_bins IS
    'Pixels of a continuous-index job per value interval of width 0.01 (non-empty intervals only), over the covered part of the requested area';
COMMENT ON COLUMN index_value_bins.value_from IS 'Lower edge of the interval, inclusive';
COMMENT ON COLUMN index_value_bins.value_to IS 'Upper edge, exclusive (value_from + 0.01); the last interval also holds 1';
COMMENT ON COLUMN index_value_bins.area_m2 IS 'pixel_count times the pixel area of the result grid (UTM metres)';
