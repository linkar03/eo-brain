-- EO Brain Vector Features Schema
-- Stores polygonized anomaly/flood regions for MVT serving
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