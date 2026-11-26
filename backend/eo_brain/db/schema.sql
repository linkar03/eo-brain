-- EO Brain Job Persistence Schema
-- Requires: PostgreSQL + PostGIS

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TYPE job_status AS ENUM ('pending', 'processing', 'completed', 'failed');

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status job_status NOT NULL DEFAULT 'pending',
    algorithm VARCHAR(50) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    
    -- Spatial bounds (PostGIS)
    bbox GEOMETRY(POLYGON, 4326) NOT NULL,
    
    -- Temporal range
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    
    -- Processing params (JSON blob)
    params JSONB DEFAULT '{}',
    
    -- Progress tracking
    progress SMALLINT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    message TEXT,
    
    -- Result (populated on completion)
    result JSONB,
    cog_path TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- CRS used for processing
    output_crs VARCHAR(20)
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX idx_jobs_bbox ON jobs USING GIST(bbox);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();