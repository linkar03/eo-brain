// Types for EO Brain Dashboard - synced with backend Pydantic models

export type BBox = [number, number, number, number]; // [west, south, east, north]

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface AnalysisRequest {
  bbox: BBox;
  start_date: string;
  end_date: string;
  algorithm: string;
  provider?: string;
  resolution_m?: number;
  apply_cloud_mask?: boolean;
  params?: Record<string, unknown>;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  message: string;
}

export interface VectorStats {
  feature_count: number;
  total_area_m2: number;
  by_severity: Record<number, number>;
  error?: string;
}

/** result.coverage of a completed job (backend: eo_brain/core/coverage.py). */
export interface JobCoverage {
  coverage_pct: number;
  low_coverage: boolean;
  low_coverage_below_pct: number;
  uncovered_masked_pct: number;
  uncovered_no_data_pct: number;
  scenes_found: number | null;
  scenes_used: number | null;
  acquisition_days_with_data: number;
  first_date_with_data: string | null;
  last_date_with_data: string | null;
  [key: string]: unknown;
}

export interface JobResult {
  summary: Record<string, {
    min: number | null;
    max: number | null;
    mean: number | null;
    count_1?: number;
    count_2?: number;
    count_3?: number;
  }>;
  bbox: BBox;
  metadata: {
    algorithm: string;
    neighborhood_m?: number;
    neighborhood_px?: number;
    bio_gate?: number;
    thresholds?: Record<string, number>;
    [key: string]: unknown;
  };
  crs: string;
  tile_url: string;
  vector_url?: string;
  vector_stats?: VectorStats;
  coverage?: JobCoverage;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  message: string | null;
  result: JobResult | null;
  tile_url: string | null;
  vector_url: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface Provider {
  catalog_url: string;
  collections: string[];
}

export interface Algorithm {
  description: string;
  output_type: 'continuous' | 'discrete';
  required_bands: string[];
}

export interface ResultLayer {
  id: string;
  tileUrl: string;
  vectorUrl?: string;
  bbox: BBox;
  algorithm: string;
  createdAt: string;
  visible: boolean;
  opacity?: number;
  vectorVisible?: boolean;
}

export interface AppState {
  bbox: BBox | null;
  isDrawing: boolean;
  currentJob: JobStatusResponse | null;
  resultLayers: ResultLayer[];
  algorithm: string;
  dateRange: { start: string; end: string };
}