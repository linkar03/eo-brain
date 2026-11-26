import axios, { AxiosError, AxiosInstance } from 'axios';
import type {
  AnalysisRequest,
  JobResponse,
  JobStatusResponse,
  Provider,
  Algorithm
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class APIError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public isRetryable: boolean
  ) {
    super(message);
    this.name = 'APIError';
  }
}

function createAPIClient(): AxiosInstance {
  const client = axios.create({
    baseURL: API_BASE,
    timeout: 30000,
    headers: { 'Content-Type': 'application/json' },
  });

  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<{ detail?: string }>) => {
      if (!error.response) {
        throw new APIError('Network error: unable to reach server', 0, true);
      }

      const status = error.response.status;
      let detail = error.response.data?.detail || error.message;

      if (typeof detail === 'object') {
        detail = JSON.stringify(detail);
      }
      const isRetryable = status >= 500 || status === 429;

      if (status === 400) {
        throw new APIError(`Validation error: ${detail}`, status, false);
      }
      if (status === 404) {
        throw new APIError('Resource not found', status, false);
      }
      if (status === 422) {
        throw new APIError(`Invalid request: ${detail}`, status, false);
      }
      if (status === 429) {
        throw new APIError('Too many requests, please slow down', status, true);
      }
      if (status >= 500) {
        throw new APIError(`Server error: ${detail}`, status, true);
      }

      throw new APIError(detail, status, isRetryable);
    }
  );

  return client;
}

const api = createAPIClient();

export const apiClient = {
  async health(): Promise<{ status: string; running_jobs: number }> {
    const { data } = await api.get('/health');
    return data;
  },

  async getProviders(): Promise<Record<string, Provider>> {
    const { data } = await api.get('/providers');
    return data;
  },

  async getAlgorithms(): Promise<Record<string, Algorithm>> {
    const { data } = await api.get('/algorithms');
    return data;
  },

  async submitAnalysis(request: AnalysisRequest): Promise<JobResponse> {
    const { data } = await api.post('/analyze', request);
    return data;
  },

  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const { data } = await api.get(`/jobs/${jobId}`);
    return data;
  },

  async listJobs(limit = 20): Promise<Array<{
    job_id: string;
    status: string;
    created_at: string;
    algorithm: string;
    progress: number;
  }>> {
    const { data } = await api.get('/jobs', { params: { limit } });
    return data;
  },

  async deleteJob(jobId: string): Promise<void> {
    await api.delete(`/jobs/${jobId}`);
  },

  async getTileInfo(jobId: string): Promise<{
    bounds: [number, number, number, number];
    crs: string;
    minzoom: number;
    maxzoom: number;
  }> {
    const { data } = await api.get(`/tiles/${jobId}/info`);
    return data;
  },

  async getTileBounds(jobId: string): Promise<{
    bounds: [number, number, number, number];
    center: [number, number];
  }> {
    const { data } = await api.get(`/tiles/${jobId}/bounds`);
    return data;
  },

  // Vector endpoints
  async getVectorGeoJSON(jobId: string, minSeverity = 1): Promise<GeoJSON.FeatureCollection> {
    const { data } = await api.get(`/vectors/${jobId}.geojson`, {
      params: { min_severity: minSeverity }
    });
    return data;
  },

  async getVectorStats(jobId: string): Promise<{
    job_id: string;
    algorithm: string;
    total_features: number;
    total_area_m2: number;
    by_severity: Record<number, { count: number; total_area_m2: number; avg_area_m2: number }>;
  }> {
    const { data } = await api.get(`/vectors/${jobId}/stats`);
    return data;
  },

  async getVectorCount(jobId: string): Promise<{ job_id: string; feature_count: number }> {
    const { data } = await api.get(`/vectors/${jobId}/count`);
    return data;
  },

  isRetryableError(error: unknown): boolean {
    return error instanceof APIError && error.isRetryable;
  }
};

export default apiClient;