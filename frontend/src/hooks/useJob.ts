import { useState, useEffect, useCallback, useRef } from 'react';
import apiClient from '../api/client';
import type { JobStatusResponse, BBox } from '../types';

interface UseJobOptions {
  initialPollingInterval?: number;
  maxPollingInterval?: number;
  backoffMultiplier?: number;
  maxRetries?: number;
  onComplete?: (job: JobStatusResponse) => void;
  onError?: (error: Error) => void;
}

interface JobState {
  jobId: string | null;
  job: JobStatusResponse | null;
  isLoading: boolean;
  error: Error | null;
  isPolling: boolean;
  retryCount: number;
}

const DEFAULT_OPTIONS = {
  initialPollingInterval: 2000,
  maxPollingInterval: 30000,
  backoffMultiplier: 1.5,
  maxRetries: 5,
};

export function useJob(options: UseJobOptions = {}) {
  const {
    initialPollingInterval = DEFAULT_OPTIONS.initialPollingInterval,
    maxPollingInterval = DEFAULT_OPTIONS.maxPollingInterval,
    backoffMultiplier = DEFAULT_OPTIONS.backoffMultiplier,
    maxRetries = DEFAULT_OPTIONS.maxRetries,
    onComplete,
    onError
  } = options;

  const [state, setState] = useState<JobState>({
    jobId: null,
    job: null,
    isLoading: false,
    error: null,
    isPolling: false,
    retryCount: 0,
  });

  const pollingRef = useRef<number | null>(null);
  const currentIntervalRef = useRef(initialPollingInterval);
  const consecutiveErrorsRef = useRef(0);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }
    currentIntervalRef.current = initialPollingInterval;
    consecutiveErrorsRef.current = 0;
    setState(prev => ({ ...prev, isPolling: false }));
  }, [initialPollingInterval]);

  const scheduleNextPoll = useCallback((id: string, pollFn: (id: string) => Promise<void>) => {
    pollingRef.current = window.setTimeout(() => {
      pollFn(id);
    }, currentIntervalRef.current);
  }, []);

  const pollJob = useCallback(async (id: string) => {
    try {
      const status = await apiClient.getJobStatus(id);

      consecutiveErrorsRef.current = 0;
      currentIntervalRef.current = initialPollingInterval;

      setState(prev => ({
        ...prev,
        job: status,
        error: null,
        retryCount: 0,
      }));

      if (status.status === 'completed') {
        stopPolling();
        onComplete?.(status);
      } else if (status.status === 'failed') {
        stopPolling();
        const err = new Error(status.message || 'Job failed');
        setState(prev => ({ ...prev, error: err }));
        onError?.(err);
      } else {
        scheduleNextPoll(id, pollJob);
      }
    } catch (err) {
      consecutiveErrorsRef.current++;

      currentIntervalRef.current = Math.min(
        currentIntervalRef.current * backoffMultiplier,
        maxPollingInterval
      );

      if (consecutiveErrorsRef.current >= maxRetries) {
        stopPolling();
        const error = new Error(`Failed to fetch job status after ${maxRetries} retries`);
        setState(prev => ({ ...prev, error, retryCount: consecutiveErrorsRef.current }));
        onError?.(error);
      } else {
        setState(prev => ({ ...prev, retryCount: consecutiveErrorsRef.current }));
        scheduleNextPoll(id, pollJob);
      }
    }
  }, [
    initialPollingInterval,
    maxPollingInterval,
    backoffMultiplier,
    maxRetries,
    onComplete,
    onError,
    stopPolling,
    scheduleNextPoll
  ]);

  useEffect(() => {
    if (!state.jobId) return;

    setState(prev => ({ ...prev, isPolling: true }));
    pollJob(state.jobId);

    return () => stopPolling();
  }, [state.jobId, pollJob, stopPolling]);

  const submitAnalysis = useCallback(async (
    bbox: BBox,
    dateRange: { start: string; end: string },
    algorithm: string,
    params?: Record<string, unknown>
  ) => {
    setState(prev => ({
      ...prev,
      isLoading: true,
      error: null,
      job: null,
    }));

    try {
      const request = {
        bbox,
        start_date: dateRange.start,
        end_date: dateRange.end,
        algorithm,
        // Native Sentinel-2 resolution for maximum quality
        resolution_m: 10,
        params
      };

      const response = await apiClient.submitAnalysis(request);

      setState(prev => ({
        ...prev,
        jobId: response.job_id,
        isLoading: false,
      }));

      return response.job_id;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to submit analysis');
      setState(prev => ({
        ...prev,
        error,
        isLoading: false,
      }));
      onError?.(error);
      throw error;
    }
  }, [onError]);

  const reset = useCallback(() => {
    stopPolling();
    setState({
      jobId: null,
      job: null,
      isLoading: false,
      error: null,
      isPolling: false,
      retryCount: 0,
    });
  }, [stopPolling]);

  const retry = useCallback(() => {
    if (state.jobId && !state.isPolling) {
      consecutiveErrorsRef.current = 0;
      currentIntervalRef.current = initialPollingInterval;
      setState(prev => ({ ...prev, error: null, isPolling: true }));
      pollJob(state.jobId);
    }
  }, [state.jobId, state.isPolling, initialPollingInterval, pollJob]);

  return {
    jobId: state.jobId,
    job: state.job,
    isLoading: state.isLoading,
    error: state.error,
    isPolling: state.isPolling,
    retryCount: state.retryCount,
    submitAnalysis,
    reset,
    retry,
  };
}

export default useJob;