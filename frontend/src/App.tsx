import { useState, useCallback } from 'react';
import Map from './components/Map';
import Sidebar from './components/Sidebar';
import { AgentChat } from './components/AgentChat';
import { useJob } from './hooks/useJob';
import apiClient from './api/client';
import type { BBox, ResultLayer, JobStatusResponse } from './types';

function App() {
  const [selectedBBox, setSelectedBBox] = useState<BBox | null>(null);
  const [resultLayers, setResultLayers] = useState<ResultLayer[]>([]);
  const [flyToBBox, setFlyToBBox] = useState<BBox | null>(null);

  const addLayerFromJob = useCallback((completedJob: JobStatusResponse, bboxFallback?: BBox | null) => {
    const tileUrl = completedJob.tile_url || completedJob.result?.tile_url;
    const vectorUrl = completedJob.vector_url || completedJob.result?.vector_url;

    if (!tileUrl || !completedJob.job_id) {
      console.error('Missing tile_url in job response');
      return;
    }

    const newLayer: ResultLayer = {
      id: completedJob.job_id,
      tileUrl,
      vectorUrl: vectorUrl || undefined,
      bbox: completedJob.result?.bbox || bboxFallback || [0, 0, 0, 0],
      algorithm: completedJob.result?.metadata?.algorithm || 'analysis',
      createdAt: completedJob.completed_at || new Date().toISOString(),
      visible: true,
      opacity: 0.8,
      vectorVisible: false,
    };

    setResultLayers(prev => {
      if (prev.some(layer => layer.id === newLayer.id)) return prev;
      return [newLayer, ...prev];
    });
  }, []);

  const {
    job,
    isLoading,
    error,
    isPolling,
    retryCount,
    submitAnalysis,
    reset,
    retry
  } = useJob({
    onComplete: useCallback((completedJob: JobStatusResponse) => {
      addLayerFromJob(completedJob, selectedBBox);
    }, [selectedBBox, addLayerFromJob]),

    onError: useCallback((err: Error) => {
      console.error('Job failed:', err);
    }, [])
  });

  const handleAgentJobCompleted = useCallback((jobId: string, result: any) => {
    const syntheticJob: JobStatusResponse = {
      job_id: jobId,
      status: 'completed',
      progress: 100,
      message: null,
      result,
      tile_url: result?.tile_url,
      vector_url: result?.vector_url,
      created_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
    };
    addLayerFromJob(syntheticJob);
  }, [addLayerFromJob]);

  // Agent requests map to fly to analysis area
  const handleAgentFlyTo = useCallback((bbox: BBox) => {
    setFlyToBBox(bbox);
  }, []);

  const handleBBoxSelected = useCallback((bbox: BBox) => {
    setSelectedBBox(bbox);
  }, []);

  const handleAnalyze = useCallback(async (
    algorithm: string,
    dateRange: { start: string; end: string }
  ) => {
    if (!selectedBBox) return;
    try {
      await submitAnalysis(selectedBBox, dateRange, algorithm);
    } catch {
      // Error handled by hook
    }
  }, [selectedBBox, submitAnalysis]);

  const handleClear = useCallback(() => {
    setSelectedBBox(null);
    reset();
  }, [reset]);

  const handleToggleLayer = useCallback((layerId: string) => {
    setResultLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
    ));
  }, []);

  const handleToggleVectorLayer = useCallback((layerId: string) => {
    setResultLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, vectorVisible: !layer.vectorVisible } : layer
    ));
  }, []);

  const handleRemoveLayer = useCallback((layerId: string) => {
    setResultLayers(prev => prev.filter(layer => layer.id !== layerId));
    apiClient.deleteJob(layerId).catch(console.error);
  }, []);

  const handleLayerOpacity = useCallback((layerId: string, opacity: number) => {
    setResultLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, opacity } : layer
    ));
  }, []);

  return (
    <div className="h-screen w-screen flex overflow-hidden">
      <Sidebar
        selectedBBox={selectedBBox}
        onAnalyze={handleAnalyze}
        onClear={handleClear}
        job={job}
        isLoading={isLoading}
        error={error}
        isPolling={isPolling}
        retryCount={retryCount}
        onRetry={retry}
        resultLayers={resultLayers}
        onToggleLayer={handleToggleLayer}
        onToggleVectorLayer={handleToggleVectorLayer}
        onRemoveLayer={handleRemoveLayer}
        onLayerOpacity={handleLayerOpacity}
      />

      <div className="flex-1 relative">
        <Map
          center={[12.49, 41.89]}
          zoom={6}
          onBBoxSelected={handleBBoxSelected}
          resultLayers={resultLayers}
          selectedBBox={selectedBBox}
          flyToBBox={flyToBBox}
        />

        {isLoading && job && (
          <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 flex items-center gap-3 z-[1000]">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent" />
            <div>
              <p className="text-sm font-medium text-gray-900">{job.message}</p>
              <p className="text-xs text-gray-500">{job.progress}% completato</p>
            </div>
          </div>
        )}

        <AgentChat
          onJobCompleted={handleAgentJobCompleted}
          onFlyTo={handleAgentFlyTo}
        />
      </div>
    </div>
  );
}

export default App;