import { useState, useEffect, useMemo } from 'react';
import { 
  Play, 
  Square, 
  Trash2, 
  Eye, 
  EyeOff, 
  MapPin,
  Calendar,
  Layers,
  AlertTriangle,
  CheckCircle,
  Loader2,
  RefreshCw,
  Droplets,
  Leaf,
  Hexagon
} from 'lucide-react';
import type { BBox, JobStatusResponse, ResultLayer } from '../types';
import apiClient from '../api/client';

interface AlgorithmInfo {
  description: string;
  output_type: 'continuous' | 'discrete';
  required_bands?: string[];
}

interface SidebarProps {
  selectedBBox: BBox | null;
  onAnalyze: (algorithm: string, dateRange: { start: string; end: string }) => void;
  onClear: () => void;
  job: JobStatusResponse | null;
  isLoading: boolean;
  error: Error | null;
  resultLayers: ResultLayer[];
  onToggleLayer: (layerId: string) => void;
  onToggleVectorLayer: (layerId: string) => void;
  onRemoveLayer: (layerId: string) => void;
  onLayerOpacity: (layerId: string, opacity: number) => void;
  onRetry?: () => void;
  isPolling?: boolean;
  retryCount?: number;
}

const MAX_AREA_KM2 = 2500;

const ALGORITHM_CATEGORIES = {
  vegetation: ['ndvi', 'lisa'],
  water: ['ndwi', 'mndwi', 'flood'],
} as const;

type AlgorithmCategory = 'vegetation' | 'water' | 'other';

function estimateAreaKm2(bbox: BBox): number {
  const [west, south, east, north] = bbox;
  const latMid = (south + north) / 2;
  const widthKm = (east - west) * 111 * Math.cos((latMid * Math.PI) / 180);
  const heightKm = (north - south) * 111;
  return Math.abs(widthKm * heightKm);
}

function getDefaultDateRange() {
  const end = new Date();
  const start = new Date();
  start.setMonth(start.getMonth() - 1);
  
  return {
    start: start.toISOString().split('T')[0],
    end: end.toISOString().split('T')[0]
  };
}

function getAlgorithmCategory(algo: string): AlgorithmCategory {
  if (ALGORITHM_CATEGORIES.vegetation.includes(algo as typeof ALGORITHM_CATEGORIES.vegetation[number])) {
    return 'vegetation';
  }
  if (ALGORITHM_CATEGORIES.water.includes(algo as typeof ALGORITHM_CATEGORIES.water[number])) {
    return 'water';
  }
  return 'other';
}

export function Sidebar({
  selectedBBox,
  onAnalyze,
  onClear,
  job,
  isLoading,
  error,
  resultLayers,
  onToggleLayer,
  onToggleVectorLayer,
  onRemoveLayer,
  onLayerOpacity,
  onRetry,
  isPolling,
  retryCount = 0
}: SidebarProps) {
  const [algorithm, setAlgorithm] = useState('ndvi');
  const [algorithms, setAlgorithms] = useState<Record<string, AlgorithmInfo>>({});
  const [dateRange, setDateRange] = useState(getDefaultDateRange);

  useEffect(() => {
    apiClient.getAlgorithms()
      .then(data => setAlgorithms(data as Record<string, AlgorithmInfo>))
      .catch(console.error);
  }, []);

  const selectedAlgoInfo = algorithms[algorithm];
  const algorithmCategory = getAlgorithmCategory(algorithm);

  const areaInfo = useMemo(() => {
    if (!selectedBBox) return null;
    const area = estimateAreaKm2(selectedBBox);
    const isOverLimit = area > MAX_AREA_KM2;
    return { area, isOverLimit };
  }, [selectedBBox]);

  const handleSubmit = () => {
    if (!selectedBBox || areaInfo?.isOverLimit) return;
    onAnalyze(algorithm, {
      start: `${dateRange.start}T00:00:00Z`,
      end: `${dateRange.end}T23:59:59Z`
    });
  };

  const formatBBox = (bbox: BBox) => {
    return `${bbox[0].toFixed(4)}, ${bbox[1].toFixed(4)} → ${bbox[2].toFixed(4)}, ${bbox[3].toFixed(4)}`;
  };

  const canSubmit = selectedBBox && !isLoading && !areaInfo?.isOverLimit;

  const renderLegend = () => {
    if (selectedAlgoInfo?.output_type === 'discrete') {
      if (algorithmCategory === 'water') {
        return (
          <>
            <h3 className="text-xs font-medium text-gray-700 mb-2 flex items-center gap-1">
              <Droplets className="w-3 h-3" />
              Legenda Flood
            </h3>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-blue-200" />
                <span>Watch</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-blue-500" />
                <span>Suspect</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-blue-900" />
                <span>Alert</span>
              </div>
            </div>
          </>
        );
      }
      // LISA severity legend
      return (
        <>
          <h3 className="text-xs font-medium text-gray-700 mb-2 flex items-center gap-1">
            <Leaf className="w-3 h-3" />
            Legenda Anomalie
          </h3>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-yellow-400" />
              <span>Watch</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-orange-500" />
              <span>Suspect</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-red-600" />
              <span>Alert</span>
            </div>
          </div>
        </>
      );
    }

    if (algorithmCategory === 'water') {
      return (
        <>
          <h3 className="text-xs font-medium text-gray-700 mb-2 flex items-center gap-1">
            <Droplets className="w-3 h-3" />
            Legenda {algorithm.toUpperCase()}
          </h3>
          <div className="flex items-center gap-1 text-xs">
            <div className="flex-1 h-3 rounded" style={{
              background: 'linear-gradient(to right, #8B5A2B, #D2B48C, #E0E0E0, #ADD8E6, #4169E1, #00008B)'
            }} />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Secco</span>
            <span>Acqua</span>
          </div>
        </>
      );
    }

    // NDVI continuous legend
    return (
      <>
        <h3 className="text-xs font-medium text-gray-700 mb-2 flex items-center gap-1">
          <Leaf className="w-3 h-3" />
          Legenda NDVI
        </h3>
        <div className="flex items-center gap-1 text-xs">
          <div className="flex-1 h-3 rounded" style={{
            background: 'linear-gradient(to right, #8B4513, #FFD700, #90EE90, #228B22, #006400)'
          }} />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Basso</span>
          <span>Alto</span>
        </div>
      </>
    );
  };

  // Filter algorithms by category for rendering
  const vegetationAlgos = Object.entries(algorithms).filter(
    ([key]) => ALGORITHM_CATEGORIES.vegetation.includes(key as typeof ALGORITHM_CATEGORIES.vegetation[number])
  );
  const waterAlgos = Object.entries(algorithms).filter(
    ([key]) => ALGORITHM_CATEGORIES.water.includes(key as typeof ALGORITHM_CATEGORIES.water[number])
  );

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-full">
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <h1 className="text-lg font-bold text-gray-900">EO Brain Analysis</h1>
        <p className="text-xs text-gray-500 mt-1">Satellite Intelligence Platform</p>
      </div>

      <div className="p-4 space-y-4 border-b border-gray-200">
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            <MapPin className="inline w-3 h-3 mr-1" />
            Area di Analisi
          </label>
          {selectedBBox ? (
            <div className={`border rounded p-2 ${
              areaInfo?.isOverLimit 
                ? 'bg-red-50 border-red-300' 
                : 'bg-blue-50 border-blue-200'
            }`}>
              <code className={`text-xs break-all ${
                areaInfo?.isOverLimit ? 'text-red-800' : 'text-blue-800'
              }`}>
                {formatBBox(selectedBBox)}
              </code>
              {areaInfo && (
                <div className={`mt-1 text-xs ${
                  areaInfo.isOverLimit ? 'text-red-600 font-medium' : 'text-gray-500'
                }`}>
                  {areaInfo.area.toFixed(0)} km²
                  {areaInfo.isOverLimit && ` (max: ${MAX_AREA_KM2} km²)`}
                </div>
              )}
            </div>
          ) : (
            <div className="bg-gray-50 border border-gray-200 rounded p-2 text-center">
              <span className="text-xs text-gray-500">
                Disegna un rettangolo sulla mappa
              </span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              <Calendar className="inline w-3 h-3 mr-1" />
              Da
            </label>
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">A</label>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            <Layers className="inline w-3 h-3 mr-1" />
            Algoritmo
          </label>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
          >
            {vegetationAlgos.length > 0 && (
              <optgroup label="Vegetazione">
                {vegetationAlgos.map(([key, algo]) => (
                  <option key={key} value={key}>
                    {key.toUpperCase()} - {algo.description}
                  </option>
                ))}
              </optgroup>
            )}
            {waterAlgos.length > 0 && (
              <optgroup label="Acqua / Alluvioni">
                {waterAlgos.map(([key, algo]) => (
                  <option key={key} value={key}>
                    {key.toUpperCase()} - {algo.description}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          {selectedAlgoInfo && (
            <p className="text-xs text-gray-500 mt-1">
              Output: {selectedAlgoInfo.output_type === 'continuous' ? 'Mappa continua' : 'Classi di severità'}
              {selectedAlgoInfo.required_bands && (
                <span className="ml-1 text-gray-400">
                  ({selectedAlgoInfo.required_bands.join(', ')})
                </span>
              )}
            </p>
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Analizza
          </button>
          <button
            onClick={onClear}
            className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Square className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </div>

      {job && (
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-xs font-medium text-gray-700 mb-2">Job Progress</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className={`font-medium ${
                job.status === 'completed' ? 'text-green-600' :
                job.status === 'failed' ? 'text-red-600' :
                'text-blue-600'
              }`}>
                {job.status === 'completed' ? 'Completato' :
                 job.status === 'failed' ? 'Fallito' :
                 `${job.progress}%`}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  job.status === 'completed' ? 'bg-green-500' :
                  job.status === 'failed' ? 'bg-red-500' :
                  'bg-blue-500'
                }`}
                style={{ width: `${job.progress}%` }}
              />
            </div>
            {job.message && (
              <p className={`text-xs ${
                job.status === 'failed' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {job.message}
              </p>
            )}
            {job.status === 'completed' && (
              <div className="flex items-center gap-1 text-xs text-green-600">
                <CheckCircle className="w-3 h-3" />
                Analisi completata
              </div>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="p-4 border-b border-gray-200 bg-red-50">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-red-700">{error.message}</p>
              {retryCount > 0 && (
                <p className="text-xs text-red-500 mt-1">
                  Retry attempts: {retryCount}
                </p>
              )}
            </div>
            {onRetry && (
              <button
                onClick={onRetry}
                className="p-1 hover:bg-red-100 rounded"
                title="Retry"
              >
                <RefreshCw className="w-4 h-4 text-red-600" />
              </button>
            )}
          </div>
        </div>
      )}

      {isPolling && !job && (
        <div className="p-4 border-b border-gray-200 bg-blue-50">
          <div className="flex items-center gap-2 text-xs text-blue-600">
            <Loader2 className="w-3 h-3 animate-spin" />
            Reconnecting to server...
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="text-xs font-medium text-gray-700 mb-2">
          Layer Risultati ({resultLayers.length})
        </h3>
        {resultLayers.length === 0 ? (
          <p className="text-xs text-gray-400 text-center py-4">
            Nessun risultato ancora
          </p>
        ) : (
          <div className="space-y-2">
            {resultLayers.map(layer => (
              <div
                key={layer.id}
                className="p-2 bg-gray-50 rounded border border-gray-200 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate flex items-center gap-1">
                      {getAlgorithmCategory(layer.algorithm) === 'water' ? (
                        <Droplets className="w-3 h-3 text-blue-500" />
                      ) : (
                        <Leaf className="w-3 h-3 text-green-500" />
                      )}
                      {layer.algorithm.toUpperCase()}
                    </p>
                    <p className="text-xs text-gray-500">
                      {new Date(layer.createdAt).toLocaleTimeString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => onToggleLayer(layer.id)}
                      className="p-1 hover:bg-gray-200 rounded"
                      title={layer.visible ? 'Nascondi raster' : 'Mostra raster'}
                    >
                      {layer.visible ? (
                        <Eye className="w-4 h-4 text-blue-600" />
                      ) : (
                        <EyeOff className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                    {layer.vectorUrl && (
                      <button
                        onClick={() => onToggleVectorLayer(layer.id)}
                        className="p-1 hover:bg-gray-200 rounded"
                        title={layer.vectorVisible ? 'Nascondi vettori' : 'Mostra vettori'}
                      >
                        <Hexagon className={`w-4 h-4 ${layer.vectorVisible ? 'text-purple-600' : 'text-gray-400'}`} />
                      </button>
                    )}
                    <button
                      onClick={() => onRemoveLayer(layer.id)}
                      className="p-1 hover:bg-red-100 rounded"
                      title="Rimuovi"
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </button>
                  </div>
                </div>
                {layer.visible && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 w-12">Opacità</span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={(layer.opacity ?? 0.8) * 100}
                      onChange={(e) => onLayerOpacity(layer.id, Number(e.target.value) / 100)}
                      className="flex-1 h-1 bg-gray-300 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-xs text-gray-500 w-8">
                      {Math.round((layer.opacity ?? 0.8) * 100)}%
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-200 bg-gray-50">
        {renderLegend()}
      </div>
    </div>
  );
}

export default Sidebar;