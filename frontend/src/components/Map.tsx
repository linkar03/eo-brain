import { useRef, useEffect, useCallback, useState } from 'react';
import maplibregl from 'maplibre-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import { Square, Trash2, MousePointer } from 'lucide-react';
import type { BBox, ResultLayer } from '../types';

interface MapProps {
  center?: [number, number];
  zoom?: number;
  onBBoxSelected: (bbox: BBox) => void;
  resultLayers: ResultLayer[];
  selectedBBox: BBox | null;
  flyToBBox?: BBox | null;
}

const SEVERITY_COLORS: Record<number, string> = {
  1: '#fbbf24',
  2: '#f97316',
  3: '#dc2626',
};

type DrawMode = 'simple_select' | 'draw_polygon';

// --- Basemaps -------------------------------------------------------------
// Both providers require attribution that stays visible on the map: the OSM tile
// policy says not to hide it "behind toggles", Esri wants the data sources
// "always visible". MapLibre's attribution control (bottom-right) lists the
// attribution of the visible sources; it is created with `compact: false`
// because MapLibre 4's default (`compact: true`) collapses it to an (i) button
// as soon as the map is dragged.

// OpenStreetMap standard tiles (default). The tile usage policy asks for
// "© OpenStreetMap contributors" with "OpenStreetMap" linking to the copyright page:
// https://operations.osmfoundation.org/policies/tiles/
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors';

// Esri World Imagery (opt-in: hidden until the user switches to it, so a default
// page load makes no request to Esri). For MapLibre, Esri asks for "Powered by
// Esri" plus the service's own `copyrightText`, read from the service metadata
// because the list of data providers changes:
// https://developers.arcgis.com/documentation/esri-and-data-attribution/interactive-maps/
const ESRI_IMAGERY_SERVICE = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer';
// The service's copyrightText as read on 2026-11-26; replaced by the live value
// the first time the layer is shown.
const ESRI_IMAGERY_COPYRIGHT_FALLBACK = 'Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community';

type BaseLayer = 'osm' | 'esri';
const DEFAULT_BASE_LAYER: BaseLayer = 'osm';
const BASE_LAYERS: Record<BaseLayer, { label: string; layerId: string }> = {
  osm: { label: 'OSM', layerId: 'osm-layer' },
  esri: { label: 'Esri', layerId: 'esri-imagery-layer' },
};
const BASE_LAYER_ORDER: BaseLayer[] = ['osm', 'esri'];

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string));
}

function esriImagerySource(copyrightText: string): maplibregl.RasterSourceSpecification {
  return {
    type: 'raster',
    tiles: [`${ESRI_IMAGERY_SERVICE}/tile/{z}/{y}/{x}`],
    tileSize: 256,
    attribution:
      'Powered by <a href="https://www.esri.com" target="_blank" rel="noopener noreferrer">Esri</a> | ' +
      escapeHtml(copyrightText),
  };
}

/**
 * Replace the fallback Esri attribution with the service's live copyrightText.
 * A source's attribution cannot be changed in place, so the source is re-added
 * (at the bottom of the stack, keeping its visibility). On failure the fallback stays.
 */
function refreshEsriAttribution(m: maplibregl.Map, signal: AbortSignal): void {
  fetch(`${ESRI_IMAGERY_SERVICE}?f=json`, { signal })
    .then(r => (r.ok ? r.json() : null))
    .then((info: { copyrightText?: unknown } | null) => {
      const text = typeof info?.copyrightText === 'string' ? info.copyrightText.trim() : '';
      if (!text || text === ESRI_IMAGERY_COPYRIGHT_FALLBACK || !m.getLayer('esri-imagery-layer')) return;
      const visibility = (m.getLayoutProperty('esri-imagery-layer', 'visibility') ?? 'visible') as 'visible' | 'none';
      m.removeLayer('esri-imagery-layer');
      m.removeSource('esri-imagery');
      m.addSource('esri-imagery', esriImagerySource(text));
      m.addLayer({ id: 'esri-imagery-layer', type: 'raster', source: 'esri-imagery', layout: { visibility } }, 'osm-layer');
    })
    .catch(() => { /* keep the fallback attribution */ });
}

export function Map({
  center = [12.5, 41.9],
  zoom = 10,
  onBBoxSelected,
  resultLayers,
  selectedBBox,
  flyToBBox
}: MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const draw = useRef<MapboxDraw | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [baseLayer, setBaseLayer] = useState<BaseLayer>(DEFAULT_BASE_LAYER);
  // Set once the Esri metadata request has been started (first time Esri is shown).
  const esriMeta = useRef<AbortController | null>(null);
  const [drawMode, setDrawMode] = useState<DrawMode>('simple_select');
  const [hasSelection, setHasSelection] = useState(false);

  // Initialize map once
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    const m = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'esri-imagery': esriImagerySource(ESRI_IMAGERY_COPYRIGHT_FALLBACK),
          'osm': {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: OSM_ATTRIBUTION
          }
        },
        // Only the default base layer is visible; MapLibre requests no tiles
        // for a source whose layers are all hidden.
        layers: [
          {
            id: 'esri-imagery-layer', type: 'raster', source: 'esri-imagery',
            layout: { visibility: DEFAULT_BASE_LAYER === 'esri' ? 'visible' : 'none' }
          },
          {
            id: 'osm-layer', type: 'raster', source: 'osm',
            layout: { visibility: DEFAULT_BASE_LAYER === 'osm' ? 'visible' : 'none' }
          }
        ]
      },
      center: center,
      zoom: zoom,
      attributionControl: {
        compact: false,
        customAttribution: '<a href="https://maplibre.org/" target="_blank" rel="noopener noreferrer">MapLibre</a>'
      }
    });

    m.addControl(new maplibregl.NavigationControl(), 'top-left');

    if (DEFAULT_BASE_LAYER === 'esri') {
      m.once('load', () => {
        esriMeta.current = new AbortController();
        refreshEsriAttribution(m, esriMeta.current.signal);
      });
    }

    const drawCtrl = new MapboxDraw({
      displayControlsDefault: false,
      controls: {},
      defaultMode: 'simple_select',
      styles: [
        {
          id: 'gl-draw-polygon-fill',
          type: 'fill',
          filter: ['all', ['==', '$type', 'Polygon']],
          paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.1 }
        },
        {
          id: 'gl-draw-polygon-stroke',
          type: 'line',
          filter: ['all', ['==', '$type', 'Polygon']],
          paint: { 'line-color': '#3b82f6', 'line-width': 2, 'line-dasharray': [2, 2] }
        },
        {
          id: 'gl-draw-polygon-and-line-vertex-active',
          type: 'circle',
          filter: ['all', ['==', 'meta', 'vertex'], ['==', '$type', 'Point']],
          paint: { 'circle-radius': 5, 'circle-color': '#3b82f6' }
        },
        {
          id: 'gl-draw-polygon-midpoint',
          type: 'circle',
          filter: ['all', ['==', 'meta', 'midpoint'], ['==', '$type', 'Point']],
          paint: { 'circle-radius': 3, 'circle-color': '#3b82f6' }
        }
      ]
    });

    m.addControl(drawCtrl as unknown as maplibregl.IControl, 'top-right');
    draw.current = drawCtrl;

    m.on('load', () => setMapLoaded(true));

    m.on('draw.create', (e) => {
      handleDrawComplete(e.features);
      setDrawMode('simple_select');
      setHasSelection(true);
    });
    m.on('draw.update', (e) => handleDrawComplete(e.features));
    m.on('draw.delete', () => {
      onBBoxSelected([0, 0, 0, 0]);
      setHasSelection(false);
    });
    m.on('draw.modechange', (e) => setDrawMode(e.mode as DrawMode));

    map.current = m;

    return () => {
      esriMeta.current?.abort();
      esriMeta.current = null;
      m.remove();
      map.current = null;
      draw.current = null;
    };
  }, []);

  // Fly to bbox when agent triggers it
  useEffect(() => {
    if (!map.current || !flyToBBox || !mapLoaded) return;
    const [west, south, east, north] = flyToBBox;
    // Ignore zero/invalid bboxes
    if (west === 0 && east === 0) return;

    map.current.fitBounds(
      [[west, south], [east, north]],
      { padding: 60, duration: 1500, maxZoom: 14 }
    );
  }, [flyToBBox, mapLoaded]);

  const handleDrawComplete = useCallback((features: GeoJSON.Feature[]) => {
    const feature = features[0];
    if (!feature || feature.geometry.type !== 'Polygon') return;

    const coords = (feature.geometry as GeoJSON.Polygon).coordinates[0];
    const lons = coords.map(c => c[0]);
    const lats = coords.map(c => c[1]);

    const bbox: BBox = [
      Math.min(...lons), Math.min(...lats),
      Math.max(...lons), Math.max(...lats)
    ];
    onBBoxSelected(bbox);
  }, [onBBoxSelected]);

  const startDrawing = useCallback(() => {
    if (!draw.current) return;
    draw.current.deleteAll();
    draw.current.changeMode('draw_polygon');
    setDrawMode('draw_polygon');
    setHasSelection(false);
    onBBoxSelected([0, 0, 0, 0]);
  }, [onBBoxSelected]);

  const clearSelection = useCallback(() => {
    if (!draw.current) return;
    draw.current.deleteAll();
    draw.current.changeMode('simple_select');
    setDrawMode('simple_select');
    setHasSelection(false);
    onBBoxSelected([0, 0, 0, 0]);
  }, [onBBoxSelected]);

  const cancelDrawing = useCallback(() => {
    if (!draw.current) return;
    draw.current.changeMode('simple_select');
    setDrawMode('simple_select');
  }, []);

  const cycleBaseLayer = useCallback(() => {
    if (!map.current) return;
    const currentIdx = BASE_LAYER_ORDER.indexOf(baseLayer);
    const newBase = BASE_LAYER_ORDER[(currentIdx + 1) % BASE_LAYER_ORDER.length];
    setBaseLayer(newBase);

    for (const key of BASE_LAYER_ORDER) {
      map.current.setLayoutProperty(BASE_LAYERS[key].layerId, 'visibility', key === newBase ? 'visible' : 'none');
    }

    // First time Esri is shown: fetch its live attribution (once per map).
    if (newBase === 'esri' && !esriMeta.current) {
      esriMeta.current = new AbortController();
      refreshEsriAttribution(map.current, esriMeta.current.signal);
    }
  }, [baseLayer]);

  // Sync result layers with map
  useEffect(() => {
    if (!map.current || !mapLoaded) return;
    const m = map.current;

    resultLayers.forEach(layer => {
      const rasterSourceId = `raster-${layer.id}`;
      const rasterLayerId = `raster-layer-${layer.id}`;
      const vectorSourceId = `vector-${layer.id}`;
      const vectorFillId = `vector-fill-${layer.id}`;
      const vectorLineId = `vector-line-${layer.id}`;

      if (!m.getSource(rasterSourceId)) {
        m.addSource(rasterSourceId, { type: 'raster', tiles: [layer.tileUrl], tileSize: 256 });
      }

      if (!m.getLayer(rasterLayerId)) {
        m.addLayer({
          id: rasterLayerId, type: 'raster', source: rasterSourceId,
          paint: { 'raster-opacity': layer.opacity ?? 0.8 },
          layout: { visibility: layer.visible ? 'visible' : 'none' }
        });
      } else {
        m.setLayoutProperty(rasterLayerId, 'visibility', layer.visible ? 'visible' : 'none');
        m.setPaintProperty(rasterLayerId, 'raster-opacity', layer.opacity ?? 0.8);
      }

      if (layer.vectorUrl) {
        if (!m.getSource(vectorSourceId)) {
          m.addSource(vectorSourceId, { type: 'vector', tiles: [layer.vectorUrl], minzoom: 0, maxzoom: 18 });
        }

        if (!m.getLayer(vectorFillId)) {
          m.addLayer({
            id: vectorFillId, type: 'fill', source: vectorSourceId, 'source-layer': 'alerts',
            paint: {
              'fill-color': ['match', ['get', 'severity'], 1, SEVERITY_COLORS[1], 2, SEVERITY_COLORS[2], 3, SEVERITY_COLORS[3], '#666'],
              'fill-opacity': 0.35
            },
            layout: { visibility: layer.vectorVisible ? 'visible' : 'none' }
          });

          m.on('click', vectorFillId, (e) => {
            if (!e.features?.length) return;
            const props = e.features[0].properties;
            if (!props) return;
            const severityLabel = { 1: 'Watch', 2: 'Suspect', 3: 'Alert' }[props.severity as number] || '?';
            const areaHa = ((props.area_m2 as number) / 10000).toFixed(2);
            const quadrant = props.quadrant as number | null;
            const qLabel = quadrant === 3 ? 'HL' : quadrant === 4 ? 'LH' : '';

            new maplibregl.Popup({ closeButton: true, maxWidth: '220px' })
              .setLngLat(e.lngLat)
              .setHTML(`
                <div class="p-1">
                  <div class="font-semibold text-gray-900">${severityLabel}</div>
                  <div class="text-gray-600 text-xs mt-1">Area: ${areaHa} ha</div>
                  <div class="text-gray-600 text-xs">Algorithm: ${(props.algorithm as string || '').toUpperCase()}</div>
                  ${qLabel ? `<div class="text-gray-600 text-xs">Quadrant: ${qLabel}</div>` : ''}
                </div>
              `)
              .addTo(m);
          });

          m.on('mouseenter', vectorFillId, () => { m.getCanvas().style.cursor = 'pointer'; });
          m.on('mouseleave', vectorFillId, () => { m.getCanvas().style.cursor = ''; });
        } else {
          m.setLayoutProperty(vectorFillId, 'visibility', layer.vectorVisible ? 'visible' : 'none');
        }

        if (!m.getLayer(vectorLineId)) {
          m.addLayer({
            id: vectorLineId, type: 'line', source: vectorSourceId, 'source-layer': 'alerts',
            paint: {
              'line-color': ['match', ['get', 'severity'], 1, SEVERITY_COLORS[1], 2, SEVERITY_COLORS[2], 3, SEVERITY_COLORS[3], '#666'],
              'line-width': 1.5, 'line-opacity': 0.9
            },
            layout: { visibility: layer.vectorVisible ? 'visible' : 'none' }
          });
        } else {
          m.setLayoutProperty(vectorLineId, 'visibility', layer.vectorVisible ? 'visible' : 'none');
        }
      }
    });

    // Cleanup removed layers
    const activeIds = new Set(resultLayers.map(l => l.id));
    const styleLayers = m.getStyle().layers || [];

    styleLayers.forEach(sl => {
      const match = sl.id.match(/^(raster-layer|vector-fill|vector-line)-(.+)$/);
      if (match && !activeIds.has(match[2])) {
        m.removeLayer(sl.id);
      }
    });

    Object.keys(m.getStyle().sources || {}).forEach(srcId => {
      const match = srcId.match(/^(raster|vector)-(.+)$/);
      if (match && !activeIds.has(match[2])) {
        const hasLayer = styleLayers.some(l => 'source' in l && l.source === srcId);
        if (!hasLayer) {
          try { m.removeSource(srcId); } catch { /* ignore */ }
        }
      }
    });
  }, [resultLayers, mapLoaded]);

  // Sync external bbox changes
  useEffect(() => {
    if (!draw.current) return;
    if (!selectedBBox || (selectedBBox[0] === 0 && selectedBBox[2] === 0)) {
      if (hasSelection) {
        draw.current.deleteAll();
        setHasSelection(false);
      }
    }
  }, [selectedBBox, hasSelection]);

  return (
    <div className="relative h-full w-full">
      <div ref={mapContainer} className="h-full w-full" />

      <div className="absolute top-3 right-3 z-10 flex flex-col gap-2">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <button
            onClick={drawMode === 'draw_polygon' ? cancelDrawing : startDrawing}
            className={`p-2.5 flex items-center justify-center transition-colors ${
              drawMode === 'draw_polygon' ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-100'
            }`}
            title={drawMode === 'draw_polygon' ? 'Cancel drawing' : 'Draw area'}
          >
            {drawMode === 'draw_polygon' ? <MousePointer className="w-5 h-5" /> : <Square className="w-5 h-5" />}
          </button>

          {hasSelection && (
            <button
              onClick={clearSelection}
              className="p-2.5 flex items-center justify-center text-gray-700 hover:bg-gray-100 border-t border-gray-200"
              title="Clear selection"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      <div className="absolute top-3 left-12 z-10">
        <button
          onClick={cycleBaseLayer}
          className="bg-white px-3 py-1.5 rounded-lg shadow-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
        >
          {BASE_LAYERS[baseLayer].label}
        </button>
      </div>

      {drawMode === 'draw_polygon' && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
          <div className="bg-gray-900/80 text-white px-4 py-2 rounded-lg text-sm">
            Click on the map to draw the analysis area. Double-click to finish.
          </div>
        </div>
      )}
    </div>
  );
}

export default Map;