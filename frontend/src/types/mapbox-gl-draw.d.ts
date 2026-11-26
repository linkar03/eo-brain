// Type declarations for @mapbox/mapbox-gl-draw with MapLibre
declare module '@mapbox/mapbox-gl-draw' {
  import type { IControl, Map } from 'maplibre-gl';

  interface DrawOptions {
    displayControlsDefault?: boolean;
    controls?: {
      point?: boolean;
      line_string?: boolean;
      polygon?: boolean;
      trash?: boolean;
      combine_features?: boolean;
      uncombine_features?: boolean;
    };
    defaultMode?: string;
    modes?: Record<string, unknown>;
    styles?: unknown[];
    userProperties?: boolean;
  }

  class MapboxDraw implements IControl {
    constructor(options?: DrawOptions);
    
    onAdd(map: Map): HTMLElement;
    onRemove(map: Map): void;
    
    add(geojson: GeoJSON.Feature | GeoJSON.FeatureCollection): string[];
    get(featureId: string): GeoJSON.Feature | undefined;
    getAll(): GeoJSON.FeatureCollection;
    delete(ids: string | string[]): this;
    deleteAll(): this;
    set(featureCollection: GeoJSON.FeatureCollection): string[];
    
    changeMode(mode: string, options?: Record<string, unknown>): this;
    getMode(): string;
    
    trash(): this;
    combineFeatures(): this;
    uncombineFeatures(): this;
    
    getSelectedIds(): string[];
    getSelected(): GeoJSON.FeatureCollection;
    getSelectedPoints(): GeoJSON.FeatureCollection;
    
    setFeatureProperty(featureId: string, property: string, value: unknown): this;

    static modes: {
      draw_line_string: unknown;
      draw_polygon: unknown;
      draw_point: unknown;
      simple_select: unknown;
      direct_select: unknown;
    };
  }

  export default MapboxDraw;
}