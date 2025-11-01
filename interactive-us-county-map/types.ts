import type { Feature, Geometry } from 'geojson';

export interface County {
  id: string;
  name?: string;
  state?: string; // state abbreviation or full name when available
  // In a real application, you would fetch and add more properties like population, etc.
}

// This is the structure of the features we get after parsing the TopoJSON file.
export type CountyGeoJSON = Feature<Geometry, {}> & { id: string };

// This is the raw TopoJSON data structure from the fetch call.
// We only define the parts we need to keep it simple.
export interface UsAtlasData {
  type: 'Topology';
  objects: {
    counties: {
      type: 'GeometryCollection';
      geometries: any[];
    };
    states: {
      type: 'GeometryCollection';
      geometries: any[];
    };
    nation: {
      type: 'GeometryCollection';
      geometries: any[];
    }
  };
  arcs: any[];
}

export interface ChatMessage {
  author: 'user' | 'bot' | 'system';
  content: string;
}
