/**
 * GeoSpot VLM Viz
 *
 * Visualization components for geolocation RL research.
 * Built for interpretability demos and research presentations.
 *
 * Usage:
 *   import { GeoGuessingRLViz, GeoRewardMap, GeoPolicyPrediction } from './viz';
 *   import './viz/styles/viz.css';
 *
 * Optional: For map components, install Leaflet:
 *   npm install react-leaflet leaflet
 *   npm install -D @types/leaflet
 */

export { default as GeoGuessingRLViz } from './components/GeoGuessingRLViz';
export { default as GeoRewardMap } from './components/GeoRewardMap';
export { default as GeoPolicyPrediction } from './components/GeoPolicyPrediction';

// Re-export types
export type { } from './components/GeoGuessingRLViz';
