# GeoSpot VLM Viz

Minimal visualization components for geolocation RL research.

## Components

| Component | Description |
|-----------|-------------|
| `GeoGuessingRLViz` | OSV5M images + multi-scale kernel rewards + tokenized output |
| `GeoRewardMap` | Geodesic reward landscape with curriculum stages |
| `GeoPolicyPrediction` | Top-K re-ranking visualization (base vs policy) |

## Usage

```tsx
import { GeoGuessingRLViz, GeoRewardMap, GeoPolicyPrediction } from './viz';
import './viz/styles/viz.css';

function Demo() {
  return (
    <div>
      <GeoGuessingRLViz imagePath="/osv5m_samples" />
      <GeoRewardMap targetLat={37.77} targetLng={-122.42} city="SF" country="USA" />
      <GeoPolicyPrediction />
    </div>
  );
}
```

## Optional: Map Support

For full map functionality:

```bash
npm install react-leaflet leaflet
npm install -D @types/leaflet
```

Add to your CSS or layout:
```css
@import 'leaflet/dist/leaflet.css';
```

## Files

```
viz/
├── components/
│   ├── GeoGuessingRLViz.tsx    # Main viz: image → kernel rewards → tokens
│   ├── GeoRewardMap.tsx         # Map with reward contours
│   └── GeoPolicyPrediction.tsx  # Base vs policy probability bars
├── styles/
│   └── viz.css                  # ~250 lines, CSS variables
├── assets/
│   └── osv5m_samples/           # Sample dashcam images
├── index.tsx                    # Exports
└── README.md
```

## Concepts

### Multi-Scale Kernels
Rewards decompose across geographic scales:
- **Continent** (5000km): exp(-d/5000)
- **Country** (750km): exp(-d/750)
- **Region** (100km): exp(-d/100)
- **City** (25km): exp(-d/25)
- **Street** (1km): exp(-d/1)

### Curriculum Stages
Training progression adjusts kernel weights:
- **Early**: Focus on continent-level accuracy
- **Mid**: Country + region precision
- **Late**: City + street-level localization

---
*Built for interpretability research.*
