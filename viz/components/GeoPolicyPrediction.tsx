"use client";

import React, { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";

// Sample candidates
const SAMPLE_CANDIDATES = [
  { id: 1, city: "Tokyo", country: "JP", lat: 35.6762, lng: 139.6503, basePr: 0.35 },
  { id: 2, city: "Osaka", country: "JP", lat: 34.6937, lng: 135.5023, basePr: 0.22 },
  { id: 3, city: "Seoul", country: "KR", lat: 37.5665, lng: 126.9780, basePr: 0.18 },
  { id: 4, city: "Shanghai", country: "CN", lat: 31.2304, lng: 121.4737, basePr: 0.12 },
  { id: 5, city: "Beijing", country: "CN", lat: 39.9042, lng: 116.4074, basePr: 0.08 },
  { id: 6, city: "Hong Kong", country: "HK", lat: 22.3193, lng: 114.1694, basePr: 0.05 },
];

type ViewMode = "base" | "policy";

export default function GeoPolicyPrediction() {
  const [mode, setMode] = useState<ViewMode>("policy");
  const [iteration, setIteration] = useState(0);

  const candidates = useMemo(() => {
    return SAMPLE_CANDIDATES.map((c, i) => {
      const boost = i === 0 ? 0.15 : i === 2 ? 0.08 : -0.03;
      const policyPr = Math.max(0.01, Math.min(0.99, c.basePr + boost + (Math.random() - 0.5) * 0.05));
      return { ...c, policyPr };
    }).sort((a, b) => (mode === "base" ? b.basePr - a.basePr : b.policyPr - a.policyPr));
  }, [mode, iteration]);

  useEffect(() => {
    const timer = setInterval(() => setIteration((i) => i + 1), 8000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="geo-policy">
      <div className="geo-policy__header">
        <div className="geo-policy__title">Policy Re-ranking on Top-K</div>
        <div className="geo-policy__tabs">
          <button
            className={`geo-policy__tab ${mode === "base" ? "geo-policy__tab--active" : ""}`}
            onClick={() => setMode("base")}
          >
            Base
          </button>
          <button
            className={`geo-policy__tab ${mode === "policy" ? "geo-policy__tab--active" : ""}`}
            onClick={() => setMode("policy")}
          >
            Policy
          </button>
        </div>
      </div>

      <div className="geo-policy__grid">
        <div className="geo-policy__list">
          <div className="geo-policy__list-header">
            <span>Rank</span>
            <span>Location</span>
            <span>Probability</span>
          </div>
          <div className="geo-policy__rows">
            {candidates.map((c, i) => {
              const basePct = c.basePr * 100;
              const policyPct = c.policyPr * 100;
              return (
                <div key={c.id} className="geo-policy__row">
                  <div className="geo-policy__rank">#{i + 1}</div>
                  <div className="geo-policy__name">{c.city}, {c.country}</div>
                  <div className="geo-policy__bars">
                    <div
                      className="geo-policy__bar geo-policy__bar--base"
                      style={{ width: `${basePct}%` }}
                      title={`Base: ${basePct.toFixed(1)}%`}
                    />
                    <div
                      className="geo-policy__bar geo-policy__bar--policy"
                      style={{ width: `${policyPct}%` }}
                      title={`Policy: ${policyPct.toFixed(1)}%`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="geo-policy__map">
          <MapContainer
            center={[35, 125]}
            zoom={3}
            style={{ height: "100%", width: "100%" }}
            scrollWheelZoom={false}
          >
            <TileLayer
              attribution='&copy; OpenStreetMap'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {candidates.map((c, i) => {
              const pr = mode === "base" ? c.basePr : c.policyPr;
              return (
                <CircleMarker
                  key={c.id}
                  center={[c.lat, c.lng]}
                  radius={8 + pr * 20}
                  pathOptions={{
                    color: mode === "base" ? "#424245" : "#0071e3",
                    fillColor: mode === "base" ? "#86868b" : "#0071e3",
                    fillOpacity: 0.4 + pr * 0.5,
                    weight: i === 0 ? 3 : 1,
                  }}
                >
                  <Popup>
                    #{i + 1} {c.city}, {c.country}<br />
                    {mode === "base" ? "Base" : "Policy"}: {(pr * 100).toFixed(1)}%
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        </div>
      </div>

      <div className="geo-policy__footer">
        <div style={{ display: "flex", gap: "16px" }}>
          <div className="geo-policy__legend">
            <span className="geo-policy__dot geo-policy__dot--base" />
            <span>Base π₀</span>
          </div>
          <div className="geo-policy__legend">
            <span className="geo-policy__dot geo-policy__dot--policy" />
            <span>Policy πθ</span>
          </div>
        </div>
        <div className="geo-policy__version">Iteration: {iteration}</div>
      </div>
    </div>
  );
}
