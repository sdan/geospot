"use client";

import React, { useEffect, useState, useMemo } from "react";
import { Map, Marker, Overlay } from "pigeon-maps";

const LOCATIONS = [
  { lat: 23.274160, lng: 79.986495, city: "Panagar", country: "India" },
  { lat: 43.261569, lng: -8.820610, city: "Malpica", country: "Spain" },
  { lat: -19.123342, lng: -64.628402, city: "Tomina", country: "Bolivia" },
  { lat: 15.663155, lng: -88.163811, city: "Cuyamel", country: "Honduras" },
];

const KERNELS = [
  { name: "Continent", scale: 5000, color: "#0071e3" },
  { name: "Country", scale: 750, color: "#34c759" },
  { name: "Region", scale: 100, color: "#ff9500" },
  { name: "City", scale: 25, color: "#5ac8fa" },
  { name: "Street", scale: 1, color: "#af52de" },
];

const STAGES = [
  { name: "Early", weights: [1.0, 0.5, 0.2, 0.1, 0.0] },
  { name: "Mid", weights: [0.3, 1.0, 0.8, 0.3, 0.1] },
  { name: "Late", weights: [0.1, 0.3, 0.5, 1.0, 0.8] },
];

function haversine(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function kernelReward(distance: number, scale: number): number {
  return Math.exp(-distance / scale);
}

export default function GeoRewardMap() {
  const [locIdx, setLocIdx] = useState(0);
  const [stageIdx, setStageIdx] = useState(1);

  const loc = LOCATIONS[locIdx];
  const stage = STAGES[stageIdx];

  const distance = useMemo(() => {
    const pred = { lat: loc.lat + 0.8, lng: loc.lng + 1.2 };
    return haversine(pred.lat, pred.lng, loc.lat, loc.lng);
  }, [loc]);

  const rewards = KERNELS.map((k, i) => kernelReward(distance, k.scale) * stage.weights[i]);
  const total = rewards.reduce((a, b) => a + b, 0) / stage.weights.reduce((a, b) => a + b, 0);

  useEffect(() => {
    const timer = setTimeout(() => setLocIdx((i) => (i + 1) % LOCATIONS.length), 10000);
    return () => clearTimeout(timer);
  }, [locIdx]);

  return (
    <div className="geo-reward-map">
      <div className="geo-reward-map__label">Geodesic Reward Landscape</div>

      <div className="geo-reward-map__map">
        <Map
          center={[loc.lat, loc.lng]}
          zoom={5}
          height={400}
        >
          {/* Reward contour circles */}
          {[200, 150, 100, 50].map((size, i) => (
            <Overlay key={i} anchor={[loc.lat, loc.lng]} offset={[size/2, size/2]}>
              <div style={{
                width: size,
                height: size,
                borderRadius: "50%",
                border: `2px solid ${KERNELS[i]?.color || "#ccc"}`,
                backgroundColor: `${KERNELS[i]?.color || "#ccc"}22`,
              }} />
            </Overlay>
          ))}
          <Marker anchor={[loc.lat, loc.lng]} color="#ff3b30" />
        </Map>
      </div>

      <div className="geo-reward-map__kernels">
        <div className="geo-reward-map__kernels-header">
          <div className="geo-reward-map__kernels-title">Multi-Scale Kernel Breakdown</div>
          <button
            className="geo-reward-map__stage-btn"
            onClick={() => setStageIdx((s) => (s + 1) % STAGES.length)}
          >
            Stage: {stage.name}
          </button>
        </div>

        <div className="geo-reward-map__info">
          <span>Target: <strong>{loc.city}, {loc.country}</strong></span>
          <span style={{ marginLeft: 24 }}>Distance: <strong>{distance.toFixed(0)} km</strong></span>
          <span style={{ marginLeft: 24 }}>Reward: <strong style={{ color: total > 0.5 ? "#34c759" : "#ff3b30" }}>{total.toFixed(4)}</strong></span>
        </div>

        {KERNELS.map((k, i) => {
          const raw = kernelReward(distance, k.scale);
          return (
            <div key={k.name} className="geo-reward-map__kernel-level">
              <div className="geo-reward-map__kernel-name">{k.name}</div>
              <div className="geo-reward-map__bar-bg">
                <div
                  className="geo-reward-map__kernel-bar"
                  style={{ width: `${Math.max(raw * 100, 2)}%`, background: k.color }}
                />
              </div>
              <div className="geo-reward-map__kernel-value">{raw.toFixed(3)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
