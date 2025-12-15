"use client";

import GeoGuessingRLViz from "../components/GeoGuessingRLViz";
import GeoRewardMap from "../components/GeoRewardMap";

export default function Page() {
  return (
    <div style={{
      maxWidth: 1000,
      margin: "0 auto",
      padding: "40px 24px",
      fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
    }}>
      <header style={{ marginBottom: 48, textAlign: "center" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: 600, marginBottom: 8 }}>
          GeoSpot VLM Viz
        </h1>
        <p style={{ color: "#86868b", fontSize: "1rem" }}>
          Geolocation RL visualization
        </p>
      </header>

      <section style={{ marginBottom: 64 }}>
        <h2 style={{ fontSize: "1.1rem", fontWeight: 500, marginBottom: 16 }}>
          1. GeoGuessingRLViz
        </h2>
        <GeoGuessingRLViz imagePath="/osv5m_samples" />
      </section>

      <section style={{ marginBottom: 64 }}>
        <h2 style={{ fontSize: "1.1rem", fontWeight: 500, marginBottom: 16 }}>
          2. GeoRewardMap
        </h2>
        <GeoRewardMap />
      </section>
    </div>
  );
}
