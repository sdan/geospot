"use client";

import React from "react";
import GeoGuessingRLViz from "./components/GeoGuessingRLViz";

export default function Demo() {
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
          Visualization components for geolocation RL research
        </p>
      </header>

      <section style={{ marginBottom: 64 }}>
        <h2 style={{ fontSize: "1.2rem", fontWeight: 500, marginBottom: 16, color: "#1d1d1f" }}>
          1. GeoGuessingRLViz
        </h2>
        <p style={{ color: "#424245", marginBottom: 24, fontSize: "0.9rem" }}>
          Shows OSV5M images with multi-scale kernel rewards and tokenized model output.
          Click image to advance.
        </p>
        <GeoGuessingRLViz imagePath="/osv5m_samples" />
      </section>

      <footer style={{
        marginTop: 80,
        paddingTop: 24,
        borderTop: "1px solid #e8e8ed",
        textAlign: "center",
        color: "#86868b",
        fontSize: "0.8rem",
      }}>
        GeoSpot VLM Viz · Hazy Research
      </footer>
    </div>
  );
}
