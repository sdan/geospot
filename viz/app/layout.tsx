import type { Metadata } from "next";
import Script from "next/script";
import "../styles/viz.css";

export const metadata: Metadata = {
  title: "GeoSpot Viz",
  description: "Geolocation RL visualization components",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {process.env.NODE_ENV === "development" && (
          <>
            <Script
              src="//unpkg.com/react-grab/dist/index.global.js"
              strategy="beforeInteractive"
            />
            <Script
              src="//unpkg.com/@react-grab/claude-code/dist/client.global.js"
              strategy="lazyOnload"
            />
          </>
        )}
      </head>
      <body style={{ margin: 0, background: "#fff" }}>{children}</body>
    </html>
  );
}
