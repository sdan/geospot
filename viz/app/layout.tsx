import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "Geospot | Tinker Dashboard",
  description: "Training dashboard for Thinking Machines Tinker API",
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
      <body className="min-h-screen bg-background antialiased">{children}</body>
    </html>
  );
}
