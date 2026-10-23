import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A static site: hosted on any static host (Vercel, GitHub Pages, ...); uploads go to the separate API.
  output: "export",
  // Set NEXT_PUBLIC_BASE_PATH when served from a sub-path such as https://user.github.io/repository
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || undefined,
};

export default nextConfig;
