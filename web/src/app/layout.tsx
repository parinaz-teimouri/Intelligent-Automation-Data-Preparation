import type { Metadata } from "next";
import { Martian_Mono, Mona_Sans } from "next/font/google";
import "./globals.css";

const sans = Mona_Sans({ subsets: ["latin", "latin-ext"], variable: "--font-sans" });
const mono = Martian_Mono({ subsets: ["latin", "latin-ext"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Messy data in, clean data out · Parinaz Teimouri",
  description:
    "Automatic data preparation that fixes spellings, prices stored as text, gaps and duplicates, and explains every change in English and German.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
