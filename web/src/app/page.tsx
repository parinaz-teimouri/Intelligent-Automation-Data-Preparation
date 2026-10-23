"use client";

import { Benchmark, Footer, Header, Hero, HowItWorks } from "@/components/Site";
import { Workspace } from "@/components/Workspace";
import { LangProvider } from "@/lib/lang";

export default function Page() {
  return (
    <LangProvider>
      <Header />
      <main>
        <Hero />
        <Workspace />
        <HowItWorks />
        <Benchmark />
      </main>
      <Footer />
    </LangProvider>
  );
}
