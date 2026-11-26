import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "CarInsight Pro | AI Vehicle Valuation",
  description: "AI-powered used vehicle valuation with XGBoost, analytics and SHAP explainability."
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
