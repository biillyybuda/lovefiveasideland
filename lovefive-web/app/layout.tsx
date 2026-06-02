import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Love Five",
  description: "Five-a-side league stats, ratings, match history and team insights."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <script
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('lovefive-theme');if(!t){t=window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}document.documentElement.dataset.theme=t}catch(e){}"
          }}
        />
        {children}
      </body>
    </html>
  );
}
