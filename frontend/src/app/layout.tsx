import type { Metadata } from "next";
import { Noto_Sans, Source_Code_Pro } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const notoSans = Noto_Sans({
  variable: "--font-noto-sans",
  subsets: ["latin"],
});

const sourceCodePro = Source_Code_Pro({
  variable: "--font-source-code-pro",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "AgentOS — AI code-fix assistant for GitHub",
    template: "%s · AgentOS",
  },
  description:
    "Connect a GitHub repository, pick an issue, and AgentOS investigates the problem, writes the fix, and opens a pull request — ready for human review.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`dark ${notoSans.variable} ${sourceCodePro.variable}`} suppressHydrationWarning>
      <body className="antialiased">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}