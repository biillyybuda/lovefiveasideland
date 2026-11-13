import type { Metadata } from "next";
import { ResetPasswordPanel } from "@/components/reset-password-panel";

export const metadata: Metadata = {
  title: "Reset password | Love Five",
  robots: { index: false, follow: false },
  referrer: "no-referrer"
};

export default function ResetPasswordPage() {
  return <main className="main"><ResetPasswordPanel /></main>;
}
