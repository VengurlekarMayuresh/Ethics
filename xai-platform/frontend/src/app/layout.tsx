'use client';
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { usePathname, useRouter } from "next/navigation";
import { useStore } from "@/lib/store";
import { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const queryClient = new QueryClient();

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const pathname = usePathname();
  const router = useRouter();
  const token = useStore((state) => state.token);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted) {
      const isAuthPage = pathname === '/login' || pathname === '/register';
      if (!token && !isAuthPage) {
        router.push('/login');
      } else if (token && isAuthPage) {
        router.push('/');
      }
    }
  }, [pathname, token, router, mounted]);

  if (!mounted) return null;

  const isAuthPage = pathname === '/login' || pathname === '/register';

  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50`}>
        <QueryClientProvider client={queryClient}>
          {isAuthPage ? (
            children
          ) : (
            <div className="flex h-screen overflow-hidden">
              <Sidebar />
              <div className="flex-1 overflow-auto">
                <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 md:px-8">
                  {children}
                </main>
              </div>
            </div>
          )}
        </QueryClientProvider>
      </body>
    </html>
  );
}
