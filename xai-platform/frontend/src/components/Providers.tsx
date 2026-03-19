'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { usePathname, useRouter } from "next/navigation";
import { useStore } from "@/lib/store";
import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";

const queryClient = new QueryClient();

export function Providers({ children }: { children: React.ReactNode }) {
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

  if (!mounted) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-12 w-12 bg-indigo-200 rounded-full mb-4"></div>
          <div className="h-4 w-32 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  const isAuthPage = pathname === '/login' || pathname === '/register';

  return (
    <QueryClientProvider client={queryClient}>
      {isAuthPage ? (
        children
      ) : (
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <div className="flex-1 overflow-auto">
            <main id="main-content" className="mx-auto max-w-7xl px-4 py-8 sm:px-6 md:px-8">
              {children}
            </main>
          </div>
        </div>
      )}
    </QueryClientProvider>
  );
}
