'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Box, Activity, GitCompare, ShieldAlert, Settings, LogOut } from 'lucide-react';
import { useStore } from '@/lib/store';

const navItems = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Models', href: '/models', icon: Box },
  { name: 'Predictions', href: '/predict/history', icon: Activity },
  { name: 'Compare', href: '/compare', icon: GitCompare },
  { name: 'Bias Analysis', href: '/bias', icon: ShieldAlert },
  { name: 'API Keys', href: '/settings/api-keys', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const logout = useStore((state) => state.logout);

  return (
    <div className="flex h-screen w-64 flex-col bg-gray-900 text-white">
      <div className="flex bg-gray-950 h-16 items-center border-b border-gray-800 px-6">
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">XAI Platform</h1>
      </div>
      
      <div className="flex-1 overflow-y-auto py-4">
        <nav className="space-y-1 px-3">
          {navItems.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive 
                    ? 'bg-indigo-600 text-white shadow-sm' 
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <item.icon className={`mr-3 h-5 w-5 flex-shrink-0 ${isActive ? 'text-indigo-200' : 'text-gray-400'}`} />
                {item.name}
              </Link>
            )
          })}
        </nav>
      </div>

      <div className="border-t border-gray-800 p-4">
        <button 
          onClick={logout}
          className="flex w-full items-center rounded-lg px-3 py-2 text-sm font-medium text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
        >
          <LogOut className="mr-3 h-5 w-5" />
          Sign Out
        </button>
      </div>
    </div>
  );
}
