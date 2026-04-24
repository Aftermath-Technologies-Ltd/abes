// Author: Bradley R. Kinnard
'use client';

import { fetchStats, Stats } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import {
    Activity,
    Brain,
    ChevronRight,
    Database,
    FileText,
    GitBranch,
    LogOut,
    MessageSquare,
    Search,
    User
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

interface ServiceCard {
  id: string;
  name: string;
  description: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  status: 'active' | 'coming-soon';
  stats?: string;
}

const services: ServiceCard[] = [
  {
    id: 'chat',
    name: 'Ingestion Debugger',
    description: 'Inject beliefs, observe ecology dynamics. Watch confidence, tension, and mutation in real time.',
    href: '/chat',
    icon: MessageSquare,
    status: 'active',
  },
  {
    id: 'documents',
    name: 'Documents',
    description: 'Upload and analyze documents. Extract facts, detect contradictions across sources.',
    href: '/documents',
    icon: FileText,
    status: 'coming-soon',
  },
  {
    id: 'explorer',
    name: 'Belief Explorer',
    description: 'Browse, search, and manage the belief ecology. View relationships and lineage.',
    href: '/explorer',
    icon: Search,
    status: 'coming-soon',
  },
  {
    id: 'integrations',
    name: 'Integrations',
    description: 'Connect external data sources via webhooks and APIs. Kafka, REST, and more.',
    href: '/integrations',
    icon: GitBranch,
    status: 'coming-soon',
  },
];

export default function HomePage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const { user, isLoading, isAuthenticated, logout } = useAuth();
  const router = useRouter();

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isLoading, isAuthenticated, router]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchStats().then(setStats).catch(() => {});

      // Check WebSocket connectivity
      const ws = new WebSocket('ws://localhost:8000/chat/ws');
      ws.onopen = () => setWsConnected(true);
      ws.onerror = () => setWsConnected(false);
      ws.onclose = () => setWsConnected(false);
      return () => ws.close();
    }
  }, [isAuthenticated]);

  // Show loading while checking auth
  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <Brain className="w-8 h-8 text-neutral-500 animate-pulse" />
      </div>
    );
  }

  // Don't render if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-neutral-100">
      {/* Header */}
      <header className="border-b border-[#1f1f1f]">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[#141414] border border-[#2a2a2a] flex items-center justify-center">
              <Brain className="w-5 h-5 text-neutral-400" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-neutral-100">ABES</h1>
              <p className="text-xs text-neutral-500">Adaptive Belief Ecology System</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs ${
              wsConnected
                ? 'bg-emerald-500/10 text-emerald-500'
                : 'bg-neutral-800 text-neutral-500'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-neutral-500'}`} />
              {wsConnected ? 'System Online' : 'Connecting...'}
            </div>
            {/* User Menu */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[#141414] rounded-lg border border-[#2a2a2a]">
              <User className="w-4 h-4 text-neutral-400" />
              <span className="text-sm text-neutral-300">{user?.name}</span>
            </div>
            <button
              onClick={logout}
              className="p-2 rounded-lg hover:bg-[#141414] text-neutral-500 hover:text-red-400 transition-colors"
              title="Sign out"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Stats Overview */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
            <StatCard label="Total Beliefs" value={stats.total_beliefs} />
            <StatCard label="Active" value={stats.active_beliefs} accent />
            <StatCard label="Avg Confidence" value={`${Math.round((stats.avg_confidence || 0) * 100)}%`} />
            <StatCard label="Clusters" value={stats.cluster_count} />
          </div>
        )}

        {/* Services Grid */}
        <div className="mb-6">
          <h2 className="text-sm font-medium text-neutral-400 mb-4">Services</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {services.map((service) => (
              <ServiceCardComponent key={service.id} service={service} />
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-10">
          <h2 className="text-sm font-medium text-neutral-400 mb-4">Quick Actions</h2>
          <div className="flex flex-wrap gap-3">
            <QuickAction href="/chat" icon={MessageSquare} label="New Session" />
            <QuickAction href="/api/docs" icon={Database} label="API Docs" external />
            <QuickAction href="#" icon={Activity} label="View Logs" disabled />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1f1f1f] mt-auto">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-neutral-600">
          <span>ABES v0.2.0</span>
          <span>Memory evolves with every interaction</span>
        </div>
      </footer>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: number | string; accent?: boolean }) {
  return (
    <div className="bg-[#141414] border border-[#1f1f1f] rounded-lg px-4 py-3">
      <p className="text-xs text-neutral-500 mb-1">{label}</p>
      <p className={`text-xl font-semibold ${accent ? 'text-emerald-500' : 'text-neutral-200'}`}>
        {value}
      </p>
    </div>
  );
}

function ServiceCardComponent({ service }: { service: ServiceCard }) {
  const Icon = service.icon;
  const isActive = service.status === 'active';

  const content = (
    <div className={`
      group relative bg-[#141414] border border-[#1f1f1f] rounded-xl p-5
      transition-all duration-200
      ${isActive ? 'hover:border-[#2a2a2a] hover:bg-[#1a1a1a] cursor-pointer' : 'opacity-60'}
    `}>
      <div className="flex items-start justify-between mb-3">
        <div className={`
          w-10 h-10 rounded-lg flex items-center justify-center
          ${isActive ? 'bg-[#1a1a1a] border border-[#2a2a2a]' : 'bg-[#1a1a1a]'}
        `}>
          <Icon className={`w-5 h-5 ${isActive ? 'text-neutral-300' : 'text-neutral-600'}`} />
        </div>
        {isActive ? (
          <ChevronRight className="w-5 h-5 text-neutral-600 group-hover:text-neutral-400 transition-colors" />
        ) : (
          <span className="text-[10px] uppercase tracking-wider text-neutral-600 bg-[#1a1a1a] px-2 py-1 rounded">
            Soon
          </span>
        )}
      </div>
      <h3 className={`font-medium mb-1 ${isActive ? 'text-neutral-200' : 'text-neutral-500'}`}>
        {service.name}
      </h3>
      <p className="text-sm text-neutral-500 leading-relaxed">
        {service.description}
      </p>
    </div>
  );

  if (isActive) {
    return <Link href={service.href}>{content}</Link>;
  }
  return content;
}

function QuickAction({
  href,
  icon: Icon,
  label,
  external,
  disabled
}: {
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  external?: boolean;
  disabled?: boolean;
}) {
  const className = `
    flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors
    ${disabled
      ? 'bg-[#141414] text-neutral-600 cursor-not-allowed'
      : 'bg-[#141414] border border-[#1f1f1f] text-neutral-300 hover:bg-[#1a1a1a] hover:border-[#2a2a2a]'
    }
  `;

  if (disabled) {
    return (
      <span className={className}>
        <Icon className="w-4 h-4" />
        {label}
      </span>
    );
  }

  if (external) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
        <Icon className="w-4 h-4" />
        {label}
      </a>
    );
  }

  return (
    <Link href={href} className={className}>
      <Icon className="w-4 h-4" />
      {label}
    </Link>
  );
}
