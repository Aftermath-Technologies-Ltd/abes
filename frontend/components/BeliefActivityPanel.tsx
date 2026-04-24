// Author: Bradley R. Kinnard
'use client';

import { BeliefEvent } from '@/lib/api';
import clsx from 'clsx';
import { AlertCircle, Plus, RefreshCw, TrendingDown, X, Zap } from 'lucide-react';

interface BeliefActivityPanelProps {
  isOpen: boolean;
  onClose: () => void;
  events: BeliefEvent[];
}

export function BeliefActivityPanel({ isOpen, onClose, events }: BeliefActivityPanelProps) {
  console.log('BeliefActivityPanel render:', { isOpen, eventsCount: events.length, events });

  if (!isOpen) return null;

  const counts = {
    created: events.filter(e => e.event_type === 'created').length,
    reinforced: events.filter(e => e.event_type === 'reinforced').length,
    mutated: events.filter(e => e.event_type === 'mutated').length,
    tensions: events.filter(e => e.event_type === 'tension_changed').length,
  };

  console.log('BeliefActivityPanel counts:', counts);

  return (
    <div className="w-80 bg-[#0d0d0d] border-l border-[#1f1f1f] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-[#1f1f1f]">
        <span className="text-sm font-medium text-neutral-300">Belief Activity</span>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-[#1a1a1a] rounded transition-colors text-neutral-500 hover:text-neutral-300"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 p-3 border-b border-[#1f1f1f]">
        <StatBadge label="Created" count={counts.created} type="created" />
        <StatBadge label="Reinforced" count={counts.reinforced} type="reinforced" />
        <StatBadge label="Evolved" count={counts.mutated} type="mutated" />
        <StatBadge label="Tensions" count={counts.tensions} type="tension" />
      </div>

      {/* Event Feed */}
      <div className="flex-1 overflow-y-auto">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center px-6">
            <div className="w-10 h-10 rounded-full bg-[#1a1a1a] flex items-center justify-center mb-3">
              <RefreshCw className="w-5 h-5 text-neutral-600" />
            </div>
            <p className="text-neutral-600 text-sm">
              Belief activity will appear here as you chat
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {events.map((event, index) => (
              <EventCard key={`${event.belief_id}-${index}`} event={event} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatBadge({
  label,
  count,
  type
}: {
  label: string;
  count: number;
  type: 'created' | 'reinforced' | 'mutated' | 'tension';
}) {
  const colors = {
    created: count > 0 ? 'text-emerald-500' : 'text-neutral-600',
    reinforced: count > 0 ? 'text-sky-500' : 'text-neutral-600',
    mutated: count > 0 ? 'text-amber-500' : 'text-neutral-600',
    tension: count > 0 ? 'text-rose-500' : 'text-neutral-600',
  };

  return (
    <div className="flex items-center justify-between px-3 py-2 rounded bg-[#141414] border border-[#1f1f1f]">
      <span className="text-xs text-neutral-500">{label}</span>
      <span className={clsx("text-sm font-medium", colors[type])}>{count}</span>
    </div>
  );
}

function EventCard({ event }: { event: BeliefEvent }) {
  const config = getEventConfig(event.event_type);

  return (
    <div className="p-3 rounded-lg bg-[#141414] border border-[#1f1f1f] hover:border-[#2a2a2a] transition-colors">
      <div className="flex items-start gap-2.5">
        <div className={clsx(
          "w-6 h-6 rounded flex items-center justify-center flex-shrink-0 mt-0.5",
          config.bgClass
        )}>
          <config.icon className={clsx("w-3.5 h-3.5", config.iconClass)} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={clsx("text-xs font-medium", config.labelClass)}>
              {config.label}
            </span>
            <span className="text-[10px] text-neutral-600">
              {formatTime(event.timestamp)}
            </span>
          </div>
          <p className="text-xs text-neutral-400 line-clamp-2 leading-relaxed">
            {event.content}
          </p>
          <div className="flex items-center gap-3 mt-2">
            <span className="text-[10px] text-neutral-600">
              {Math.round(event.confidence * 100)}% confidence
            </span>
            {event.tension > 0.3 && (
              <span className="text-[10px] text-amber-600">
                {Math.round(event.tension * 100)}% tension
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function getEventConfig(type: BeliefEvent['event_type']) {
  const configs = {
    created: {
      icon: Plus,
      label: 'New Belief',
      bgClass: 'bg-emerald-500/10',
      iconClass: 'text-emerald-500',
      labelClass: 'text-emerald-500',
    },
    reinforced: {
      icon: Zap,
      label: 'Reinforced',
      bgClass: 'bg-sky-500/10',
      iconClass: 'text-sky-500',
      labelClass: 'text-sky-500',
    },
    mutated: {
      icon: RefreshCw,
      label: 'Evolved',
      bgClass: 'bg-amber-500/10',
      iconClass: 'text-amber-500',
      labelClass: 'text-amber-500',
    },
    deprecated: {
      icon: TrendingDown,
      label: 'Deprecated',
      bgClass: 'bg-neutral-500/10',
      iconClass: 'text-neutral-500',
      labelClass: 'text-neutral-500',
    },
    tension_changed: {
      icon: AlertCircle,
      label: 'Tension',
      bgClass: 'bg-rose-500/10',
      iconClass: 'text-rose-500',
      labelClass: 'text-rose-500',
    },
  };

  return configs[type] || configs.created;
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);

  if (diffSec < 60) return 'now';
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h`;
  return date.toLocaleDateString();
}
