// Author: Bradley R. Kinnard
'use client';

import { Stats } from '@/lib/api';

interface StatsPanelProps {
  stats?: Stats;
  isLoading: boolean;
}

function StatItem({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700 last:border-0">
      <span className="text-gray-600 dark:text-gray-400">{label}</span>
      <span className={`font-semibold ${color ?? ''}`}>{value}</span>
    </div>
  );
}

export function StatsPanel({ stats, isLoading }: StatsPanelProps) {
  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
        <h2 className="text-xl font-semibold mb-4">System Stats</h2>
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-6 bg-gray-100 dark:bg-gray-700 rounded animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
        <h2 className="text-xl font-semibold mb-4">System Stats</h2>
        <p className="text-gray-500">Unable to load stats</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
      <h2 className="text-xl font-semibold mb-4">System Stats</h2>

      <div className="space-y-1">
        <StatItem
          label="Total Beliefs"
          value={stats.total_beliefs.toLocaleString()}
        />
        <StatItem
          label="Active"
          value={stats.active_beliefs.toLocaleString()}
          color="text-belief-active"
        />
        <StatItem
          label="Deprecated"
          value={stats.deprecated_beliefs.toLocaleString()}
          color="text-belief-deprecated"
        />
        <StatItem
          label="Clusters"
          value={stats.cluster_count.toLocaleString()}
        />
        <StatItem
          label="Snapshots"
          value={stats.snapshot_count.toLocaleString()}
        />
      </div>

      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <h3 className="text-sm font-medium text-gray-500 mb-3">Averages</h3>

        <div className="space-y-3">
          {/* Confidence bar */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Confidence</span>
              <span>{Math.round(stats.avg_confidence * 100)}%</span>
            </div>
            <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-belief-active rounded-full h-2"
                style={{ width: `${stats.avg_confidence * 100}%` }}
              />
            </div>
          </div>

          {/* Tension bar */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Tension</span>
              <span>{Math.round(stats.avg_tension * 100)}%</span>
            </div>
            <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className={`rounded-full h-2 ${
                  stats.avg_tension < 0.3
                    ? 'bg-tension-low'
                    : stats.avg_tension < 0.7
                    ? 'bg-tension-medium'
                    : 'bg-tension-high'
                }`}
                style={{ width: `${Math.min(stats.avg_tension * 100, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
