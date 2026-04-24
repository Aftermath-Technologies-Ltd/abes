// Author: Bradley R. Kinnard
'use client';

import { Belief } from '@/lib/api';
import { BeliefCard } from './BeliefCard';

interface BeliefListProps {
  beliefs: Belief[];
  isLoading: boolean;
  onSelect?: (belief: Belief) => void;
}

export function BeliefList({ beliefs, isLoading, onSelect }: BeliefListProps) {
  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
        <h2 className="text-xl font-semibold mb-4">Beliefs</h2>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-24 bg-gray-100 dark:bg-gray-700 rounded-lg animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Beliefs</h2>
        <span className="text-sm text-gray-500">{beliefs.length} total</span>
      </div>

      {beliefs.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No beliefs yet. Create one to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[600px] overflow-y-auto">
          {beliefs.map((belief) => (
            <BeliefCard
              key={belief.id}
              belief={belief}
              onClick={() => onSelect?.(belief)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
