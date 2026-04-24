// Author: Bradley R. Kinnard
'use client';

import { Belief } from '@/lib/api';
import { clsx } from 'clsx';

interface BeliefCardProps {
  belief: Belief;
  onClick?: () => void;
}

export function BeliefCard({ belief, onClick }: BeliefCardProps) {
  const tensionLevel =
    belief.tension < 0.3 ? 'low' : belief.tension < 0.7 ? 'medium' : 'high';

  return (
    <div
      className={clsx('belief-node cursor-pointer', belief.status)}
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-2">
        <span
          className={clsx(
            'text-xs font-medium px-2 py-0.5 rounded-full',
            {
              'bg-belief-active/20 text-belief-active':
                belief.status === 'active',
              'bg-belief-decaying/20 text-belief-decaying':
                belief.status === 'decaying',
              'bg-belief-deprecated/20 text-belief-deprecated':
                belief.status === 'deprecated',
              'bg-belief-mutated/20 text-belief-mutated':
                belief.status === 'mutated',
            }
          )}
        >
          {belief.status}
        </span>
        <span className="text-xs text-gray-500">
          {Math.round(belief.confidence * 100)}%
        </span>
      </div>

      <p className="text-sm mb-3 line-clamp-3">{belief.content}</p>

      <div className="space-y-2">
        {/* Tension bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 w-12">Tension</span>
          <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
            <div
              className={clsx('tension-bar', tensionLevel)}
              style={{ width: `${Math.min(belief.tension * 100, 100)}%` }}
            />
          </div>
        </div>

        {/* Tags */}
        {belief.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {belief.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="text-xs bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded"
              >
                {tag}
              </span>
            ))}
            {belief.tags.length > 3 && (
              <span className="text-xs text-gray-400">
                +{belief.tags.length - 3}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
