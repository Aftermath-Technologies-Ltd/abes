// Author: Bradley R. Kinnard
'use client';

import { fetchClusters } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';

export function ClusterView() {
  const { data, isLoading } = useQuery({
    queryKey: ['clusters'],
    queryFn: fetchClusters,
  });

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
        <h2 className="text-xl font-semibold mb-4">Clusters</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="h-32 bg-gray-100 dark:bg-gray-700 rounded-xl animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  const clusters = data?.clusters ?? [];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Clusters</h2>
        <span className="text-sm text-gray-500">{clusters.length} total</span>
      </div>

      {clusters.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No clusters formed yet.
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {clusters.map((cluster) => (
            <div
              key={cluster.id}
              className="cluster-container hover:border-solid transition-all duration-200 cursor-pointer"
            >
              <div className="text-center">
                <div className="text-2xl font-bold text-belief-active">
                  {cluster.size}
                </div>
                <div className="text-xs text-gray-500 mt-1">beliefs</div>
              </div>
              <div className="text-xs text-gray-400 mt-2 truncate text-center">
                {cluster.id.slice(0, 8)}...
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
