// Author: Bradley R. Kinnard
'use client';

import { Brain, Clock, Database, Info, RefreshCw, X, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SystemSettings {
  storage_backend: string;
  llm_provider: string;
  llm_fallback_enabled: boolean;
  decay_profile: string;
  embedding_model: string;
  total_beliefs: number;
  active_beliefs: number;
  avg_confidence: number;
  avg_tension: number;
  cluster_count: number;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchSettings();
    }
  }, [isOpen]);

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/bel/stats');
      if (response.ok) {
        const data = await response.json();
        setSettings({
          storage_backend: data.storage_backend || 'memory',
          llm_provider: data.llm_provider || 'ollama',
          llm_fallback_enabled: data.llm_fallback_enabled ?? true,
          decay_profile: data.decay_profile || 'moderate',
          embedding_model: data.embedding_model || 'all-MiniLM-L6-v2',
          total_beliefs: data.total_beliefs || 0,
          active_beliefs: data.active_beliefs || 0,
          avg_confidence: data.avg_confidence || 0,
          avg_tension: data.avg_tension || 0,
          cluster_count: data.cluster_count || 0,
        });
      } else {
        setError('Failed to load settings');
      }
    } catch (err) {
      setError('Cannot connect to backend');
    }
    setLoading(false);
  };

  if (!isOpen) return null;

  const decayDescriptions: Record<string, string> = {
    aggressive: 'Beliefs decay quickly (0.99/hr)',
    moderate: 'Balanced decay rate (0.995/hr)',
    conservative: 'Slow decay (0.999/hr)',
    persistent: 'Very slow decay (0.9999/hr)',
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border border-[#1f1f1f] rounded-xl w-full max-w-md mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1f1f1f]">
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-neutral-400" />
            <h2 className="text-lg font-semibold text-white">System Info</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-[#1f1f1f] text-neutral-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-5">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 text-neutral-500 animate-spin" />
            </div>
          ) : error ? (
            <div className="text-red-400 text-sm text-center py-4">{error}</div>
          ) : settings ? (
            <>
              {/* LLM Status */}
              <div className="bg-gradient-to-r from-[#141414] to-[#1a1a1a] rounded-lg p-4 border border-[#2a2a2a]">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                    <Brain className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium capitalize">{settings.llm_provider}</div>
                    <div className="text-xs text-neutral-500">
                      {settings.llm_fallback_enabled ? 'Fallback enabled' : 'No fallback'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-[#141414] rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-white">{settings.active_beliefs}</div>
                  <div className="text-xs text-neutral-500">Active</div>
                </div>
                <div className="bg-[#141414] rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-white">{Math.round(settings.avg_confidence * 100)}%</div>
                  <div className="text-xs text-neutral-500">Avg Confidence</div>
                </div>
                <div className="bg-[#141414] rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-white">{settings.cluster_count}</div>
                  <div className="text-xs text-neutral-500">Clusters</div>
                </div>
              </div>

              {/* Configuration Details */}
              <div className="space-y-2">
                <div className="flex items-center justify-between py-2 border-b border-[#1f1f1f]">
                  <div className="flex items-center gap-2 text-sm text-neutral-400">
                    <Database className="w-4 h-4" />
                    Storage
                  </div>
                  <div className="text-sm text-white capitalize">{settings.storage_backend}</div>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-[#1f1f1f]">
                  <div className="flex items-center gap-2 text-sm text-neutral-400">
                    <Clock className="w-4 h-4" />
                    Decay Profile
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-white capitalize">{settings.decay_profile}</div>
                    <div className="text-xs text-neutral-600">
                      {decayDescriptions[settings.decay_profile] || ''}
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center gap-2 text-sm text-neutral-400">
                    <Zap className="w-4 h-4" />
                    Embeddings
                  </div>
                  <div className="text-sm text-white font-mono text-xs">{settings.embedding_model}</div>
                </div>
              </div>

              {/* Footer Note */}
              <div className="bg-[#141414] rounded-lg p-3 text-xs text-neutral-500">
                <p>💡 Configuration is managed via environment variables in <code className="text-neutral-400">.env</code></p>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
