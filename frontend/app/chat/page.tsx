// Author: Bradley R. Kinnard
'use client';

import { ChatInterface } from '@/components/ChatInterface';
import { useAuth } from '@/lib/auth';
import { Brain } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function ChatPage() {
  const { isLoading, isAuthenticated, token } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isLoading, isAuthenticated, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <Brain className="w-8 h-8 text-neutral-500 animate-pulse" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return <ChatInterface token={token} />;
}
