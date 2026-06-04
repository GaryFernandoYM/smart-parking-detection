import { useState, useEffect, useRef } from 'react';
import type { SlotStatus } from '@/components/parking/ParkingSlot';

export interface ParkingSlotRaw {
  id: string;
  status: SlotStatus;
}

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected';

export interface UseParkingDataResult {
  slots: ParkingSlotRaw[] | null;
  connectionState: ConnectionState;
  isConnected: boolean;
  lastUpdated: Date | null;
}

// Maps the backend's Spanish class names to the component's English status type.
const CLASS_TO_STATUS: Record<string, SlotStatus> = {
  libre:         'free',
  ocupado:       'occupied',
  no_disponible: 'unavailable',
};

export function useParkingData(): UseParkingDataResult {
  const [slots, setSlots]                     = useState<ParkingSlotRaw[] | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle');
  const [lastUpdated, setLastUpdated]         = useState<Date | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Initial snapshot from REST endpoint.
  useEffect(() => {
    fetch('/data/espacios.json')
      .then((r) => r.json())
      .then((json: { espacios: { id: string; clase: string }[] }) => {
        setSlots(
          json.espacios.map((e) => ({
            id:     e.id,
            status: CLASS_TO_STATUS[e.clase] ?? 'unavailable',
          }))
        );
        setLastUpdated(new Date());
      })
      .catch(console.error);
  }, []);

  // WebSocket for live slot updates with automatic reconnect.
  useEffect(() => {
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setConnectionState('connecting');

      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;

      ws.onopen = () => {
        if (!cancelled) setConnectionState('connected');
      };

      ws.onmessage = (e) => {
        try {
          const fullState: Record<string, string> = JSON.parse(e.data);
          setSlots((prev) =>
            prev
              ? prev.map((s) => ({
                  ...s,
                  status: CLASS_TO_STATUS[fullState[s.id]] ?? s.status,
                }))
              : prev
          );
          setLastUpdated(new Date());
        } catch { /* ignore malformed frames */ }
      };

      ws.onclose = () => {
        if (!cancelled) {
          setConnectionState('disconnected');
          setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, []);

  return {
    slots,
    connectionState,
    isConnected: connectionState === 'connected',
    lastUpdated,
  };
}
