'use client';

export type SlotStatus = 'free' | 'occupied' | 'unavailable';

interface ParkingSlotProps {
  id: string;
  status: SlotStatus;
  onClick?: () => void;
}

const STATUS_COLOR: Record<SlotStatus, string> = {
  free:        '#22c55e',
  occupied:    '#ef4444',
  unavailable: '#f59e0b',
};

export default function ParkingSlot({ id, status, onClick }: ParkingSlotProps) {
  const fill = STATUS_COLOR[status];

  return (
    <svg
      width="56"
      height="80"
      role="img"
      aria-label={`Slot ${id}`}
      onClick={onClick}
      className="cursor-pointer transition-transform duration-150 ease-out hover:scale-105"
    >
      {/* Asphalt-colored interior */}
      <rect x="0" y="0" width="56" height="80" fill={fill} />

      {/* Left painted line — inset 4 px from edge */}
      <line x1="4"  y1="0" x2="4"  y2="80" stroke="white" strokeWidth="3" opacity="0.7" />

      {/* Right painted line — inset 4 px from edge */}
      <line x1="52" y1="0" x2="52" y2="80" stroke="white" strokeWidth="3" opacity="0.7" />

      {status === 'occupied' ? (
        <>
          {/* Car body */}
          <rect x="12" y="18" width="32" height="44" rx="6" fill="rgba(0,0,0,0.35)" />

          {/* Windshield front */}
          <rect x="15" y="22" width="26" height="12" rx="3" fill="rgba(255,255,255,0.3)" />

          {/* Windshield rear */}
          <rect x="15" y="50" width="26" height="10" rx="3" fill="rgba(255,255,255,0.2)" />

          {/* Left wheels */}
          <rect x="8"  y="24" width="6" height="10" rx="2" fill="rgba(0,0,0,0.5)" />
          <rect x="8"  y="50" width="6" height="10" rx="2" fill="rgba(0,0,0,0.5)" />

          {/* Right wheels */}
          <rect x="42" y="24" width="6" height="10" rx="2" fill="rgba(0,0,0,0.5)" />
          <rect x="42" y="50" width="6" height="10" rx="2" fill="rgba(0,0,0,0.5)" />

          {/* Slot ID — small, bottom of slot */}
          <text
            x="28"
            y="76"
            textAnchor="middle"
            fill="white"
            fontWeight="bold"
            fontSize="9"
            fontFamily="system-ui, -apple-system, sans-serif"
            opacity="0.75"
          >
            {id}
          </text>
        </>
      ) : (
        <text
          x="28"
          y="48"
          textAnchor="middle"
          fill="white"
          fontWeight="bold"
          fontSize="18"
          fontFamily="system-ui, -apple-system, sans-serif"
        >
          {id}
        </text>
      )}
    </svg>
  );
}
