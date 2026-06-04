'use client';

import ParkingSlot from './ParkingSlot';
import type { SlotStatus } from './ParkingSlot';

export interface SlotData {
  id: string;
  status: SlotStatus;
}

interface ParkingZoneProps {
  name: string;
  topSlots: SlotData[];
  bottomSlots: SlotData[];
}

export default function ParkingZone({ name, topSlots, bottomSlots }: ParkingZoneProps) {
  return (
    <div className="w-fit bg-zinc-800 rounded-xl p-4">

      {/* Zone name */}
      <p className="text-zinc-400 text-[10px] font-semibold uppercase tracking-[0.18em] mb-2 pl-0.5">
        {name}
      </p>

      {/* Top row — rotated 180° so entry faces down toward the lane */}
      <div className="flex gap-1">
        {topSlots.map((slot) => (
          <div key={slot.id} style={{ transform: 'rotate(180deg)' }}>
            <ParkingSlot id={slot.id} status={slot.status} />
          </div>
        ))}
      </div>

      {/* Driving lane */}
      <div className="relative bg-zinc-700" style={{ height: '28px' }}>
        <div
          style={{
            position: 'absolute',
            top: '50%',
            transform: 'translateY(-50%)',
            left: 0,
            right: 0,
            height: '4px',
            background:
              'repeating-linear-gradient(90deg, #eab308 0px, #eab308 20px, transparent 20px, transparent 40px)',
          }}
        />
      </div>

      {/* Bottom row — normal orientation, entry faces up toward the lane */}
      <div className="flex gap-1">
        {bottomSlots.map((slot) => (
          <ParkingSlot key={slot.id} id={slot.id} status={slot.status} />
        ))}
      </div>

    </div>
  );
}
