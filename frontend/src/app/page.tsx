import MapaEstacionamiento from '@/components/MapaEstacionamiento';

export default function Home() {
  return (
    <div style={{ position: 'relative', zIndex: 1, width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <MapaEstacionamiento />
    </div>
  );
}
