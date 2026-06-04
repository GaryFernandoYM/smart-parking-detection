'use client';

import { useState, useEffect, useRef } from 'react';
import EvaluacionPanel from './EvaluacionPanel';

interface Espacio {
  id: string;
  clase: 'libre' | 'ocupado' | 'no_disponible';
  estado_inicial: string;
}

interface DataEstacionamiento {
  metadata: { total_espacios: number };
  espacios: Espacio[];
}

interface WsPayload {
  estados:       Record<string, Espacio['clase']>;
  track_ids:     Record<string, number | null>;
  tiempos_s:     Record<string, number>;
  fps:           number;
  frame:         number;
  total:         number;
  latencia_ms:   number;
  conf_promedio: number;
  detecciones:   number;
}

// ── Colores por estado ────────────────────────────────────────────────────────
const COLORES = {
  libre:         { fill: '#22c55e', stroke: '#16a34a' },
  ocupado:       { fill: '#ef4444', stroke: '#dc2626' },
  no_disponible: { fill: '#f59e0b', stroke: '#d97706' },
};

const ETIQUETAS: Record<Espacio['clase'], string> = {
  libre: 'Libre',
  ocupado: 'Ocupado',
  no_disponible: 'No disponible',
};

// ── Layout fijo del estacionamiento ──────────────────────────────────────────
const W  = 90;
const H  = 148;
const SK = 24;
const SP = 10;

type Pt = { x: number; y: number };

function makeSlot(x: number, y: number): Pt[] {
  return [
    { x: x + SK,     y: y },
    { x: x + SK + W, y: y },
    { x: x + W,      y: y + H },
    { x: x,          y: y + H },
  ];
}

function makeRect(x: number, y: number, w: number, h: number): Pt[] {
  return [
    { x: x,     y: y },
    { x: x + w, y: y },
    { x: x + w, y: y + h },
    { x: x,     y: y + h },
  ];
}

const ROW1_Y  = 60;
const ROW1_X0 = 55;

const ROW1_IDS = ['espacio_Q','espacio_R','espacio_S','espacio_T','espacio_U',
                  'espacio_V','espacio_W','espacio_X','espacio_Y','espacio_Z'];

const Z_RIGHT = ROW1_X0 + 9 * (W + SP) + SK + W;
const A_X     = Z_RIGHT - 12;

const B_SZ = 108;
const B_X  = A_X + (SK + W - B_SZ) / 2 + 80;
const B_Y  = ROW1_Y + H + 30;

const VW = B_X + B_SZ + 45;

const ROW2_Y = B_Y + B_SZ + 60;

const DEF_IDS = ['espacio_d','espacio_e','espacio_f'];

const LAYOUT: Record<string, Pt[]> = {};

ROW1_IDS.forEach((id, i) => {
  LAYOUT[id] = makeSlot(ROW1_X0 + i * (W + SP), ROW1_Y);
});

LAYOUT['espacio_a'] = makeSlot(A_X, ROW1_Y);
LAYOUT['espacio_b'] = makeRect(B_X, B_Y, B_SZ, B_SZ);

DEF_IDS.forEach((id, i) => {
  LAYOUT[id] = makeSlot(ROW1_X0 + (5 + i) * (W + SP), ROW2_Y);
});

const VH = ROW2_Y + H + 55;

function centroide(pts: Pt[]) {
  return {
    cx: pts.reduce((s, p) => s + p.x, 0) / pts.length,
    cy: pts.reduce((s, p) => s + p.y, 0) / pts.length,
  };
}

function letra(id: string) { return id.replace('espacio_', ''); }

function fmtTiempo(s: number): string {
  if (s < 60)   return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ── Componente ────────────────────────────────────────────────────────────────
export default function MapaEstacionamiento() {
  const [espacios, setEspacios]  = useState<Espacio[]>([]);
  const [total, setTotal]        = useState(0);
  const [seleccionado, setSelec] = useState<string | null>(null);
  const [wsStatus, setWsStatus]  = useState<'conectando' | 'conectado' | 'desconectado'>('desconectado');
  const [snapshot, setSnapshot]  = useState<string | null>(null);
  const [fps, setFps]            = useState(0);
  const [frameN, setFrameN]      = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [vistaVivo, setVistaVivo]    = useState<'off' | 'modelo' | 'raw'>('off');
  const [latencia, setLatencia]      = useState(0);
  const [confPromedio, setConf]      = useState(0);
  const [detecciones, setDetecciones]= useState(0);
  const [mostrarEval, setMostrarEval]= useState(false);
  const [hora, setHora]              = useState('');
  const [trackIds, setTrackIds]      = useState<Record<string, number | null>>({});
  const [tiempos, setTiempos]        = useState<Record<string, number>>({});
  const wsRef = useRef<WebSocket | null>(null);

  // Reloj
  useEffect(() => {
    const tick = () => setHora(new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Estado inicial desde archivo estático
  useEffect(() => {
    fetch('/data/espacios.json')
      .then((r) => r.json())
      .then((json: DataEstacionamiento) => {
        setEspacios(json.espacios);
        setTotal(json.metadata.total_espacios);
      })
      .catch(console.error);
  }, []);

  // WebSocket
  useEffect(() => {
    let cancelled = false;
    const connect = () => {
      if (cancelled) return;
      setWsStatus('conectando');
      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;
      ws.onopen    = () => setWsStatus('conectado');
      ws.onmessage = (e) => {
        try {
          const msg: WsPayload = JSON.parse(e.data);
          setEspacios((prev) =>
            prev.map((s) => msg.estados[s.id] ? { ...s, clase: msg.estados[s.id] } : s)
          );
          setFps(msg.fps);
          setFrameN(msg.frame);
          setTotalFrames(msg.total);
          setLatencia(msg.latencia_ms ?? 0);
          setConf(msg.conf_promedio ?? 0);
          setDetecciones(msg.detecciones ?? 0);
          if (msg.track_ids) setTrackIds(msg.track_ids);
          if (msg.tiempos_s) setTiempos(msg.tiempos_s);
        } catch { /* ignorar frames malformados */ }
      };
      ws.onclose = () => { setWsStatus('desconectado'); if (!cancelled) setTimeout(connect, 3000); };
      ws.onerror  = () => ws.close();
    };
    connect();
    return () => { cancelled = true; wsRef.current?.close(); };
  }, []);

  // Polling snapshot
  useEffect(() => {
    if (vistaVivo === 'off') return;
    const url = vistaVivo === 'modelo'
      ? 'http://localhost:8000/snapshot'
      : 'http://localhost:8000/snapshot/raw';
    const poll = async () => {
      try {
        const r    = await fetch(url);
        const data = await r.json();
        if (data.image) setSnapshot(data.image);
      } catch { /* backend no disponible */ }
    };
    poll();
    const id = setInterval(poll, 500);
    return () => clearInterval(id);
  }, [vistaVivo]);

  const conteo = {
    libre:         espacios.filter((e) => e.clase === 'libre').length,
    ocupado:       espacios.filter((e) => e.clase === 'ocupado').length,
    no_disponible: espacios.filter((e) => e.clase === 'no_disponible').length,
  };

  const activo = espacios.find((e) => e.id === seleccionado);

  const wsColor = wsStatus === 'conectado'
    ? 'var(--teal)'
    : wsStatus === 'conectando'
    ? 'var(--amber)'
    : 'var(--red)';
  const wsLabel = wsStatus === 'conectado' ? 'EN VIVO' : wsStatus === 'conectando' ? 'CONECTANDO' : 'SIN SEÑAL';

  const FS = VH * 0.055;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', position: 'relative', zIndex: 1 }}>

      {/* ══ HEADER ══════════════════════════════════════════════════════════ */}
      <header style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border-hi)', flexShrink: 0 }}>
        <div style={{ height: 2, background: 'linear-gradient(90deg, var(--teal) 0%, rgba(0,212,170,0.3) 60%, transparent 100%)' }} />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '9px 20px' }}>

          {/* Logo + title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 34, height: 34, background: 'var(--teal)', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, boxShadow: '0 0 12px rgba(0,212,170,0.35)' }}>
              <svg width="17" height="17" viewBox="0 0 24 24" fill="#05080D">
                <path d="M13 3H6v18h4v-6h3a5 5 0 000-10zm0 6H10V7h3a1 1 0 010 2z"/>
              </svg>
            </div>
            <div>
              <p style={{ fontFamily: 'var(--font-d)', fontWeight: 800, fontSize: 13, letterSpacing: '0.06em', color: 'var(--text)', lineHeight: 1 }}>
                UPEU PARKING CONTROL
              </p>
              <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', letterSpacing: '0.14em', marginTop: 3 }}>
                SMART PARKING · YOLO v8s
              </p>
            </div>
          </div>

          {/* Clock — centered */}
          <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', textAlign: 'center', pointerEvents: 'none' }}>
            <p style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 19, color: 'var(--teal)', letterSpacing: '0.06em', lineHeight: 1 }}>{hora}</p>
            <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', letterSpacing: '0.12em', marginTop: 3 }}>
              {new Date().toLocaleDateString('es-ES', { weekday: 'short', day: 'numeric', month: 'short' }).toUpperCase()}
            </p>
          </div>

          {/* Right controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>

            {/* View toggle */}
            <div style={{ display: 'flex', background: 'var(--bg)', border: '1px solid var(--border-hi)', borderRadius: 5, padding: 2, gap: 2 }}>
              {(['off', 'modelo', 'raw'] as const).map(v => {
                const active = vistaVivo === v;
                return (
                  <button key={v}
                    onClick={() => { setVistaVivo(v); if (v === 'off') setSnapshot(null); }}
                    style={{
                      fontFamily: 'var(--font-m)', fontSize: 9, fontWeight: 600,
                      padding: '4px 10px', borderRadius: 4,
                      letterSpacing: '0.1em', border: 'none', cursor: 'pointer',
                      transition: 'all 0.15s',
                      background: active ? (v === 'off' ? 'var(--border-hi)' : 'var(--teal)') : 'transparent',
                      color: active ? (v === 'off' ? 'var(--text)' : 'var(--bg)') : 'var(--text-sub)',
                    }}
                  >
                    {v === 'off' ? 'OFF' : v === 'modelo' ? 'YOLO' : 'RAW'}
                  </button>
                );
              })}
            </div>

            {/* WS status chip */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              fontFamily: 'var(--font-m)', fontSize: 9, fontWeight: 600, letterSpacing: '0.12em',
              padding: '5px 11px', borderRadius: 4,
              border: `1px solid ${wsColor}40`,
              color: wsColor,
              background: `${wsColor}10`,
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: 'currentColor', display: 'inline-block',
                boxShadow: wsStatus === 'conectado' ? '0 0 6px currentColor' : 'none',
              }} className={wsStatus !== 'conectado' ? 'do-blink' : ''} />
              {wsLabel}
            </div>
          </div>
        </div>
      </header>

      {/* ══ STATS STRIP ═════════════════════════════════════════════════════ */}
      <div style={{ display: 'flex', background: 'var(--surface2)', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        {([
          { label: 'LIBRES',          value: conteo.libre,         color: '#22c55e' },
          { label: 'OCUPADOS',        value: conteo.ocupado,       color: '#ef4444' },
          { label: 'NO DISPONIBLES',  value: conteo.no_disponible, color: '#f59e0b' },
          { label: 'TOTAL',           value: total,                color: 'var(--text-sub)' },
        ] as const).map((s, i) => (
          <div key={s.label} style={{
            flex: 1, display: 'flex', alignItems: 'center', gap: 10,
            padding: '7px 16px',
            borderLeft: `2px solid ${s.color}`,
            borderRight: i < 3 ? '1px solid var(--border)' : 'none',
          }}>
            <span style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 20, color: s.color, lineHeight: 1 }}>{s.value}</span>
            <span style={{ fontFamily: 'var(--font-m)', fontSize: 8, letterSpacing: '0.15em', color: 'var(--text-sub)' }}>{s.label}</span>
          </div>
        ))}

        {/* FPS / Frame readout */}
        <div style={{ padding: '7px 16px', borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 2, flexShrink: 0 }}>
          <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.1em', color: fps >= 15 ? 'var(--teal)' : fps > 0 ? 'var(--amber)' : 'var(--text-sub)' }}>
            {fps > 0 ? `${fps.toFixed(1)} FPS` : '— FPS'}
          </span>
          <span style={{ fontFamily: 'var(--font-m)', fontSize: 8, letterSpacing: '0.1em', color: 'var(--text-sub)' }}>
            {frameN} / {totalFrames || '—'}
          </span>
        </div>
      </div>

      {/* ══ MAIN AREA — cámara o mapa, nunca los dos ═══════════════════════ */}

      {vistaVivo !== 'off' ? (

        /* ── VISTA CÁMARA (pantalla completa) ── */
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#000' }} className="fade-up">
          {/* Title bar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '5px 12px', borderBottom: '1px solid var(--border)', background: 'var(--surface)', flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444', boxShadow: '0 0 6px #ef4444', display: 'inline-block' }} className="do-blink" />
              <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.14em', color: 'var(--text-sub)' }}>
                CCTV · {vistaVivo === 'modelo' ? 'DETECCIÓN ACTIVA' : 'SIN MODELO'}
              </span>
            </div>
            <span style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', letterSpacing: '0.1em' }}>
              {fps.toFixed(1)} fps · {frameN}/{totalFrames || '–'}
            </span>
          </div>
          {/* Imagen llenando todo el espacio restante */}
          {snapshot ? (
            <img
              src={`data:image/jpeg;base64,${snapshot}`}
              alt="Frame anotado del detector"
              style={{ flex: 1, minHeight: 0, width: '100%', objectFit: 'contain' }}
            />
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--text-sub)', letterSpacing: '0.12em' }}>
              <span className="do-blink">●</span> ESPERANDO SEÑAL…
            </div>
          )}
        </div>

      ) : (

        /* ── VISTA MAPA (pantalla completa) ── */
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 18px', display: 'flex', flexDirection: 'column', gap: 10 }}>

        {/* ── SVG Map ── */}
        <div style={{ borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border-hi)', position: 'relative', background: 'var(--surface)' }}>

          {/* Map title bar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '5px 12px', borderBottom: '1px solid var(--border)', background: 'var(--surface2)' }}>
            <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.16em', color: 'var(--teal)', fontWeight: 600 }}>MAPA DE ESTACIONAMIENTO</span>
            <span style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', letterSpacing: '0.1em' }}>UPEU · {total} ESP.</span>
          </div>

          {/* Floating count cards */}
          <div style={{ position: 'absolute', bottom: 10, left: 10, zIndex: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              { label: 'Libres',    value: conteo.libre,         color: COLORES.libre.fill },
              { label: 'Ocupados',  value: conteo.ocupado,       color: COLORES.ocupado.fill },
              { label: 'No disp.',  value: conteo.no_disponible, color: COLORES.no_disponible.fill },
            ].map(c => (
              <div key={c.label} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                background: 'rgba(5,8,13,0.84)',
                border: `1px solid ${c.color}45`,
                borderLeft: `2px solid ${c.color}`,
                borderRadius: '0 5px 5px 0',
                padding: '3px 10px',
                backdropFilter: 'blur(6px)',
              }}>
                <span style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 15, color: c.color, lineHeight: 1 }}>{c.value}</span>
                <span style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: c.color, opacity: 0.6, letterSpacing: '0.12em' }}>{c.label.toUpperCase()}</span>
              </div>
            ))}
          </div>

          {/* ── SVG PARKING MAP (logic identical to original) ── */}
          <svg
            viewBox={`0 0 ${VW} ${VH}`}
            className="w-full h-auto block"
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              <pattern id="concreto" x="0" y="0" width="80" height="80" patternUnits="userSpaceOnUse">
                <rect width="80" height="80" fill="#d6d9df" />
                <line x1="0" y1="0" x2="80" y2="0" stroke="#cacdd5" strokeWidth="1" />
                <line x1="0" y1="0" x2="0" y2="80" stroke="#cacdd5" strokeWidth="1" />
              </pattern>
              <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur stdDeviation="12" result="b" />
                <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>

            <rect x="0" y="0" width={VW} height={VH} fill="url(#concreto)" />
            <rect x="20" y="20" width={VW - 40} height={VH - 40} rx="12" fill="#bfc3cc" />

            <text x="38" y="44" fill="#6b7280" fontSize={FS * 0.52}
              fontFamily="system-ui, sans-serif" fontWeight="700" letterSpacing="2.5">
              ZONA DE ESTACIONAMIENTO · UPEU
            </text>

            {espacios.map((espacio) => {
              const pts = LAYOUT[espacio.id];
              if (!pts) return null;

              const color    = COLORES[espacio.clase];
              const isActivo = seleccionado === espacio.id;
              const puntos   = pts.map((p) => `${p.x},${p.y}`).join(' ');
              const { cx, cy } = centroide(pts);

              return (
                <g
                  key={espacio.id}
                  onClick={() => setSelec((prev) => prev === espacio.id ? null : espacio.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <polygon points={puntos} fill="#00000020" transform="translate(4,6)" />
                  {isActivo && (
                    <polygon points={puntos} fill={color.fill} opacity={0.4} filter="url(#glow)" />
                  )}
                  <polygon
                    points={puntos}
                    fill={color.fill}
                    fillOpacity={isActivo ? 0.92 : 0.72}
                    stroke="#ffffff"
                    strokeWidth={isActivo ? 7 : 4}
                    strokeLinejoin="round"
                    style={{ transition: 'fill-opacity 0.18s' }}
                  />
                  {espacio.clase === 'ocupado' && (
                    <image
                      href="/car-top.png"
                      x={cx - 38} y={cy - 58}
                      width={76} height={116}
                      style={{ pointerEvents: 'none' }}
                    />
                  )}
                  <text
                    x={cx} y={cy + (espacio.clase === 'ocupado' ? 62 : 14)}
                    textAnchor="middle"
                    fill="#fff"
                    fontSize={FS * 0.55}
                    fontFamily="system-ui, sans-serif"
                    fontWeight="700"
                    style={{ pointerEvents: 'none', textShadow: '0 1px 3px #0006' }}
                  >
                    {letra(espacio.id)}
                  </text>
                  {espacio.clase === 'ocupado' && (
                    <>
                      {trackIds[espacio.id] != null && (
                        <text
                          x={cx} y={cy + 78}
                          textAnchor="middle"
                          fill="#ffffffdd"
                          fontSize={FS * 0.43}
                          fontFamily="system-ui, sans-serif"
                          fontWeight="700"
                          style={{ pointerEvents: 'none' }}
                        >
                          ID #{trackIds[espacio.id]}
                        </text>
                      )}
                      {tiempos[espacio.id] != null && (
                        <text
                          x={cx} y={cy + 93}
                          textAnchor="middle"
                          fill="#fde68a"
                          fontSize={FS * 0.40}
                          fontFamily="system-ui, sans-serif"
                          fontWeight="600"
                          style={{ pointerEvents: 'none' }}
                        >
                          {fmtTiempo(tiempos[espacio.id])}
                        </text>
                      )}
                    </>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* ── Legend + slot detail ── */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, paddingBottom: 6 }}>
          <div style={{ display: 'flex', gap: 18 }}>
            {(['libre', 'ocupado', 'no_disponible'] as const).map(c => (
              <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: COLORES[c].fill, border: `1.5px solid ${COLORES[c].stroke}`, display: 'inline-block' }} />
                <span style={{ fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--text-sub)' }}>{ETIQUETAS[c]}</span>
              </div>
            ))}
          </div>

          {activo && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--surface2)', border: `1px solid ${COLORES[activo.clase].stroke}50`, borderRadius: 5, padding: '5px 14px' }}>
              <div style={{ width: 32, height: 32, borderRadius: 4, background: COLORES[activo.clase].fill, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 16, color: '#000' }}>
                {letra(activo.id)}
              </div>
              <div>
                <p style={{ fontFamily: 'var(--font-m)', fontSize: 12, color: 'var(--text)', fontWeight: 600 }}>{activo.id}</p>
                <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: COLORES[activo.clase].fill, letterSpacing: '0.08em' }}>
                  {ETIQUETAS[activo.clase].toUpperCase()}
                </p>
              </div>
              <button onClick={() => setSelec(null)} style={{ color: 'var(--text-sub)', background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, marginLeft: 4, lineHeight: 1, opacity: 0.7 }}>×</button>
            </div>
          )}
        </div>

        </div>
      )}

      {/* ══ EVAL TAB BUTTON ═════════════════════════════════════════════════ */}
      <div style={{ position: 'fixed', left: 0, top: '50%', transform: 'translateY(-50%)', zIndex: 30 }}>
        <button
          onClick={() => setMostrarEval(v => !v)}
          aria-label="Panel de evaluación"
          style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
            padding: '16px 8px',
            borderRadius: '0 8px 8px 0',
            border: `1px solid ${mostrarEval ? 'var(--teal)' : 'var(--border-hi)'}`,
            borderLeft: 'none',
            background: mostrarEval ? 'var(--teal)' : 'var(--surface)',
            color: mostrarEval ? 'var(--bg)' : 'var(--text-sub)',
            cursor: 'pointer',
            transition: 'all 0.2s',
            boxShadow: mostrarEval ? '3px 0 18px rgba(0,212,170,0.3)' : 'none',
          }}
        >
          <span style={{ fontFamily: 'var(--font-m)', fontSize: 12, fontWeight: 700 }}>
            {mostrarEval ? '◀' : '▶'}
          </span>
          <span style={{ fontFamily: 'var(--font-m)', fontSize: 7, letterSpacing: '0.18em', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
            EVAL
          </span>
        </button>
      </div>

      {/* ══ BACKDROP ════════════════════════════════════════════════════════ */}
      {mostrarEval && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.65)', zIndex: 40, backdropFilter: 'blur(3px)' }}
          onClick={() => setMostrarEval(false)}
        />
      )}

      {/* ══ EVAL PANEL ══════════════════════════════════════════════════════ */}
      <div style={{
        position: 'fixed', top: 0, left: 0, height: '100%', width: 420, maxWidth: '92vw',
        zIndex: 50, background: 'var(--surface)',
        borderRight: '1px solid var(--border-hi)',
        display: 'flex', flexDirection: 'column',
        transform: mostrarEval ? 'translateX(0)' : 'translateX(-100%)',
        transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        boxShadow: mostrarEval ? '8px 0 48px rgba(0,0,0,0.65)' : 'none',
      }}>
        <EvaluacionPanel
          fps={fps}
          latenciams={latencia}
          confPromedio={confPromedio}
          detecciones={detecciones}
          espacios={espacios}
          onCerrar={() => setMostrarEval(false)}
        />
      </div>

    </div>
  );
}
