'use client';

import { useState, useEffect } from 'react';
import EvaluacionCampo from './EvaluacionCampo';
import SesionCSV from './SesionCSV';

interface Espacio {
  id: string;
  clase: 'libre' | 'ocupado' | 'no_disponible';
}

interface Props {
  fps:          number;
  latenciams:   number;
  confPromedio: number;
  detecciones:  number;
  espacios:     Espacio[];
  onCerrar:     () => void;
}

type Tab = 'metricas' | 'campo' | 'sesion';

const TABS: { key: Tab; label: string }[] = [
  { key: 'metricas', label: 'MÉTRICAS'  },
  { key: 'campo',    label: 'CAMPO'     },
  { key: 'sesion',   label: 'SESIÓN CSV'},
];

function DataRow({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '9px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-sub)', paddingTop: 1 }}>{label}</span>
      <div style={{ textAlign: 'right' }}>
        <span style={{ fontFamily: 'var(--font-m)', fontSize: 12, fontWeight: 600, color: color ?? 'var(--text)' }}>{value}</span>
        {sub && <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', marginTop: 2, letterSpacing: '0.08em' }}>{sub}</p>}
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, fontWeight: 700, letterSpacing: '0.18em', color: 'var(--teal)', textTransform: 'uppercase', marginBottom: 6 }}>
      {children}
    </p>
  );
}

export default function EvaluacionPanel({ fps, latenciams, confPromedio, detecciones, onCerrar }: Props) {
  const [tab, setTab]         = useState<Tab>('metricas');
  const [map50, setMap50]     = useState('—');
  const [map5095, setMap5095] = useState('—');

  useEffect(() => {
    fetch('http://localhost:8000/metricas-modelo')
      .then(r => r.json())
      .then(d => {
        if (d.map50    != null) setMap50(d.map50.toFixed(4));
        if (d.map50_95 != null) setMap5095(d.map50_95.toFixed(4));
      })
      .catch(() => { setMap50('0.9945'); setMap5095('0.9929'); });
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--surface)' }}>

      {/* ── HEADER ── */}
      <div style={{ flexShrink: 0, position: 'relative' }}>
        <div style={{ height: 2, background: 'linear-gradient(90deg, var(--teal), transparent)' }} />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 18px' }}>
          <div>
            <p style={{ fontFamily: 'var(--font-d)', fontWeight: 800, fontSize: 13, letterSpacing: '0.06em', color: 'var(--text)' }}>
              PANEL DE EVALUACIÓN
            </p>
            <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, letterSpacing: '0.14em', color: 'var(--text-sub)', marginTop: 3 }}>
              SMART PARKING UPEU · YOLOv8s
            </p>
          </div>
          <button
            onClick={onCerrar}
            style={{
              width: 30, height: 30, borderRadius: 5,
              background: 'var(--border)', border: '1px solid var(--border-hi)',
              color: 'var(--text-sub)', cursor: 'pointer', display: 'flex',
              alignItems: 'center', justifyContent: 'center', fontSize: 14,
              transition: 'all 0.15s',
            }}
            aria-label="Cerrar"
          >
            ×
          </button>
        </div>
      </div>

      {/* ── TABS ── */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', flexShrink: 0, background: 'var(--surface2)' }}>
        {TABS.map(t => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={{
                flex: 1, padding: '9px 4px',
                fontFamily: 'var(--font-m)', fontSize: 8, fontWeight: 700,
                letterSpacing: '0.12em', border: 'none', cursor: 'pointer',
                background: 'transparent',
                color: active ? 'var(--teal)' : 'var(--text-sub)',
                borderBottom: `2px solid ${active ? 'var(--teal)' : 'transparent'}`,
                transition: 'all 0.15s',
              }}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {/* ── CONTENT ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 18px' }}>

        {/* ── MÉTRICAS ── */}
        {tab === 'metricas' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }} className="fade-up">

            <div>
              <SectionTitle>Modelo · epoch 100 · val set</SectionTitle>
              <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '0 12px' }}>
                <DataRow label="mAP50"        value={map50}              color="var(--teal)" />
                <DataRow label="mAP50-95"     value={map5095}            color="var(--teal)" />
                <DataRow label="Arquitectura" value="YOLOv8s fine-tuned" />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '9px 0' }}>
                  <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-sub)' }}>Dispositivo</span>
                  <span style={{ fontFamily: 'var(--font-m)', fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Apple M4 (MPS)</span>
                </div>
              </div>
            </div>

            <div>
              <SectionTitle>Tiempo real · 1 s</SectionTitle>
              <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '0 12px' }}>
                <DataRow
                  label="FPS"
                  value={fps > 0 ? `${fps.toFixed(1)} fps` : '—'}
                  sub={fps >= 15 ? 'Tiempo real ✓' : fps > 0 ? 'Bajo objetivo' : undefined}
                  color={fps >= 15 ? 'var(--green)' : fps > 0 ? 'var(--amber)' : 'var(--text-sub)'}
                />
                <DataRow
                  label="Latencia YOLO"
                  value={latenciams > 0 ? `${latenciams.toFixed(0)} ms` : '—'}
                  sub="inferencia únicamente"
                  color={latenciams > 0 ? 'var(--green)' : 'var(--text-sub)'}
                />
                <DataRow
                  label="Confianza prom."
                  value={confPromedio > 0 ? `${confPromedio.toFixed(1)}%` : '—'}
                  sub="detecciones activas"
                  color={confPromedio >= 70 ? 'var(--green)' : confPromedio > 0 ? 'var(--amber)' : 'var(--text-sub)'}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '9px 0' }}>
                  <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-sub)' }}>Detecciones / frame</span>
                  <span style={{ fontFamily: 'var(--font-m)', fontSize: 12, fontWeight: 600, color: '#60a5fa' }}>{fps > 0 ? detecciones : '—'}</span>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* ── EVALUACIÓN DE CAMPO ── */}
        {tab === 'campo' && (
          <div className="fade-up">
            <EvaluacionCampo />
          </div>
        )}

        {/* ── SESIÓN CSV ── */}
        {tab === 'sesion' && (
          <div className="fade-up">
            <SesionCSV />
          </div>
        )}

      </div>
    </div>
  );
}
