'use client';

import { useState, useEffect } from 'react';

const API = 'http://localhost:8000';

const S: Record<string, React.CSSProperties> = {
  label: { fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.14em', color: 'var(--text-sub)', textTransform: 'uppercase' as const, marginBottom: 5, display: 'block' },
  input: { width: '100%', fontFamily: 'var(--font-m)', fontSize: 12, background: 'var(--bg)', border: '1px solid var(--border-hi)', borderRadius: 4, padding: '7px 10px', color: 'var(--text)', outline: 'none' },
  btn: { fontFamily: 'var(--font-m)', fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', padding: '8px 16px', borderRadius: 4, border: 'none', cursor: 'pointer', transition: 'opacity 0.15s' },
  row: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' },
  rowLabel: { fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--text-sub)' },
  rowValue: { fontFamily: 'var(--font-m)', fontSize: 11, fontWeight: 600, color: 'var(--text)' },
};

interface CsvEstado {
  activa: boolean;
  nombre: string;
  condicion: string;
  minutos_registrados: number;
  filas_acumuladas: number;
  progreso: number;
}

interface Metricas {
  slot_accuracy: number;
  fpr: number;
  fnr: number;
  total_filas: number;
  n_libres: number;
  n_ocupados: number;
  error?: string;
}

function MetricGauge({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ position: 'relative', width: 72, height: 72, margin: '0 auto 8px' }}>
        <svg width="72" height="72" viewBox="0 0 72 72">
          <circle cx="36" cy="36" r="28" fill="none" stroke="var(--border-hi)" strokeWidth="6" />
          <circle
            cx="36" cy="36" r="28"
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeDasharray={`${2 * Math.PI * 28}`}
            strokeDashoffset={`${2 * Math.PI * 28 * (1 - value)}`}
            strokeLinecap="round"
            transform="rotate(-90 36 36)"
            style={{ transition: 'stroke-dashoffset 0.6s ease', filter: `drop-shadow(0 0 4px ${color})` }}
          />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 14, color }}>{pct}%</span>
        </div>
      </div>
      <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.12em', color: 'var(--text-sub)', textTransform: 'uppercase' as const }}>{label}</span>
    </div>
  );
}

export default function SesionCSV() {
  const [fase, setFase] = useState<'idle' | 'corriendo' | 'terminado'>('idle');
  const [nombre, setNombre] = useState('Sesion1');
  const [condicion, setCondicion] = useState<'diurna' | 'nocturna'>('diurna');
  const [minutos, setMinutos] = useState(30);
  const [estado, setEstado] = useState<CsvEstado | null>(null);
  const [metricas, setMetricas] = useState<Metricas | null>(null);
  const [archivo, setArchivo] = useState('');
  const [error, setError] = useState('');
  const [cargando, setCargando] = useState(false);

  useEffect(() => {
    if (fase !== 'corriendo') return;
    const poll = async () => {
      try {
        const r = await fetch(`${API}/sesion/estado`);
        const d: CsvEstado = await r.json();
        setEstado(d);
        if (!d.activa && d.minutos_registrados > 0) setFase('terminado');
      } catch { /* backend no disponible */ }
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, [fase]);

  const iniciar = async () => {
    setError(''); setCargando(true);
    try {
      const r = await fetch(
        `${API}/sesion/iniciar?nombre=${encodeURIComponent(nombre)}&condicion=${condicion}&minutos=${minutos}`,
        { method: 'POST' }
      );
      if (!r.ok) throw new Error();
      setFase('corriendo'); setMetricas(null); setArchivo('');
    } catch { setError('No se pudo iniciar la sesión'); }
    setCargando(false);
  };

  const detener = async () => {
    setCargando(true);
    try {
      const r = await fetch(`${API}/sesion/detener`, { method: 'POST' });
      const d = await r.json();
      if (d.archivo) setArchivo(d.archivo);
      setFase('terminado');
    } catch { setError('Error al detener'); }
    setCargando(false);
  };

  const calcularMetricas = async () => {
    setError(''); setCargando(true);
    try {
      const r = await fetch(`${API}/sesion/metricas`);
      const d: Metricas = await r.json();
      if (d.error) setError(d.error);
      else setMetricas(d);
    } catch { setError('Error calculando métricas'); }
    setCargando(false);
  };

  const reiniciar = () => {
    setFase('idle'); setEstado(null);
    setMetricas(null); setArchivo(''); setError('');
  };

  /* ── IDLE ── */
  if (fase === 'idle') return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }} className="fade-up">

      <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.14em', color: 'var(--text-sub)', lineHeight: 1.6 }}>
        Captura automática del estado de cada plaza cada 60 s. Al finalizar, rellena la columna <span style={{ color: 'var(--teal)' }}>ground_truth</span> en el CSV y calcula las métricas.
      </p>

      <div>
        <label style={S.label}>Nombre de sesión</label>
        <input style={S.input} value={nombre} onChange={e => setNombre(e.target.value)} placeholder="Sesion1" />
      </div>

      <div>
        <label style={S.label}>Condición</label>
        <div style={{ display: 'flex', gap: 6 }}>
          {(['diurna', 'nocturna'] as const).map(c => (
            <button key={c}
              onClick={() => setCondicion(c)}
              style={{
                ...S.btn, flex: 1,
                background: condicion === c ? 'var(--teal)' : 'var(--border)',
                color: condicion === c ? 'var(--bg)' : 'var(--text-sub)',
              }}
            >
              {c === 'diurna' ? '☀ DIURNA' : '☾ NOCTURNA'}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label style={S.label}>Duración (minutos)</label>
        <div style={{ display: 'flex', gap: 6 }}>
          {[15, 30, 60].map(m => (
            <button key={m}
              onClick={() => setMinutos(m)}
              style={{
                ...S.btn, flex: 1,
                background: minutos === m ? 'var(--amber)' : 'var(--border)',
                color: minutos === m ? 'var(--bg)' : 'var(--text-sub)',
              }}
            >
              {m} MIN
            </button>
          ))}
        </div>
      </div>

      {error && (
        <p style={{ fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--red)', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 4, padding: '6px 10px' }}>
          ⚠ {error}
        </p>
      )}

      <button
        onClick={iniciar}
        disabled={cargando}
        style={{ ...S.btn, background: 'var(--teal)', color: 'var(--bg)', width: '100%', fontSize: 11, padding: '10px', opacity: cargando ? 0.6 : 1 }}
      >
        {cargando ? '▸ INICIANDO…' : '▶ INICIAR SESIÓN'}
      </button>
    </div>
  );

  /* ── CORRIENDO ── */
  if (fase === 'corriendo') return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }} className="fade-up">

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--teal)', boxShadow: '0 0 8px var(--teal)', display: 'inline-block', flexShrink: 0 }} className="do-glow" />
        <span style={{ fontFamily: 'var(--font-m)', fontSize: 11, color: 'var(--teal)', fontWeight: 600, letterSpacing: '0.1em' }}>
          SESIÓN ACTIVA
        </span>
      </div>

      {estado && (
        <>
          {/* Progreso */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', letterSpacing: '0.1em' }}>PROGRESO</span>
              <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--teal)', fontWeight: 600 }}>{estado.progreso}%</span>
            </div>
            <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', background: 'var(--teal)', width: `${estado.progreso}%`, transition: 'width 0.5s ease', boxShadow: '0 0 8px var(--teal)' }} />
            </div>
          </div>

          {/* Stats */}
          <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 12px' }}>
            <div style={S.row}>
              <span style={S.rowLabel}>SESIÓN</span>
              <span style={S.rowValue}>{estado.nombre} · {estado.condicion}</span>
            </div>
            <div style={S.row}>
              <span style={S.rowLabel}>MINUTOS REGISTRADOS</span>
              <span style={{ ...S.rowValue, color: 'var(--teal)' }}>{estado.minutos_registrados}</span>
            </div>
            <div style={{ ...S.row, borderBottom: 'none' }}>
              <span style={S.rowLabel}>FILAS ACUMULADAS</span>
              <span style={{ ...S.rowValue, color: 'var(--amber)' }}>{estado.filas_acumuladas}</span>
            </div>
          </div>

          <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', letterSpacing: '0.1em', lineHeight: 1.6 }}>
            Captura automática cada 60 s. Puedes detener antes de que finalice el tiempo.
          </p>
        </>
      )}

      {error && (
        <p style={{ fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--red)', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 4, padding: '6px 10px' }}>
          ⚠ {error}
        </p>
      )}

      <button
        onClick={detener}
        disabled={cargando}
        style={{ ...S.btn, background: 'rgba(239,68,68,0.15)', color: 'var(--red)', border: '1px solid rgba(239,68,68,0.3)', width: '100%', fontSize: 11, padding: '10px', opacity: cargando ? 0.6 : 1 }}
      >
        {cargando ? '▸ DETENIENDO…' : '■ DETENER Y GUARDAR CSV'}
      </button>
    </div>
  );

  /* ── TERMINADO ── */
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }} className="fade-up">

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: 'var(--green)', fontSize: 14 }}>✓</span>
        <span style={{ fontFamily: 'var(--font-m)', fontSize: 11, color: 'var(--green)', fontWeight: 600, letterSpacing: '0.1em' }}>
          SESIÓN COMPLETADA
        </span>
      </div>

      {estado && (
        <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 12px' }}>
          <div style={S.row}>
            <span style={S.rowLabel}>MINUTOS CAPTURADOS</span>
            <span style={{ ...S.rowValue, color: 'var(--teal)' }}>{estado.minutos_registrados}</span>
          </div>
          <div style={{ ...S.row, borderBottom: 'none' }}>
            <span style={S.rowLabel}>FILAS EN CSV</span>
            <span style={{ ...S.rowValue, color: 'var(--amber)' }}>{estado.filas_acumuladas}</span>
          </div>
        </div>
      )}

      {archivo && (
        <div style={{ background: 'var(--teal-d)', border: '1px solid rgba(0,212,170,0.25)', borderRadius: 5, padding: '8px 12px' }}>
          <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, letterSpacing: '0.12em', color: 'var(--teal)', marginBottom: 4 }}>ARCHIVO GUARDADO EN:</p>
          <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text)', wordBreak: 'break-all' as const }}>{archivo}</p>
        </div>
      )}

      {!metricas && (
        <div style={{ background: 'var(--amber-d)', border: '1px solid rgba(245,163,0,0.2)', borderRadius: 5, padding: '10px 12px' }}>
          <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--amber)', lineHeight: 1.7 }}>
            1. Abre el CSV guardado<br/>
            2. Rellena la columna <strong>ground_truth</strong> con:<br/>
            &nbsp;&nbsp;&nbsp;<span style={{ color: 'var(--green)' }}>libre</span> · <span style={{ color: 'var(--red)' }}>ocupado</span> · <span style={{ color: 'var(--amber)' }}>no_disponible</span><br/>
            3. Guarda el archivo y pulsa el botón
          </p>
        </div>
      )}

      {!metricas && (
        <button
          onClick={calcularMetricas}
          disabled={cargando}
          style={{ ...S.btn, background: 'var(--teal)', color: 'var(--bg)', width: '100%', fontSize: 11, padding: '10px', opacity: cargando ? 0.6 : 1 }}
        >
          {cargando ? (
            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <span className="do-spin" style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', border: '2px solid var(--bg)', borderTopColor: 'transparent' }} />
              CALCULANDO…
            </span>
          ) : '◆ CALCULAR MÉTRICAS'}
        </button>
      )}

      {error && (
        <p style={{ fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--red)', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 4, padding: '6px 10px' }}>
          ⚠ {error}
        </p>
      )}

      {metricas && !metricas.error && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }} className="fade-up">
          <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, letterSpacing: '0.14em', color: 'var(--text-sub)', textAlign: 'center' }}>MÉTRICAS DE EVALUACIÓN</p>

          <div style={{ display: 'flex', justifyContent: 'space-around' }}>
            <MetricGauge label="Slot Accuracy" value={metricas.slot_accuracy} color="var(--teal)" />
            <MetricGauge label="FPR" value={metricas.fpr} color="var(--amber)" />
            <MetricGauge label="FNR" value={metricas.fnr} color="var(--red)" />
          </div>

          <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 12px' }}>
            <div style={S.row}>
              <span style={S.rowLabel}>TOTAL FILAS</span>
              <span style={S.rowValue}>{metricas.total_filas}</span>
            </div>
            <div style={S.row}>
              <span style={S.rowLabel}>FILAS LIBRE (gt)</span>
              <span style={{ ...S.rowValue, color: 'var(--green)' }}>{metricas.n_libres}</span>
            </div>
            <div style={{ ...S.row, borderBottom: 'none' }}>
              <span style={S.rowLabel}>FILAS OCUPADO (gt)</span>
              <span style={{ ...S.rowValue, color: 'var(--red)' }}>{metricas.n_ocupados}</span>
            </div>
          </div>
        </div>
      )}

      <button
        onClick={reiniciar}
        style={{ ...S.btn, background: 'transparent', color: 'var(--text-sub)', border: '1px solid var(--border-hi)', width: '100%', fontSize: 10, padding: '8px' }}
      >
        ↺ NUEVA SESIÓN
      </button>
    </div>
  );
}
