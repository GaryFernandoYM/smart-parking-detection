'use client';

import { useState, useEffect, useCallback } from 'react';

const ESPACIOS = [
  'espacio_Q','espacio_R','espacio_S','espacio_T','espacio_U',
  'espacio_V','espacio_W','espacio_X','espacio_Y','espacio_Z',
  'espacio_a','espacio_b','espacio_d','espacio_e','espacio_f',
];
const letra = (id: string) => id.replace('espacio_', '').toUpperCase();

interface AutoEstado {
  activa:            boolean;
  progreso:          number;
  elapsed_s:         number;
  duracion_s:        number;
  capturas_tomadas:  number;
  capturas_total:    number;
  proxima_en_s:      number;
  pendientes_anotar: number;
}

interface Resultado {
  accuracy: number; precision: number; recall: number; f1: number;
  tp: number; tn: number; fp: number; fn: number; errores: string[];
}

interface Captura {
  idx: number; timestamp: string;
  estados: Record<string,string>;
  snapshot: string | null;
  anotacion: Resultado | null;
}

interface Reporte {
  n: number; pendientes: number;
  accuracy_prom: number; precision_prom: number;
  recall_prom: number; f1_prom: number;
  fps_promedio: number; fps_min: number; fps_max: number;
  latencia_prom: number; latencia_min: number; latencia_max: number;
  detalle: (Resultado & { idx: number; timestamp: string })[];
}

function fmt(s: number) {
  const m = Math.floor(s / 60), ss = s % 60;
  return `${m}:${String(ss).padStart(2,'0')}`;
}

const S = {
  sectionTitle: {
    fontFamily: 'var(--font-m)', fontSize: 8, fontWeight: 700,
    letterSpacing: '0.18em', color: 'var(--teal)',
    textTransform: 'uppercase' as const, marginBottom: 8, display: 'block',
  } as React.CSSProperties,
  row: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
    padding: '8px 0', borderBottom: '1px solid var(--border)',
  } as React.CSSProperties,
  rowLabel: { fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)' } as React.CSSProperties,
  rowValue: { fontFamily: 'var(--font-m)', fontSize: 11, fontWeight: 600, color: 'var(--text)' } as React.CSSProperties,
  btn: {
    fontFamily: 'var(--font-m)', fontSize: 9, fontWeight: 700,
    letterSpacing: '0.1em', padding: '7px 14px', borderRadius: 4,
    border: 'none', cursor: 'pointer', transition: 'opacity 0.15s',
  } as React.CSSProperties,
  select: {
    background: 'var(--bg)', border: '1px solid var(--border-hi)',
    color: 'var(--text)', borderRadius: 4, padding: '6px 10px',
    fontFamily: 'var(--font-m)', fontSize: 11, outline: 'none', width: '100%',
  } as React.CSSProperties,
};

function metColor(v: number): string {
  return v >= 90 ? 'var(--green)' : v >= 70 ? 'var(--amber)' : 'var(--red)';
}

function MetPill({ v, label }: { v: number; label: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', background: 'var(--bg)', border: `1px solid ${metColor(v)}40`, borderTop: `2px solid ${metColor(v)}`, borderRadius: 5, padding: '8px 12px', minWidth: 68 }}>
      <span style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 18, color: metColor(v), lineHeight: 1 }}>{v}%</span>
      <span style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', marginTop: 4, letterSpacing: '0.1em' }}>{label}</span>
    </div>
  );
}

export default function EvaluacionCampo() {
  const [fase, setFase]           = useState<'config'|'corriendo'|'revision'|'reporte'>('config');
  const [minutos, setMinutos]     = useState(30);
  const [intervalo, setIntervalo] = useState(1);
  const [estado, setEstado]       = useState<AutoEstado | null>(null);
  const [captura, setCaptura]     = useState<Captura | null>(null);
  const [idxActual, setIdxActual] = useState(0);
  const [anotacion, setAnotacion] = useState<Record<string,boolean>>({});
  const [reporte, setReporte]     = useState<Reporte | null>(null);
  const [guardando, setGuardando] = useState(false);

  // ── Exportar CSV ───────────────────────────────────────────────────────────
  const exportarCSV = useCallback(() => {
    if (!reporte) return;

    const headers = ['captura','hora','exactitud','precision','recall','f1','tp','tn','fp','fn'];
    const rows = reporte.detalle.map(d => [
      d.idx + 1,
      d.timestamp,
      d.accuracy,
      d.precision,
      d.recall,
      d.f1,
      d.tp,
      d.tn,
      d.fp,
      d.fn,
    ]);

    // Fila de promedios
    rows.push([
      'PROMEDIO', '',
      reporte.accuracy_prom,
      reporte.precision_prom,
      reporte.recall_prom,
      reporte.f1_prom,
      '', '', '', '',
    ]);

    const csv = [
      `# Reporte de Evaluación de Campo · Smart Parking UPEU · YOLOv8s`,
      `# Generado: ${new Date().toLocaleString('es-PE')} · ${reporte.n} capturas`,
      `# FPS promedio: ${reporte.fps_promedio} · Latencia prom: ${reporte.latencia_prom} ms`,
      '',
      headers.join(','),
      ...rows.map(r => r.join(',')),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href     = url;
    link.download = `evaluacion_campo_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }, [reporte]);

  // ── Exportar imagen PNG (lógica original intacta) ──────────────────────────
  const generarImagen = useCallback(async () => {
    if (!reporte) return;

    let map50 = '—', map5095 = '—';
    try {
      const r = await fetch('http://localhost:8000/metricas-modelo');
      const d = await r.json();
      if (d.map50)    map50   = d.map50.toFixed(4);
      if (d.map50_95) map5095 = d.map50_95.toFixed(4);
    } catch { /* usa defaults */ }

    const W    = 900;
    const PAD  = 40;
    const ROW  = 32;
    const tableRows = reporte.detalle.length + 2;
    const H    = PAD + 70 + 20 + 130 + 20 + 120 + 20 + 40 + tableRows * ROW + 20 + 50 + PAD;

    const canvas  = document.createElement('canvas');
    canvas.width  = W * 2;
    canvas.height = H * 2;
    const ctx     = canvas.getContext('2d')!;
    ctx.scale(2, 2);

    const BG   = '#05080D';
    const CARD = '#0D1724';
    const LINE = '#162030';
    const TEXT = '#BDD4EC';
    const DIM  = '#3E5E7A';
    const GRN  = '#22c55e';
    const AMB  = '#F5A300';
    const RED  = '#ef4444';
    const IND  = '#00D4AA';

    const mColor = (v: number) => v >= 90 ? GRN : v >= 70 ? AMB : RED;

    const rr = (x: number, y: number, w: number, h: number, r: number, color: string) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
      ctx.fill();
    };

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    // Línea teal superior
    ctx.fillStyle = IND;
    ctx.fillRect(0, 0, W * 0.6, 2);

    let y = PAD;

    ctx.fillStyle = TEXT;
    ctx.font      = 'bold 20px "JetBrains Mono", "Courier New", monospace';
    ctx.fillText('REPORTE DE EVALUACIÓN DE CAMPO', PAD, y + 24);

    ctx.fillStyle = DIM;
    ctx.font      = '11px "JetBrains Mono", "Courier New", monospace';
    const fecha   = new Date().toLocaleDateString('es-PE', { year:'numeric', month:'long', day:'numeric', hour:'2-digit', minute:'2-digit' });
    ctx.fillText(`Smart Parking UPEU · YOLOv8s · ${fecha} · ${reporte.n} capturas`, PAD, y + 48);
    y += 70;

    ctx.fillStyle = LINE;
    ctx.fillRect(PAD, y, W - PAD * 2, 1);
    y += 20;

    const cols4 = [
      { label: 'Exactitud',  v: reporte.accuracy_prom  },
      { label: 'Precisión',  v: reporte.precision_prom },
      { label: 'Recall',     v: reporte.recall_prom    },
      { label: 'F1-Score',   v: reporte.f1_prom        },
    ];
    const boxW = (W - PAD * 2 - 12 * 3) / 4;
    cols4.forEach(({ label, v }, i) => {
      const bx = PAD + i * (boxW + 12);
      rr(bx, y, boxW, 110, 8, CARD);
      ctx.fillStyle = mColor(v);
      ctx.font      = 'bold 36px "JetBrains Mono", "Courier New", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${v}%`, bx + boxW / 2, y + 60);
      ctx.fillStyle = DIM;
      ctx.font      = '10px "JetBrains Mono", monospace';
      ctx.fillText(label.toUpperCase(), bx + boxW / 2, y + 88);
      ctx.textAlign = 'left';
    });
    y += 130;

    ctx.fillStyle = LINE;
    ctx.fillRect(PAD, y, W - PAD * 2, 1);
    y += 20;

    const half = (W - PAD * 2 - 16) / 2;

    rr(PAD, y, half, 110, 8, CARD);
    ctx.fillStyle = GRN;
    ctx.font      = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillText('RENDIMIENTO DE SESIÓN', PAD + 16, y + 22);
    [
      [`FPS  ${reporte.fps_promedio}`,          `mín ${reporte.fps_min}  ·  máx ${reporte.fps_max}`],
      [`Latencia  ${reporte.latencia_prom} ms`,  `mín ${reporte.latencia_min}  ·  máx ${reporte.latencia_max} ms`],
      [`Velocidad`, reporte.fps_promedio >= 15 ? 'Tiempo real ✓' : 'Por debajo del objetivo'],
    ].forEach(([a, b], i) => {
      const ry = y + 42 + i * 24;
      ctx.fillStyle = TEXT; ctx.font = '11px "JetBrains Mono", monospace'; ctx.fillText(a, PAD + 16, ry);
      ctx.fillStyle = DIM;  ctx.font = '10px system-ui'; ctx.textAlign = 'right'; ctx.fillText(b, PAD + half - 16, ry); ctx.textAlign = 'left';
    });

    const rx2 = PAD + half + 16;
    rr(rx2, y, half, 110, 8, CARD);
    ctx.fillStyle = IND;
    ctx.font      = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillText('MÉTRICAS DEL MODELO', rx2 + 16, y + 22);
    [['mAP50', map50], ['mAP50-95', map5095], ['Arquitectura', 'YOLOv8s fine-tuned']].forEach(([a, b], i) => {
      const ry = y + 42 + i * 24;
      ctx.fillStyle = DIM; ctx.font = '10px system-ui'; ctx.fillText(a, rx2 + 16, ry);
      ctx.fillStyle = i < 2 ? IND : TEXT; ctx.font = i < 2 ? 'bold 11px "JetBrains Mono", monospace' : '11px system-ui';
      ctx.textAlign = 'right'; ctx.fillText(b, rx2 + half - 16, ry); ctx.textAlign = 'left';
    });
    y += 130;

    ctx.fillStyle = LINE; ctx.fillRect(PAD, y, W - PAD * 2, 1); y += 20;
    ctx.fillStyle = DIM; ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillText('DETALLE POR CAPTURA', PAD, y + 14); y += 26;

    const cols = ['#','Hora','Exactitud','Precisión','Recall','F1','TP','TN','FP','FN'];
    const cw   = [30, 65, 80, 80, 70, 70, 40, 40, 40, 40];
    const cx0  = [0, 30, 95, 175, 255, 325, 395, 435, 475, 515].map(x => PAD + x);

    rr(PAD, y, W - PAD * 2, ROW, 4, CARD);
    cols.forEach((c, i) => {
      ctx.fillStyle = DIM; ctx.font = 'bold 9px system-ui';
      ctx.textAlign = i > 1 ? 'center' : 'left';
      ctx.fillText(c.toUpperCase(), cx0[i] + (i > 1 ? cw[i] / 2 : 8), y + ROW / 2 + 4);
    });
    ctx.textAlign = 'left'; y += ROW;

    reporte.detalle.forEach((d, ri) => {
      if (ri % 2 === 0) { rr(PAD, y, W - PAD * 2, ROW, 0, '#080C14'); }
      const vals = [String(d.idx + 1), d.timestamp, `${d.accuracy}%`, `${d.precision}%`, `${d.recall}%`, `${d.f1}%`, String(d.tp), String(d.tn), String(d.fp), String(d.fn)];
      vals.forEach((v, i) => {
        let color = TEXT;
        if (i >= 2 && i <= 5) color = mColor(parseFloat(v));
        if (i === 6 || i === 7) color = GRN;
        if (i === 8 || i === 9) color = parseFloat(v) > 0 ? RED : GRN;
        ctx.fillStyle = i <= 1 ? DIM : color;
        ctx.font      = i <= 1 ? '10px system-ui' : 'bold 10px "JetBrains Mono", monospace';
        ctx.textAlign = i > 1 ? 'center' : 'left';
        ctx.fillText(v, cx0[i] + (i > 1 ? cw[i] / 2 : 8), y + ROW / 2 + 4);
      });
      ctx.textAlign = 'left'; y += ROW;
    });

    rr(PAD, y, W - PAD * 2, ROW, 0, CARD);
    ctx.fillStyle = DIM; ctx.font = 'bold 9px system-ui'; ctx.fillText('PROMEDIO', PAD + 8, y + ROW / 2 + 4);
    [reporte.accuracy_prom, reporte.precision_prom, reporte.recall_prom, reporte.f1_prom].forEach((v, i) => {
      ctx.fillStyle = mColor(v); ctx.font = 'bold 10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center'; ctx.fillText(`${v}%`, cx0[i + 2] + cw[i + 2] / 2, y + ROW / 2 + 4);
    });
    ctx.textAlign = 'left'; y += ROW + 20;

    ctx.fillStyle = LINE; ctx.fillRect(PAD, y, W - PAD * 2, 1); y += 16;
    ctx.fillStyle = DIM; ctx.font = '10px system-ui';
    ctx.fillText('Universidad Peruana Unión · Smart Parking UPEU · Tesis 2026', PAD, y + 14);
    ctx.textAlign = 'right';
    ctx.fillText('Generado automáticamente por el sistema de evaluación', W - PAD, y + 14);
    ctx.textAlign = 'left';

    const link    = document.createElement('a');
    link.download = `reporte_evaluacion_${new Date().toISOString().slice(0,10)}.png`;
    link.href     = canvas.toDataURL('image/png');
    link.click();
  }, [reporte]);

  // ── Polling ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (fase !== 'corriendo' && fase !== 'revision') return;
    const id = setInterval(async () => {
      const r = await fetch('http://localhost:8000/auto/estado');
      const d: AutoEstado = await r.json();
      setEstado(d);
      if (fase === 'corriendo' && !d.activa && d.capturas_tomadas > 0) {
        setFase('revision');
        setIdxActual(0);
        cargarCaptura(0);
      }
    }, 1000);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fase]);

  const cargarCaptura = useCallback(async (idx: number) => {
    setCaptura(null);
    const r = await fetch(`http://localhost:8000/auto/captura/${idx}`);
    const c: Captura = await r.json();
    setCaptura(c);
    const anot: Record<string,boolean> = {};
    for (const eid of ESPACIOS) anot[eid] = (c.estados[eid] ?? 'libre') !== 'libre';
    setAnotacion(anot);
  }, []);

  const iniciar = async () => {
    await fetch(`http://localhost:8000/auto/iniciar?minutos=${minutos}&intervalo=${intervalo * 60}`, { method: 'POST' });
    const r = await fetch('http://localhost:8000/auto/estado');
    setEstado(await r.json());
    setFase('corriendo');
  };

  const guardarAnotacion = async () => {
    if (!captura) return;
    setGuardando(true);
    const gt: Record<string,string> = {};
    for (const eid of ESPACIOS) gt[eid] = anotacion[eid] ? 'ocupado' : 'libre';
    await fetch(`http://localhost:8000/auto/anotar/${captura.idx}`, {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(gt),
    });
    setGuardando(false);

    const siguiente = idxActual + 1;
    if (estado && siguiente < estado.capturas_tomadas) {
      setIdxActual(siguiente);
      cargarCaptura(siguiente);
    } else {
      const r = await fetch('http://localhost:8000/auto/reporte');
      setReporte(await r.json());
      setFase('reporte');
    }
  };

  const reiniciar = async () => {
    await fetch('http://localhost:8000/auto', { method: 'DELETE' });
    setEstado(null); setCaptura(null); setReporte(null); setIdxActual(0);
    setFase('config');
  };

  // ── RENDER ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* ── CONFIGURACIÓN ── */}
      {fase === 'config' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }} className="fade-up">
          <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', lineHeight: 1.8, letterSpacing: '0.06em' }}>
            El sistema captura automáticamente cada intervalo. Al terminar, revisas cada captura y anotas el estado real del estacionamiento.
          </p>

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 100 }}>
              <label style={S.sectionTitle}>Duración</label>
              <select style={S.select} value={minutos} onChange={e => setMinutos(Number(e.target.value))}>
                <option value={15}>15 min</option>
                <option value={30}>30 min</option>
                <option value={60}>60 min</option>
              </select>
            </div>
            <div style={{ flex: 1, minWidth: 100 }}>
              <label style={S.sectionTitle}>Captura cada</label>
              <select style={S.select} value={intervalo} onChange={e => setIntervalo(Number(e.target.value))}>
                <option value={1}>1 minuto</option>
                <option value={2}>2 minutos</option>
                <option value={5}>5 minutos</option>
              </select>
            </div>
            <div style={{ flexShrink: 0 }}>
              <label style={S.sectionTitle}>Capturas</label>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 34, padding: '0 14px', background: 'var(--bg)', border: '1px solid var(--border-hi)', borderRadius: 4, fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 16, color: 'var(--teal)' }}>
                {minutos / intervalo}
              </div>
            </div>
          </div>

          <button onClick={iniciar} style={{ ...S.btn, background: 'var(--teal)', color: 'var(--bg)', padding: '10px', width: '100%', fontSize: 10 }}>
            ▶ INICIAR SESIÓN AUTOMÁTICA
          </button>
        </div>
      )}

      {/* ── EN CURSO ── */}
      {fase === 'corriendo' && estado && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }} className="fade-up">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--teal)', display: 'inline-block', boxShadow: '0 0 8px var(--teal)' }} className="do-glow" />
            <span style={{ fontFamily: 'var(--font-m)', fontSize: 11, color: 'var(--teal)', fontWeight: 600, letterSpacing: '0.1em' }}>SESIÓN ACTIVA</span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
            <div>
              <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', letterSpacing: '0.1em' }}>
                {fmt(estado.elapsed_s)} / {fmt(estado.duracion_s)} &nbsp;·&nbsp; próxima en {fmt(estado.proxima_en_s)}
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 22, color: 'var(--teal)', lineHeight: 1 }}>{estado.capturas_tomadas}</span>
              <span style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', marginLeft: 6 }}>/ {estado.capturas_total}</span>
            </div>
          </div>

          <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--teal)', width: `${estado.progreso}%`, transition: 'width 1s ease', boxShadow: '0 0 8px var(--teal)' }} />
          </div>

          <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', textAlign: 'center', letterSpacing: '0.1em' }}>
            Puedes seguir usando el sistema. Al terminar podrás anotar cada captura.
          </p>

          <button onClick={reiniciar} style={{ ...S.btn, background: 'transparent', color: 'var(--red)', border: '1px solid rgba(239,68,68,0.3)', width: '100%', fontSize: 9 }}>
            ■ CANCELAR SESIÓN
          </button>
        </div>
      )}

      {/* ── REVISIÓN ── */}
      {fase === 'revision' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }} className="fade-up">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <p style={{ fontFamily: 'var(--font-m)', fontSize: 11, color: 'var(--text)', fontWeight: 600 }}>
                Revisión &nbsp;
                <span style={{ color: 'var(--amber)' }}>{idxActual + 1} / {estado?.capturas_tomadas ?? '?'}</span>
              </p>
              {captura && <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', marginTop: 2 }}>Captura a las {captura.timestamp}</p>}
            </div>
            <div style={{ display: 'flex', gap: 3 }}>
              {Array.from({ length: estado?.capturas_tomadas ?? 0 }, (_, i) => (
                <div key={i} style={{ width: 7, height: 7, borderRadius: '50%', background: i < idxActual ? 'var(--green)' : i === idxActual ? 'var(--teal)' : 'var(--border-hi)' }} />
              ))}
            </div>
          </div>

          {!captura && (
            <p style={{ fontFamily: 'var(--font-m)', fontSize: 10, color: 'var(--text-sub)', textAlign: 'center', padding: '16px 0' }} className="do-blink">
              CARGANDO CAPTURA…
            </p>
          )}

          {captura && (
            <>
              {captura.snapshot && (
                <div style={{ borderRadius: 5, overflow: 'hidden', border: '1px solid var(--border-hi)' }}>
                  <img src={`data:image/jpeg;base64,${captura.snapshot}`} style={{ width: '100%', display: 'block' }} alt="" />
                </div>
              )}

              <p style={{ fontFamily: 'var(--font-m)', fontSize: 9, color: 'var(--text-sub)', lineHeight: 1.7 }}>
                Mira la imagen y <span style={{ color: 'var(--amber)', fontWeight: 700 }}>corrige</span> cuáles espacios estaban realmente ocupados:
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 5 }}>
                {ESPACIOS.map(eid => {
                  const predOc = (captura.estados[eid] ?? 'libre') !== 'libre';
                  const anotOc = anotacion[eid] ?? false;
                  const diff   = predOc !== anotOc;
                  return (
                    <button key={eid}
                      onClick={() => setAnotacion(p => ({ ...p, [eid]: !p[eid] }))}
                      style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                        padding: '8px 4px', borderRadius: 5, cursor: 'pointer', transition: 'all 0.12s',
                        background: anotOc ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.1)',
                        border: `2px solid ${anotOc ? 'var(--red)' : 'var(--green)'}`,
                        outline: diff ? '2px solid var(--amber)' : 'none',
                        outlineOffset: 1,
                      }}
                    >
                      <span style={{ fontFamily: 'var(--font-m)', fontWeight: 700, fontSize: 15, color: anotOc ? 'var(--red)' : 'var(--green)', lineHeight: 1 }}>{letra(eid)}</span>
                      <span style={{ fontFamily: 'var(--font-m)', fontSize: 7, color: anotOc ? 'var(--red)' : 'var(--green)', marginTop: 3, letterSpacing: '0.08em' }}>{anotOc ? 'OC' : 'LI'}</span>
                      {diff && <span style={{ fontFamily: 'var(--font-m)', fontSize: 7, color: 'var(--amber)', marginTop: 1 }}>≠</span>}
                    </button>
                  );
                })}
              </div>
              <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', letterSpacing: '0.08em' }}>
                Borde ámbar = sistema predijo diferente · Clic para cambiar
              </p>

              <button onClick={guardarAnotacion} disabled={guardando} style={{ ...S.btn, background: 'var(--teal)', color: 'var(--bg)', width: '100%', padding: '10px', fontSize: 10, opacity: guardando ? 0.6 : 1 }}>
                {guardando ? '▸ GUARDANDO…' : idxActual + 1 < (estado?.capturas_tomadas ?? 0) ? '▶ GUARDAR Y SIGUIENTE' : '✓ FINALIZAR Y VER REPORTE'}
              </button>
            </>
          )}
        </div>
      )}

      {/* ── REPORTE FINAL ── */}
      {fase === 'reporte' && reporte && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }} className="fade-up">

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
            <div>
              <p style={{ fontFamily: 'var(--font-m)', fontSize: 10, fontWeight: 700, color: 'var(--text)', letterSpacing: '0.06em' }}>
                REPORTE FINAL
              </p>
              <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', marginTop: 2 }}>
                {reporte.n} capturas evaluadas
                {reporte.pendientes > 0 && <span style={{ color: 'var(--amber)', marginLeft: 8 }}>· {reporte.pendientes} sin anotar</span>}
              </p>
            </div>

            {/* Botones de exportación */}
            <div style={{ display: 'flex', gap: 6 }}>
              <button onClick={exportarCSV} style={{ ...S.btn, background: 'var(--teal)', color: 'var(--bg)' }}>
                ↓ CSV
              </button>
              <button onClick={generarImagen} style={{ ...S.btn, background: 'var(--border-hi)', color: 'var(--text)' }}>
                ↓ PNG
              </button>
              <button onClick={reiniciar} style={{ ...S.btn, background: 'transparent', color: 'var(--text-sub)', border: '1px solid var(--border-hi)', fontSize: 8 }}>
                ↺ NUEVA
              </button>
            </div>
          </div>

          {/* Métricas principales */}
          <div>
            <span style={S.sectionTitle}>Exactitud de detección</span>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <MetPill v={reporte.accuracy_prom}  label="Exactitud" />
              <MetPill v={reporte.precision_prom} label="Precisión" />
              <MetPill v={reporte.recall_prom}    label="Recall"    />
              <MetPill v={reporte.f1_prom}        label="F1-Score"  />
            </div>
          </div>

          {/* Rendimiento */}
          {reporte.fps_promedio > 0 && (
            <div>
              <span style={S.sectionTitle}>Rendimiento de sesión</span>
              <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '0 12px' }}>
                {[
                  { label: 'FPS promedio',      value: `${reporte.fps_promedio} fps`, sub: `mín ${reporte.fps_min} · máx ${reporte.fps_max}` },
                  { label: 'Latencia promedio', value: `${reporte.latencia_prom} ms`, sub: `mín ${reporte.latencia_min} · máx ${reporte.latencia_max} ms` },
                  { label: 'Velocidad',         value: reporte.fps_promedio >= 15 ? 'Tiempo real ✓' : 'Por debajo', sub: 'objetivo ≥ 15 fps' },
                ].map(({ label, value, sub }) => (
                  <div key={label} style={{ ...S.row, ':last-child': { borderBottom: 'none' } as React.CSSProperties }}>
                    <span style={S.rowLabel}>{label}</span>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ ...S.rowValue, color: label.includes('Vel') && reporte.fps_promedio >= 15 ? 'var(--green)' : 'var(--text)' }}>{value}</span>
                      <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', marginTop: 2 }}>{sub}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tabla detalle */}
          <div>
            <span style={S.sectionTitle}>Detalle por captura</span>
            <div style={{ overflowX: 'auto', maxHeight: 200, borderRadius: 5, border: '1px solid var(--border)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-m)', fontSize: 9 }}>
                <thead>
                  <tr style={{ background: 'var(--surface2)', position: 'sticky', top: 0 }}>
                    {['#','Hora','Exact.','Prec.','Recall','F1','TP','TN','FP','FN'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: 'var(--text-sub)', fontWeight: 700, letterSpacing: '0.1em', textAlign: h === '#' || h === 'Hora' ? 'left' : 'center', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {reporte.detalle.map((d, i) => (
                    <tr key={i} style={{ borderTop: '1px solid var(--border)', background: i % 2 === 0 ? 'var(--bg)' : 'transparent' }}>
                      <td style={{ padding: '5px 8px', color: 'var(--text-sub)' }}>{d.idx + 1}</td>
                      <td style={{ padding: '5px 8px', color: 'var(--text-sub)' }}>{d.timestamp}</td>
                      {[d.accuracy, d.precision, d.recall, d.f1].map((v, j) => (
                        <td key={j} style={{ padding: '5px 8px', textAlign: 'center', fontWeight: 700, color: metColor(v) }}>{v}%</td>
                      ))}
                      <td style={{ padding: '5px 8px', textAlign: 'center', color: 'var(--green)', fontWeight: 600 }}>{d.tp}</td>
                      <td style={{ padding: '5px 8px', textAlign: 'center', color: 'var(--green)', fontWeight: 600 }}>{d.tn}</td>
                      <td style={{ padding: '5px 8px', textAlign: 'center', color: d.fp > 0 ? 'var(--red)' : 'var(--text-sub)', fontWeight: 600 }}>{d.fp}</td>
                      <td style={{ padding: '5px 8px', textAlign: 'center', color: d.fn > 0 ? 'var(--red)' : 'var(--text-sub)', fontWeight: 600 }}>{d.fn}</td>
                    </tr>
                  ))}
                  <tr style={{ borderTop: '2px solid var(--border-hi)', background: 'var(--surface2)' }}>
                    <td colSpan={2} style={{ padding: '5px 8px', color: 'var(--text-sub)', fontWeight: 700, letterSpacing: '0.1em', fontSize: 8 }}>PROMEDIO</td>
                    {[reporte.accuracy_prom, reporte.precision_prom, reporte.recall_prom, reporte.f1_prom].map((v,j) => (
                      <td key={j} style={{ padding: '5px 8px', textAlign: 'center', fontWeight: 700, color: metColor(v) }}>{v}%</td>
                    ))}
                    <td colSpan={4} />
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <p style={{ fontFamily: 'var(--font-m)', fontSize: 8, color: 'var(--text-sub)', letterSpacing: '0.08em' }}>
            CSV incluye encabezados y fila de promedios. PNG incluye métricas del modelo y tabla completa.
          </p>
        </div>
      )}

    </div>
  );
}
