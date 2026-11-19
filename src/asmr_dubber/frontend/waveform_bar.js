import { $, state, api, t, time, guard, projectBusy } from './session.js';

const waveCache = new Map();
const MAX_CACHE = 48;
let activeSentenceId = '';
let currentWindow = { start: 0, end: 1, peaks: [] };
let dragging = null;

function windowBounds(sentence, mediaDuration) {
  const span = Math.max(0.25, sentence.end_seconds - sentence.start_seconds);
  const pad = Math.max(0.8, Math.min(2.5, span * 0.45));
  const start = Math.max(0, Number((sentence.start_seconds - pad).toFixed(2)));
  const end = Math.min(
    mediaDuration,
    Math.max(start + 0.3, Number((sentence.end_seconds + pad).toFixed(2))),
  );
  return { start, end };
}

async function fetchPeaks(manifest, start, end) {
  const key = `${manifest}:${start.toFixed(2)}:${end.toFixed(2)}`;
  if (waveCache.has(key)) return waveCache.get(key);
  const data = await api('projects/waveform', { project: manifest, start, end, bins: 180 });
  if (waveCache.size >= MAX_CACHE) waveCache.delete(waveCache.keys().next().value);
  waveCache.set(key, data);
  return data;
}

export function drawWaveform(sentence) {
  const canvas = $('#sentenceWaveform');
  if (!canvas || !sentence) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(240, Math.floor(rect.width || 520));
  const height = 56;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const { start, end, peaks } = currentWindow;
  const span = Math.max(0.05, end - start);
  const xStart = Math.max(0, Math.min(width, ((sentence.start_seconds - start) / span) * width));
  const xEnd = Math.max(xStart + 4, Math.min(width, ((sentence.end_seconds - start) / span) * width));

  ctx.fillStyle = 'rgba(0, 113, 227, 0.11)';
  ctx.fillRect(xStart, 0, xEnd - xStart, height);

  const n = peaks.length || 1;
  const barW = Math.max(1.5, width / n - 1);
  for (let i = 0; i < peaks.length; i += 1) {
    const x = (i / n) * width;
    const amp = Math.max(0.06, Number(peaks[i] || 0));
    const barH = amp * (height - 12);
    const inside = x + barW * 0.5 >= xStart && x <= xEnd;
    ctx.fillStyle = inside ? '#0071e3' : 'rgba(120, 120, 128, 0.36)';
    ctx.fillRect(x, (height - barH) * 0.5, barW, barH);
  }

  for (const hx of [xStart, xEnd]) {
    ctx.fillStyle = '#0071e3';
    ctx.fillRect(Math.round(hx) - 1, 0, 2, height);
    ctx.beginPath();
    ctx.roundRect(Math.round(hx) - 4, height * 0.5 - 10, 8, 20, 3);
    ctx.fill();
  }

  const label = $('#waveformLabel');
  if (label) {
    const dur = Math.max(0, sentence.end_seconds - sentence.start_seconds).toFixed(2);
    label.textContent = `${time(sentence.start_seconds)} – ${time(sentence.end_seconds)} (${dur}s)`;
  }
}

export async function refreshWaveform(sentence) {
  if (!state.project || !sentence) return;
  activeSentenceId = sentence.id;
  const mediaDur = state.project.source.duration_seconds;
  if (
    !dragging &&
    (!currentWindow.peaks.length ||
      sentence.start_seconds < currentWindow.start + 0.05 ||
      sentence.end_seconds > currentWindow.end - 0.05 ||
      currentWindow.end - currentWindow.start > Math.max(8, (sentence.end_seconds - sentence.start_seconds) * 4))
  ) {
    const bounds = windowBounds(sentence, mediaDur);
    currentWindow = { ...bounds, peaks: [] };
    drawWaveform(sentence);
    try {
      const result = await fetchPeaks(state.project.manifest, bounds.start, bounds.end);
      if (activeSentenceId !== sentence.id) return;
      currentWindow = { start: result.start, end: result.end, peaks: result.peaks || [] };
    } catch {
      return;
    }
  }
  drawWaveform(sentence);
}

export function clearWaveformCache() {
  waveCache.clear();
  currentWindow = { start: 0, end: 1, peaks: [] };
}

export function initializeWaveform({ getSelectedSentence, onBoundaryCommit, onAuditionSlice }) {
  const timeAtPointer = (canvas, clientX) => {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
    return currentWindow.start + ratio * (currentWindow.end - currentWindow.start);
  };

  document.addEventListener('pointerdown', (event) => {
    const canvas = event.target.closest('#sentenceWaveform');
    if (!canvas) return;
    const sentence = getSelectedSentence();
    if (!sentence) return;
    const rect = canvas.getBoundingClientRect();
    const span = Math.max(0.05, currentWindow.end - currentWindow.start);
    const xStart = ((sentence.start_seconds - currentWindow.start) / span) * rect.width;
    const xEnd = ((sentence.end_seconds - currentWindow.start) / span) * rect.width;
    const localX = event.clientX - rect.left;
    const dStart = Math.abs(localX - xStart);
    const dEnd = Math.abs(localX - xEnd);
    const side = Math.min(dStart, dEnd) <= 12 ? (dStart <= dEnd ? 'start' : 'end') : null;
    if (!side || projectBusy()) {
      onAuditionSlice(sentence);
      return;
    }
    dragging = {
      side,
      id: sentence.id,
      beforeStart: sentence.start_seconds,
      beforeEnd: sentence.end_seconds,
    };
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  document.addEventListener('pointermove', (event) => {
    const canvas = $('#sentenceWaveform');
    if (!canvas) return;
    const sentence = getSelectedSentence();
    if (!dragging) {
      if (event.target === canvas && sentence) {
        const rect = canvas.getBoundingClientRect();
        const span = Math.max(0.05, currentWindow.end - currentWindow.start);
        const xStart = ((sentence.start_seconds - currentWindow.start) / span) * rect.width;
        const xEnd = ((sentence.end_seconds - currentWindow.start) / span) * rect.width;
        const lx = event.clientX - rect.left;
        canvas.style.cursor = Math.min(Math.abs(lx - xStart), Math.abs(lx - xEnd)) <= 12 ? 'ew-resize' : 'pointer';
      }
      return;
    }
    if (!sentence || sentence.id !== dragging.id) return;
    const mediaDur = state.project.source.duration_seconds;
    const tVal = Number(timeAtPointer(canvas, event.clientX).toFixed(2));
    if (dragging.side === 'start') {
      sentence.start_seconds = Math.max(0, Math.min(sentence.end_seconds - 0.08, tVal));
      const inp = $('[data-detail="start_seconds"]');
      if (inp) inp.value = sentence.start_seconds;
    } else {
      sentence.end_seconds = Math.min(mediaDur, Math.max(sentence.start_seconds + 0.08, tVal));
      const inp = $('[data-detail="end_seconds"]');
      if (inp) inp.value = sentence.end_seconds;
    }
    const rowTime = $(`#rows tr[data-sentence="${CSS.escape(sentence.id)}"] .time-span`);
    if (rowTime) rowTime.textContent = `${time(sentence.start_seconds)} – ${time(sentence.end_seconds)}`;
    drawWaveform(sentence);
  });

  document.addEventListener(
    'pointerup',
    guard(async () => {
      if (!dragging) return;
      const finished = dragging;
      dragging = null;
      const sentence = getSelectedSentence();
      if (!sentence || sentence.id !== finished.id) return;
      if (
        sentence.start_seconds !== finished.beforeStart ||
        sentence.end_seconds !== finished.beforeEnd
      ) {
        await onBoundaryCommit(sentence, finished.beforeStart, finished.beforeEnd);
      }
      await refreshWaveform(sentence);
    }),
  );
  document.addEventListener('pointercancel', () => {
    dragging = null;
  });
}
