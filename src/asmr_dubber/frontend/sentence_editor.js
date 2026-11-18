import { $, $$, state, esc, t, time, field, guard, projectBusy } from './session.js';
import { dubCell, originalCell, flagsCell } from './project_review.js';
import { refreshWaveform, initializeWaveform, clearWaveformCache } from './waveform_bar.js';

const ROW_HEIGHT = 48;
const OVERSCAN = 12;
const MAX_HISTORY = 80;

let undoStack = [];
let redoStack = [];
let sliceEndAt = null;
let hooks = null;

export function resetEditorHistory() {
  undoStack = [];
  redoStack = [];
  sliceEndAt = null;
  clearWaveformCache();
}

export function recordFieldUndo(id, changesBefore, changesAfter) {
  undoStack.push({ kind: 'fields', id, before: changesBefore, after: changesAfter });
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  syncUndoButtons();
}

export function recordListUndo(beforeList, afterList, beforeSelected, afterSelected) {
  undoStack.push({
    kind: 'list',
    before: structuredClone(beforeList),
    after: structuredClone(afterList),
    beforeSelected,
    afterSelected,
  });
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  syncUndoButtons();
}

export function syncUndoButtons() {
  const busy = projectBusy();
  const u = $('#undoSentence'),
    r = $('#redoSentence');
  if (u) u.disabled = busy || undoStack.length === 0;
  if (r) r.disabled = busy || redoStack.length === 0;
}

async function applyHistoryEntry(entry, direction) {
  if (!state.project || projectBusy()) return;
  if (entry.kind === 'fields') {
    const sentence = state.project.sentences.find((s) => s.id === entry.id);
    if (!sentence) return;
    const patch = direction === 'undo' ? entry.before : entry.after;
    Object.assign(sentence, patch);
    if ('source_text' in patch || 'zh_text' in patch || 'start_seconds' in patch || 'end_seconds' in patch) {
      if ('zh_text' in patch) sentence.target_text = patch.zh_text;
      sentence.tts_file = null;
      sentence.tts_url = null;
      sentence.tts_candidates = [];
    }
    hooks.setSelected(sentence.id);
    await hooks.saveRows(sentence.id);
  } else if (entry.kind === 'list') {
    const snap = direction === 'undo' ? entry.before : entry.after;
    state.project.sentences = structuredClone(snap);
    hooks.setSelected(direction === 'undo' ? entry.beforeSelected : entry.afterSelected);
    await hooks.saveRows();
  }
  hooks.renderProject();
}

export async function undoEdit() {
  if (!undoStack.length || projectBusy()) return;
  const entry = undoStack.pop();
  redoStack.push(entry);
  await applyHistoryEntry(entry, 'undo');
  syncUndoButtons();
}

export async function redoEdit() {
  if (!redoStack.length || projectBusy()) return;
  const entry = redoStack.pop();
  undoStack.push(entry);
  await applyHistoryEntry(entry, 'redo');
  syncUndoButtons();
}

export async function playSourceSlice(sentence) {
  if (!sentence) return;
  const source = $('#sourcePlayer'),
    dub = $('#dubPlayer');
  dub.pause();
  source.currentTime = sentence.start_seconds;
  sliceEndAt = sentence.end_seconds;
  $('#playSource').textContent = 'Ⅱ';
  await source.play().catch(() => {});
}

export async function playDubClip(sentence) {
  if (!sentence?.tts_url) return;
  const source = $('#sourcePlayer'),
    dub = $('#dubPlayer');
  sliceEndAt = null;
  source.pause();
  $('#playSource').textContent = '▶';
  dub.src = sentence.tts_url;
  dub.currentTime = 0;
  await dub.play().catch(() => {});
}

export function checkSourceSliceStop(source) {
  if (sliceEndAt !== null && source.currentTime >= sliceEndAt) {
    sliceEndAt = null;
    source.pause();
    $('#playSource').textContent = '▶';
  }
}

export function clearSourceSliceStop() {
  sliceEndAt = null;
}

export function renderVirtualRows(sentences, selectedId) {
  if (!state.project) return;
  const box = $('.tablebox');
  const busy = projectBusy();
  const total = sentences.length;
  const scrollTop = box ? box.scrollTop : 0;
  const viewHeight = box && box.clientHeight > 0 ? box.clientHeight : 560;
  const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
  const visibleCount = Math.ceil(viewHeight / ROW_HEIGHT) + OVERSCAN * 2;
  const endIdx = Math.min(total, startIdx + Math.max(40, visibleCount));
  const topPad = startIdx * ROW_HEIGHT;
  const bottomPad = Math.max(0, (total - endIdx) * ROW_HEIGHT);
  const srcLang = state.project.source_language || 'ja';

  const sliceHTML = sentences
    .slice(startIdx, endIdx)
    .map((sentence) => {
      const txt = sentence.zh_text ?? sentence.target_text ?? '';
      return `<tr data-sentence="${esc(sentence.id)}" class="${!sentence.enabled ? 'off' : ''} ${selectedId === sentence.id ? 'sel' : ''}"><td><input type="checkbox" data-row-field="enabled"${sentence.enabled ? ' checked' : ''}${busy ? ' disabled' : ''} aria-label="${t('启用句子')}"></td><td class="id">${esc(sentence.id)}${sentence.speaker_id ? ` <span class="badge">${esc(sentence.speaker_id)}</span>` : ''}</td><td class="time"><button type="button" class="play src-play" data-play-source="${esc(sentence.id)}" title="${t('试听原声片段')}" aria-label="${t('试听原声片段')}">▶</button><span class="time-span">${time(sentence.start_seconds)} – ${time(sentence.end_seconds)}</span></td><td lang="${esc(srcLang)}" contenteditable="${!busy}" data-row-field="source_text">${esc(sentence.source_text)}</td><td contenteditable="${!busy}" data-row-field="zh_text">${esc(txt)}</td><td class="dub">${dubCell(sentence)}</td><td class="orig">${originalCell(sentence)}${flagsCell(sentence)}</td></tr>`;
    })
    .join('');

  const topSpacer = topPad > 0 ? `<tr class="vs-spacer" aria-hidden="true"><td colspan="7" style="height:${topPad}px"></td></tr>` : '';
  const botSpacer = bottomPad > 0 ? `<tr class="vs-spacer" aria-hidden="true"><td colspan="7" style="height:${bottomPad}px"></td></tr>` : '';
  $('#rows').innerHTML = `${topSpacer}${sliceHTML}${botSpacer}`;
}

export function scrollToSentenceIndex(index) {
  const box = $('.tablebox');
  if (!box || index < 0) return;
  const targetTop = index * ROW_HEIGHT;
  const viewH = box.clientHeight || 480;
  if (targetTop < box.scrollTop + ROW_HEIGHT || targetTop > box.scrollTop + viewH - ROW_HEIGHT * 2) {
    box.scrollTop = Math.max(0, targetTop - Math.floor(viewH * 0.35));
  }
}

export function renderSentenceControls(selectedId, totalCount, syncBusy) {
  const sentence = state.project?.sentences.find((s) => s.id === selectedId);
  $('#sentenceControls').innerHTML =
    `<div class="tools"><button class="btn" id="addSentence">${t('新增句子')}</button>${sentence ? `<button class="btn" id="deleteSentence">${t('删除句子')}</button><button class="btn plain" id="sentenceReference">${t('设为项目音色参考')}</button>` : ''}<button class="btn plain small" id="undoSentence" title="Ctrl+Z">${t('撤销')}</button><button class="btn plain small" id="redoSentence" title="Ctrl+Y">${t('重做')}</button><span class="saved kbd-hint">${t('共 {count} 句 · ↑↓ 选句 · 空格试听 · 回车编辑', { count: totalCount })}</span></div>` +
    (sentence
      ? `<div class="waveform-wrap"><canvas id="sentenceWaveform" height="56" aria-label="${t('句子波形与起止点拖动')}"></canvas><span class="waveform-meta" id="waveformLabel"></span></div><details><summary>${t('所选句子')} · ${esc(sentence.id)}</summary><div class="card group">${field('说话人', `<input type="text" data-detail="speaker_id" value="${esc(sentence.speaker_id || '')}" placeholder="S01">`)}${field('开始（秒）', `<input type="number" step="any" data-detail="start_seconds" value="${sentence.start_seconds}">`)}${field('结束（秒）', `<input type="number" step="any" data-detail="end_seconds" value="${sentence.end_seconds}">`)}${field('原声音量微调（dB）', `<input type="number" step="any" min="-60" max="12" data-detail="original_audio_gain_db" value="${sentence.original_audio_gain_db}">`)}${field('播放配音', `<input type="checkbox" data-detail="chinese_audio_enabled"${sentence.chinese_audio_enabled ? ' checked' : ''}>`)}${field('配音音量微调（dB）', `<input type="number" step="any" min="-60" max="12" data-detail="chinese_audio_gain_db" value="${sentence.chinese_audio_gain_db}">`)}</div><p class="stat">${esc(sentence.script_review_note || (sentence.error_code ? `[${sentence.error_code}] ${sentence.error}` : sentence.error) || '')}</p></details>`
      : '');
  syncBusy();
  syncUndoButtons();
  if (sentence) refreshWaveform(sentence);
}

export function initializeSentenceEditor(editorHooks) {
  hooks = editorHooks;
  let rafPending = false;
  const box = $('.tablebox');
  if (box) {
    box.addEventListener(
      'scroll',
      () => {
        if (rafPending || !state.project) return;
        rafPending = true;
        requestAnimationFrame(() => {
          rafPending = false;
          renderVirtualRows(hooks.filteredSentences(), hooks.getSelected());
          hooks.syncBusy();
        });
      },
      { passive: true },
    );
  }

  initializeWaveform({
    getSelectedSentence: () => state.project?.sentences.find((s) => s.id === hooks.getSelected()),
    onBoundaryCommit: async (sentence, beforeStart, beforeEnd) => {
      recordFieldUndo(
        sentence.id,
        { start_seconds: beforeStart, end_seconds: beforeEnd },
        { start_seconds: sentence.start_seconds, end_seconds: sentence.end_seconds },
      );
      sentence.tts_file = null;
      sentence.tts_url = null;
      sentence.tts_candidates = [];
      await hooks.saveRows(sentence.id);
    },
    onAuditionSlice: (sentence) => playSourceSlice(sentence),
  });

  document.addEventListener(
    'keydown',
    guard(async (event) => {
      if (state.page !== 'project' || !state.project) return;
      if ($('#dialog')?.classList.contains('on') || $('#wizard')?.classList.contains('on')) return;

      const editable = event.target.closest('[contenteditable="true"]');
      if (editable) {
        if (event.key === 'Escape' || (event.key === 'Enter' && (event.ctrlKey || event.metaKey))) {
          event.preventDefault();
          editable.blur();
        }
        return;
      }

      const inControl = event.target.closest('input, textarea, select');
      const mod = event.ctrlKey || event.metaKey;
      if (!inControl && mod && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) await redoEdit();
        else await undoEdit();
        return;
      }
      if (!inControl && mod && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        await redoEdit();
        return;
      }
      if (inControl || event.target.closest('button, summary, a')) return;

      const list = hooks.filteredSentences();
      if (!list.length) return;
      const curIdx = Math.max(0, list.findIndex((s) => s.id === hooks.getSelected()));

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const nextIdx = Math.max(0, Math.min(list.length - 1, curIdx + (event.key === 'ArrowDown' ? 1 : -1)));
        const target = list[nextIdx];
        hooks.setSelected(target.id);
        $('#sourcePlayer').currentTime = target.start_seconds;
        scrollToSentenceIndex(nextIdx);
        renderVirtualRows(list, target.id);
        renderSentenceControls(target.id, list.length, hooks.syncBusy);
      } else if (event.key === ' ') {
        event.preventDefault();
        const current = list[curIdx];
        const source = $('#sourcePlayer'),
          dub = $('#dubPlayer');
        if (!source.paused || !dub.paused) {
          source.pause();
          dub.pause();
          clearSourceSliceStop();
          $('#playSource').textContent = '▶';
        } else if (hooks.getListenMode() === 'dub' && current.tts_url) {
          await playDubClip(current);
        } else {
          await playSourceSlice(current);
        }
      } else if (event.key === 'Enter' && !projectBusy()) {
        event.preventDefault();
        scrollToSentenceIndex(curIdx);
        renderVirtualRows(list, list[curIdx].id);
        const cell = $(`#rows tr[data-sentence="${CSS.escape(list[curIdx].id)}"] [data-row-field="zh_text"]`);
        if (cell) {
          cell.focus();
          const sel = window.getSelection();
          sel?.selectAllChildren(cell);
          sel?.collapseToEnd();
        }
      }
    }),
  );
}
