import { state, api, esc, t, options, flushSaves, startTask, guard, projectBusy, notice } from './session.js';

// Original-voice handling per sentence: Sentence.original_audio_enabled is null (auto),
// true (keep the original) or false (replace / mute it).
const ORIGINAL_CHOICES = [
  ['自动', 'default'],
  ['保留原声', 'on'],
  ['替换掉原声', 'off'],
];
const ORIGINAL_VALUES = { default: null, on: true, off: false };
const originalKey = (sentence) =>
  sentence.original_audio_enabled === null || sentence.original_audio_enabled === undefined
    ? 'default'
    : sentence.original_audio_enabled
      ? 'on'
      : 'off';

// Residual-voice flag from the last replacement mix (an energy estimate, see replacement-report.json).
export const residualFlag = (sentence) => state.project?.replacement_report?.[sentence.id] || null;
export const dubWarning = (sentence) => (sentence.tts_file && sentence.tts_warning) || '';
export const needsCheck = (sentence) =>
  Boolean(sentence.enabled && (sentence.translation_warning || dubWarning(sentence) || residualFlag(sentence)));

function flagsHTML(sentence) {
  const warning = dubWarning(sentence),
    flag = residualFlag(sentence),
    busy = projectBusy() ? ' disabled' : '';
  let html = '';
  if (sentence.enabled && sentence.translation_warning)
    html += `<div class="flag warn" title="${esc(t(sentence.translation_warning))}"><b>⚠ ${t('译文可能有问题')}</b><small>${esc(t(sentence.translation_warning))}</small><button class="btn plain small" data-retranslate="${esc(sentence.id)}"${busy}>${t('重译这一句')}</button></div>`;
  if (warning)
    html += `<div class="flag bad" title="${esc(t(warning))}"><b>⚠ ${t('配音质检未通过')}</b><small>${esc(t(warning))}</small></div>`;
  if (flag) {
    const reasons = flag.reasons
      .map((reason) =>
        reason === 'residual_voice'
          ? t('替换后残留的声音偏大（{db} dB）', { db: Number(flag.residual_db).toFixed(1) })
          : t('区间外紧邻处有未被句子覆盖的人声（约 {seconds} 秒）', {
              seconds: Math.max(flag.uncovered_before_seconds, flag.uncovered_after_seconds).toFixed(1),
            }),
      )
      .join('；');
    html += `<div class="flag warn" title="${esc(reasons)}"><b>⚠ ${t('疑似残留原声')}</b><small>${esc(reasons)}</small><button class="btn plain small" data-keep-original="${esc(sentence.id)}"${busy}>${t('这一句改为保留原声')}</button></div>`;
  }
  return html;
}

// Cells of one sentence row that belong to the review features: original-voice choice,
// dub redo button and the warning column.
export function dubCell(sentence) {
  if (!sentence.tts_url)
    return sentence.error
      ? `<span class="batch-error" title="${esc(sentence.error)}">${t('翻译或配音待重试')}</span>`
      : t(sentence.enabled ? '未生成' : '不配音');
  const redo = projectBusy() ? ' disabled' : '';
  return `<button class="play" data-play="${esc(sentence.id)}" aria-label="${t('试听')}">▶</button>${Number(sentence.tts_duration_seconds || 0).toFixed(1)} ${t('秒')} <button class="play redo" data-regenerate="${esc(sentence.id)}" title="${t('重新生成这一句配音')}" aria-label="${t('重新生成这一句配音')}"${redo}>↻</button>`;
}
export function originalCell(sentence) {
  const disabled = projectBusy() ? ' disabled' : '';
  return `<select data-original aria-label="${t('原声处理')}"${disabled}>${options(ORIGINAL_CHOICES, originalKey(sentence))}</select>`;
}
export const flagsCell = flagsHTML;

export function initializeReview({ save, render, getSelected }) {
  Object.defineProperty(state, 'selectedSentence', { get: getSelected });
  const find = (id) => state.project.sentences.find((s) => s.id === id);
  async function reload() {
    await flushSaves();
    state.project = await api('projects/get', { project: state.project.manifest });
    render();
  }
  const busy = () => {
    if (projectBusy()) throw new Error(t('请等当前任务结束后再操作。'));
  };
  document.addEventListener(
    'change',
    guard(async (event) => {
      const select = event.target.closest('select[data-original]');
      if (!select) return;
      busy();
      const sentence = find(select.closest('tr').dataset.sentence);
      sentence.original_audio_enabled = ORIGINAL_VALUES[select.value];
      await save(sentence.id);
      await reload();
    }),
  );
  document.addEventListener(
    'click',
    guard(async (event) => {
      const keep = event.target.closest('[data-keep-original]');
      if (keep) {
        busy();
        find(keep.dataset.keepOriginal).original_audio_enabled = true;
        await save(keep.dataset.keepOriginal);
        await reload();
        notice(t('这一句将保留原声。重新导出后生效。'));
        return;
      }
      const again = event.target.closest('[data-retranslate]');
      if (again) {
        busy();
        await startTask('translate', { sentence_ids: [again.dataset.retranslate] });
        render();
        notice(t('正在重译 {id}。', { id: again.dataset.retranslate }));
        return;
      }
      const redo = event.target.closest('[data-regenerate]');
      if (redo) {
        busy();
        const sentence = find(redo.dataset.regenerate);
        // The ordinary dubbing task, limited to this sentence and forced to make a new take.
        await startTask('synthesize', { sentence_ids: [sentence.id], force: true });
        render();
        notice(t('正在重新生成 {id} 的配音。', { id: sentence.id }));
      }
    }),
  );
}
