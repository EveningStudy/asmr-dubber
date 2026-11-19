import { $, $$, state, api, t, guard, esc, startTask } from './session.js';
import { initializeRevise, renderProposals } from './project_revise.js';

const SKIPPED = new Set(['skipped_sound', 'skipped_filler']);

export function renderTranslationPanels(project) {
  renderRecovery(project);
  renderTranslationResult(project);
  renderProposals(project);
  renderBackup(project);
}

export function initializeTranslateAll(dialog, field, getSelected) {
  initializeRevise(getSelected);
  $('#translateAll').onclick = guard(() =>
    dialog(
      '全部重新翻译',
      `<p>${t('现有译文会逐句被新的翻译替换，替换前会自动保存一份备份，可以随时恢复。已生成的配音需要重做。')}</p>${field('也覆盖我手动改过的句子', '<input type="checkbox" id="overwriteEdited">')}`,
      '重新翻译',
      () => startTask('translate', { force: true, overwrite_edited: $('#overwriteEdited').checked }),
    ),
  );
}

export function renderTranslationResult(project) {
  const box = $('#translationResult');
  const failed = project.sentences.filter((s) => s.enabled && s.status === 'translation_failed');
  const skipped = project.sentences.filter(
    (s) => SKIPPED.has(s.status) || (!s.enabled && s.status === 'translation_failed'),
  );
  const done = project.sentences.filter(
    (s) => s.enabled && s.target_text && !['translation_failed', 'skipped_sound'].includes(s.status),
  ).length;
  const tooLong = project.sentences.filter(
    (s) => s.enabled && s.target_text && (s.translation_warning || '').includes('超出时长预算'),
  );
  box.hidden = !(done || skipped.length || failed.length || tooLong.length);
  if (box.hidden) return;
  const failures = failed
    .slice(0, 30)
    .map(
      (s) =>
        `<li><button class="btn plain small" data-locate="${esc(s.id)}">${esc(s.id)}</button><span class="why">${esc(s.error || '')}</span></li>`,
    )
    .join('');
  const skippedList = skipped
    .slice(0, 30)
    .map((s) => `<li><button class="btn plain small" data-locate="${esc(s.id)}">${esc(s.id)}</button><span>${esc(s.source_text)}</span></li>`)
    .join('');
  box.innerHTML =
    `<div class="t">${t('翻译结果')}</div>` +
    `<p class="stat">${t('成功 {done} 句，跳过 {skipped} 句，失败 {failed} 句。', { done, skipped: skipped.length, failed: failed.length })}</p>` +
    (tooLong.length
      ? `<div class="actions"><button class="btn" id="compressTooLong">${t('一键压缩过长译文（{count} 句）', { count: tooLong.length })}</button></div>`
      : '') +
    (failed.length
      ? `<p class="stat">${t('失败的句子，点编号可在表格里定位。')}</p><ul>${failures}</ul><div class="actions"><button class="btn" data-task="translate">${t('只重试失败句')}</button><button class="btn plain" id="disableFailed">${t('把这些句子设为不配音')}</button></div>`
      : '') +
    (skipped.length
      ? `<details><summary>${t('跳过的句子（{count}）', { count: skipped.length })}</summary><ul>${skippedList}</ul></details>`
      : '');
  const running = state.tasks.some(
    (task) => task.request.project === project.manifest && ['queued', 'running', 'cancelling'].includes(task.status),
  );
  box.querySelectorAll('[data-task], #compressTooLong').forEach((button) => {
    button.toggleAttribute('disabled', running);
  });
}

export function renderBackup(project) {
  const backup = project.translation_backup;
  $('#restoreBackup').hidden = !backup;
  if (backup) $('#restoreBackup').textContent = t('恢复上一版译文（{count} 句）', { count: backup.count });
}

export function renderRecovery(project) {
  const count = (project.recoverable_translations || []).length;
  $('#restoreTranslations').hidden = !count;
  $('#restoreTranslations').textContent = t('恢复误跳过的句子（{count}）', {count});
}

export function initializeRecovery(saveRows, renderProject, dialog) {
  $('#restoreBackup').onclick = guard(() =>
    dialog('恢复译文备份', `<p>${t('译文会回到最近一次重新翻译之前的版本。')}</p>`, '恢复', () =>
      startTask('restore_translations'),
    ),
  );
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-locate]');
    if (target) window.dispatchEvent(new CustomEvent('locate-sentence', { detail: target.dataset.locate }));
  });
  $('#translationResult').addEventListener(
    'click',
    guard(async (event) => {
      if (event.target.closest('#compressTooLong')) {
        await startTask('translate', { compress: true });
        return;
      }
      if (!event.target.closest('#disableFailed')) return;
      state.project.sentences.forEach((sentence) => {
        if (sentence.enabled && sentence.status === 'translation_failed') sentence.enabled = false;
      });
      await saveRows();
      state.project = await api('projects/get', { project: state.project.manifest });
      renderProject();
    }),
  );
  $('#restoreTranslations').onclick = guard(async () => {
    const ids = new Set(state.project.recoverable_translations || []);
    state.project.sentences.forEach(sentence => {
      if (ids.has(sentence.id)) sentence.enabled = true;
    });
    await saveRows();
    state.project = await api('projects/get', {project: state.project.manifest});
    renderProject();
  });
}
