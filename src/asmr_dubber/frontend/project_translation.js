import { $, state, api, t, guard } from './session.js';

export function renderRecovery(project) {
  const count = (project.recoverable_translations || []).length;
  $('#restoreTranslations').hidden = !count;
  $('#restoreTranslations').textContent = t('恢复误跳过的句子（{count}）', {count});
}

export function initializeRecovery(saveRows, renderProject) {
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
