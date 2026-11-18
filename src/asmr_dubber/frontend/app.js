import { $, $$, state, api, t, localize, notice, guard, taskHTML, flushSaves } from './session.js';
import { initializeForms, renderForms, saveParameters } from './forms.js';
import {
  initializeProjects,
  refreshHome,
  openProject,
  renderProject,
  finishCreation,
  syncProjectControls,
} from './projects.js';
import { initializeBatch, refreshBatch, renderBatch } from './batch.js';
import { initializeModels, refreshModels, renderModels } from './models.js';
import { initializeSettings, renderKeys, refreshLogs, refreshStorage } from './settings.js';
import { initializeUpdates, renderUpdate, updateFinished } from './updates.js';
import { initializeWizard, wizardTick } from './wizard.js';
import { initializePresets } from './presets.js';
import { initializeTaskPanel, renderTaskPanel } from './task_panel.js';
import { notifyBatchFinished } from './notify.js';

async function navigate(page, setTab) {
  if (page === 'project' && !state.project) return;
  await flushSaves({ ignoreError: true });
  state.page = page;
  $$('.page').forEach((section) => section.classList.toggle('on', section.id === page));
  $$('.nav button').forEach((button) =>
    button.classList.toggle('on', button.dataset.page === (page === 'project' ? 'home' : page)),
  );
  if (page === 'home') await refreshHome();
  if (page === 'batch') await refreshBatch();
  if (page === 'models') await refreshModels();
  if (setTab) await setSettingsTab(setTab);
}
async function setSettingsTab(tab) {
  $$('#setNav button').forEach((button) => button.classList.toggle('on', button.dataset.set === tab));
  $$('.setview').forEach((view) => view.classList.toggle('on', view.dataset.setview === tab));
  if (tab === 'storage') await refreshStorage();
  if (tab === 'logs') await refreshLogs();
}
function setStep(step) {
  state.step = Number(step);
  $$('#steps button').forEach((button) =>
    button.classList.toggle('on', Number(button.dataset.step) === state.step),
  );
  $$('.stepview').forEach((view) => view.classList.toggle('on', Number(view.dataset.view) === state.step));
}
function refreshTaskViews() {
  const tasks = state.tasks.filter((task) => task.request.project === state.project?.manifest);
  // Running tasks always stay visible; of the finished ones only the latest few.
  const running = (task) => ['queued', 'running', 'cancelling'].includes(task.status);
  const shownTasks = tasks.filter((task, index) => running(task) || index >= tasks.length - 3);
  $('#projectTask').innerHTML = shownTasks.map(taskHTML).join('');
  renderTaskPanel();
  wizardTick();
  renderUpdate();
  renderBatch();
  renderModels();
  syncProjectControls();
}
let polling = false, pollingError = '';
async function pollTasks() {
  if (polling || !state.boot) return;
  polling = true;
  let listed = false;
  try {
    state.tasks = await api('tasks/list');
    listed = true;
    pollingError = '';
    for (const task of state.tasks) {
      const key = task.id + task.updated_at;
      if (['queued', 'running', 'cancelling'].includes(task.status) || state.seenTasks.has(key))
        continue;
      if (task.status === 'completed') {
        if (task.kind === 'preview_edge' && task.result?.url) {
          $('#dubPlayer').src = task.result.url;
          await $('#dubPlayer').play().catch(() => {});
        }
        if (task.kind === 'create') await finishCreation(task);
        else if (task.request.project === state.project?.manifest) {
          await flushSaves({ ignoreError: true });
          state.project = await api('projects/get', { project: state.project.manifest });
          $('#saveState').textContent = t('修改会自动保存');
          renderProject();
        }
        if (['download', 'import_models'].includes(task.kind)) await refreshModels();
        if (['health', 'repair', 'diagnostic'].includes(task.kind) && state.page === 'settings')
          await refreshLogs();
        if (task.kind === 'batch') notifyBatchFinished(task);
        if (task.kind === 'update') updateFinished(task);
        else notice(task.result?.message || t('任务完成'));
      } else if (task.status === 'failed') {
        if (task.kind === 'batch') notifyBatchFinished(task);
        if (state.project && task.request.project === state.project.manifest) {
          await flushSaves({ ignoreError: true });
          state.project = await api('projects/get', { project: state.project.manifest });
          $('#saveState').textContent = t('修改会自动保存');
          renderProject();
        }
        notice(task.error, true, { sticky: true });
      } else if (
        task.status === 'cancelled' &&
        state.project &&
        task.request.project === state.project.manifest
      ) {
        await flushSaves({ ignoreError: true });
        state.project = await api('projects/get', { project: state.project.manifest });
        $('#saveState').textContent = t('修改会自动保存');
        renderProject();
      }
      state.seenTasks.add(key);
    }
    refreshTaskViews();
  } catch (error) {
    if (listed) refreshTaskViews();
    if (pollingError !== error.message) notice(error.message, true, { dedupe: true });
    pollingError = error.message;
  } finally {
    polling = false;
  }
}
document.addEventListener(
  'click',
  guard(async (event) => {
    const page = event.target.closest('[data-page],[data-go]');
    if (page) {
      await navigate(page.dataset.page || page.dataset.go, page.dataset.set);
      return;
    }
    const tab = event.target.closest('#setNav [data-set]');
    if (tab) {
      await setSettingsTab(tab.dataset.set);
      return;
    }
    const step = event.target.closest('[data-step]');
    if (step) {
      setStep(step.dataset.step);
      return;
    }
    const language = event.target.closest('[data-language]');
    if (language) {
      await saveParameters({ ui_language: language.dataset.language }, 'def');
      return;
    }
    const cancel = event.target.closest('[data-cancel]');
    if (cancel) {
      await api('tasks/cancel', { identifier: cancel.dataset.cancel });
      await pollTasks();
      return;
    }
    const resume = event.target.closest('[data-resume]');
    if (resume) {
      await api('tasks/resume', { identifier: resume.dataset.resume });
      await pollTasks();
      return;
    }
    if (event.target.closest('[data-close]') || event.target.classList.contains('mask'))
      $('#dialog').classList.remove('on');
  }),
);
window.addEventListener(
  'navigate',
  guard((event) => navigate(event.detail.page, event.detail.tab)),
);
window.addEventListener('tasks-changed', refreshTaskViews);
window.addEventListener('settings-changed', () => {
  state.language = state.boot.settings.ui_language;
  localize();
  renderForms();
  renderProject();
  renderKeys();
  renderBatch();
  renderModels();
});
window.addEventListener('beforeunload', (event) => {
  if (state.saveError) {
    event.preventDefault();
    event.returnValue = '';
  }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') $('#dialog').classList.remove('on');
});
async function boot() {
  state.boot = await api('bootstrap');
  state.tasks = state.boot.tasks;
  state.queue = state.boot.queue;
  state.language = state.boot.settings.ui_language;
  state.tasks.forEach((task) => {
    if (!['queued', 'running', 'cancelling'].includes(task.status))
      state.seenTasks.add(task.id + task.updated_at);
  });
  initializeForms();
  initializeProjects();
  initializeBatch();
  initializeModels();
  initializeSettings();
  initializeUpdates();
  initializePresets();
  initializeTaskPanel();
  localize();
  renderForms();
  renderKeys();
  renderBatch();
  renderTaskPanel();
  await refreshHome();
  await refreshModels();
  initializeWizard();
  setInterval(pollTasks, 1200);
}
boot().catch((error) => notice(error.message, true));
