import { $, $$, state, api, t, localize, notice, guard, taskHTML, flushSaves } from './session.js';
import { initializeForms, renderForms, saveParameters } from './forms.js';
import { initializeProjects, refreshHome, openProject, renderProject, finishCreation } from './projects.js';
import { initializeBatch, refreshBatch, renderBatch } from './batch.js';
import { initializeModels, refreshModels, renderModels } from './models.js';
import { initializeSettings, renderKeys, refreshLogs, refreshStorage } from './settings.js';

async function navigate(page, setTab) {
  if (page === 'project' && !state.project) return;
  await flushSaves();
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
  $('#projectTask').innerHTML = tasks.slice(-3).map(taskHTML).join('');
  renderBatch();
  renderModels();
  if (state.project)
    $$('[data-task],#translateAll,#exportProject').forEach(
      (button) =>
        (button.disabled = tasks.some((task) => ['queued', 'running', 'cancelling'].includes(task.status))),
    );
}
let polling = false;
async function pollTasks() {
  if (polling || !state.boot) return;
  polling = true;
  try {
    state.tasks = await api('tasks/list');
    for (const task of state.tasks) {
      if (
        ['queued', 'running', 'cancelling'].includes(task.status) ||
        state.seenTasks.has(task.id + task.updated_at)
      )
        continue;
      state.seenTasks.add(task.id + task.updated_at);
      if (task.status === 'completed') {
        if (task.kind === 'preview_edge' && task.result?.url) {
          $('#dubPlayer').src = task.result.url;
          await $('#dubPlayer').play();
        }
        if (task.kind === 'create') await finishCreation(task);
        else if (task.request.project === state.project?.manifest) {
          await flushSaves();
          state.project = await api('projects/get', { project: state.project.manifest });
          renderProject();
        }
        if (['download', 'import_models'].includes(task.kind)) await refreshModels();
        if (['health', 'repair', 'diagnostic'].includes(task.kind) && state.page === 'settings')
          await refreshLogs();
        notice(task.result?.message || t('任务完成'));
      } else if (task.status === 'failed') notice(task.error, true);
    }
    refreshTaskViews();
  } catch (error) {
    notice(error.message, true);
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
  localize();
  renderForms();
  renderKeys();
  renderBatch();
  await refreshHome();
  await refreshModels();
  setInterval(pollTasks, 1200);
}
boot().catch((error) => notice(error.message, true));
