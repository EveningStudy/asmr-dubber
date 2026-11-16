import { $, state, esc, t } from './session.js';

const ACTIVE = ['queued', 'running', 'cancelling'];
// Download progress restarts for every file, so an estimate from it would mislead.
const NO_ESTIMATE = ['download', 'import_models', 'update', 'repair', 'health', 'diagnostic'];
const PAGES = { batch: 'batch', download: 'models', import_models: 'models', repair: 'models', update: 'models' };
let collapsed = false;

export function remainingSeconds(task, now = Date.now() / 1000) {
  if (NO_ESTIMATE.includes(task.kind) || !(task.total > 0) || !(task.current > 0)) return null;
  const ratio = task.current / task.total;
  const elapsed = now - task.started_at;
  // Too early in the run, the first minutes say little about the rest.
  if (ratio >= 1 || ratio < 0.03 || elapsed < 5) return null;
  return (elapsed * (1 - ratio)) / ratio;
}

export function durationText(seconds) {
  if (seconds >= 3600) {
    const hours = Math.floor(seconds / 3600);
    return t('{hours} 小时 {minutes} 分钟', { hours, minutes: Math.round((seconds % 3600) / 60) });
  }
  if (seconds >= 90) return t('{minutes} 分钟', { minutes: Math.round(seconds / 60) });
  return t('{seconds} 秒', { seconds: Math.max(5, Math.round(seconds / 5) * 5) });
}

function where(task) {
  const manifest = task.request.project;
  if (!manifest) return '';
  const parts = manifest.split(/[\\/]/);
  return parts[parts.length - 2] || '';
}

function row(task) {
  const ratio = task.total > 0 ? Math.min(100, (task.current / task.total) * 100) : 0;
  const left = remainingSeconds(task);
  const page = PAGES[task.kind];
  const title = `${esc(t(task.kind))}${where(task) ? ` · ${esc(where(task))}` : ''}`;
  const meta = [
    task.status === 'cancelling' ? t('正在暂停…') : task.total > 0 ? `${Math.round(ratio)}%` : t('处理中'),
    left === null ? '' : t('剩余约 {time}', { time: durationText(left) }),
  ]
    .filter(Boolean)
    .join(' · ');
  return `<div class="tp-row" data-task-row="${esc(task.id)}"><div class="tp-main"><div class="tp-title">${
    page ? `<a href="#" data-go="${page}">${title}</a>` : title
  }</div><div class="tp-msg">${esc(t(task.message || ''))}</div><div class="bar"><i style="width:${ratio}%"></i></div><div class="tp-meta">${esc(meta)}</div></div><button class="btn small" data-cancel="${esc(task.id)}"${
    task.status === 'cancelling' ? ' disabled' : ''
  }>${t('暂停')}</button></div>`;
}

export function renderTaskPanel() {
  const box = $('#taskPanel');
  if (!box) return;
  const active = state.tasks.filter((task) => ACTIVE.includes(task.status));
  box.hidden = !active.length;
  if (!active.length) return;
  box.classList.toggle('collapsed', collapsed);
  box.innerHTML = `<button class="tp-head" id="taskPanelToggle" aria-expanded="${!collapsed}"><b>${esc(
    t('正在运行的任务'),
  )}</b><span class="tp-count">${active.length}</span><span class="tp-chevron" aria-hidden="true">${collapsed ? '▴' : '▾'}</span></button>${
    collapsed ? '' : `<div class="tp-body">${active.map(row).join('')}</div>`
  }`;
}

export function initializeTaskPanel() {
  document.addEventListener('click', (event) => {
    if (!event.target.closest('#taskPanelToggle')) return;
    collapsed = !collapsed;
    renderTaskPanel();
  });
  // Links inside the panel must not jump to "#".
  $('#taskPanel').addEventListener('click', (event) => {
    if (event.target.closest('a[data-go]')) event.preventDefault();
  });
}
