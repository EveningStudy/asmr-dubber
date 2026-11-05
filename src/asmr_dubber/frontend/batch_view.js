import { state, esc, t, taskHTML } from './session.js';

const labels = {
  queued: '等待中', running: '正在处理…', completed: '已完成', failed: '失败',
  cancelled: '已暂停', interrupted: '已中断', awaiting_reference: '等待音色参考',
};
const contentLabels = {
  both: '完全配音版 + 双语版', replacement: '完全配音版', dubbing: '双语版',
  subtitles: '只要字幕文件', source_subtitles: '原声 + 字幕，不配音',
};

function statusOf(plan, active) {
  const recent = [...state.tasks].filter(task => task.kind === 'batch')
    .sort((a,b) => b.started_at - a.started_at);
  for (const task of recent) {
    const entry = task.plans?.[plan.plan_id] || task.result?.outcomes?.[plan.plan_id];
    if (entry) {
      if (task.reference?.plan_id === plan.plan_id) return {...entry, status:'awaiting_reference'};
      return entry;
    }
    if (task === active) return {status:'queued'};
    if (task.status === 'completed' && task.result?.plans?.includes(plan.plan_id))
      return {status:'completed'};
  }
  return {status:'queued'};
}

export function renderQueue(active) {
  if (!state.queue.length) return `<div class="card empty"><b>${t('队列是空的')}</b><p>${t('添加一个已解压的作品文件夹开始。')}</p></div>`;
  const entries = state.queue.map(plan => statusOf(plan, active));
  const count = status => entries.filter(e => e.status === status).length;
  const summary = t('已完成 {done}/{total} · 失败 {failed} · 等待 {waiting}', {
    done:count('completed'), total:entries.length, failed:count('failed'), waiting:count('queued'),
  });
  return `<div class="label"><b>${t('队列')}</b><span>${t('按顺序处理，可拖动调整')}</span></div><p class="batch-summary" role="status">${esc(summary)}</p><div class="card q">${state.queue.map((plan,index) => {
    const entry = entries[index], status = entry.status || 'queued';
    return `<div class="row" draggable="${!active}" data-plan-row="${esc(plan.plan_id)}" data-plan-status="${status}"><div class="grow"><div class="t">${esc(plan.folder.split(/[\\/]/).pop())}</div><div class="d">${t('{count} 条音轨',{count:plan.sources.length})} · ${esc(t(contentLabels[plan.edition.output_policy?.content] || '双语版'))} · ${esc(t(plan.layout))}</div>${entry.error ? `<div class="d batch-error">${esc(entry.error)}</div>` : ''}</div><span class="batch-status ${status}">${t(labels[status] || status)}</span>${status === 'completed' ? `<button class="btn" data-batch-output="${esc(plan.output_root)}">${t('打开文件夹')}</button>` : ''}${active ? '' : `<button class="btn" data-edit-plan="${plan.plan_id}">${t('编辑')}</button><button class="btn plain" data-remove-plan="${plan.plan_id}">${t('删除')}</button><button class="btn plain" data-plan-up="${index}"${index===0 ? ' disabled':''}>↑</button>`}</div>`;
  }).join('')}</div>${active ? taskHTML(active) : ''}`;
}

export function renderHistory(history) {
  return [...history].sort((a,b) => b.started_at - a.started_at).map(task => {
    const empty = /队列还是空的|队列没有作品/.test(task.error || '');
    const date = new Date(task.started_at * 1000).toLocaleString();
    const entries = Object.values(task.plans || task.result?.outcomes || {});
    const body = empty
      ? `<div class="card empty">${t('这次运行没有可执行作品，未处理任何文件。')}</div>`
      : taskHTML(task);
    return `<div class="label"><b>${esc(date)}</b><button class="btn plain small" data-dismiss-batch="${task.id}">${t('隐藏记录')}</button></div>${body}${empty ? '' : entries.map(entry => `<div class="row"><span class="grow">${esc(entry.work || '')}${entry.error ? `<small class="batch-error">${esc(entry.error)}</small>` : ''}</span><span class="batch-status ${entry.status}">${t(labels[entry.status] || entry.status)}</span>${entry.output && entry.status === 'completed' ? `<button class="btn" data-batch-output="${esc(entry.output)}">${t('打开文件夹')}</button>` : ''}</div>`).join('')}${!entries.length && !empty ? (task.result?.outputs || []).map(path => `<div class="row"><span class="grow">${esc(path)}</span><button class="btn" data-batch-output="${esc(path)}">${t('打开文件夹')}</button></div>`).join('') : ''}`;
  }).join('') || `<p class="muted">${t('还没有历史记录')}</p>`;
}
