import { state, esc, t, taskHTML } from './session.js';

const labels = {
  queued: '等待中', running: '正在处理…', completed: '已完成', failed: '失败',
  cancelled: '已暂停', interrupted: '已中断', awaiting_reference: '等待音色参考',
  partial: '部分完成',
};
const contentLabels = {
  both: '完全配音版 + 双语版', replacement: '完全配音版', dubbing: '双语版',
  subtitles: '只要字幕文件', source_subtitles: '原声 + 字幕，不配音',
};
// Yellow, between the green "completed" and the red "failed"; kept here so the
// shared stylesheet does not need a rule for one status.
const PARTIAL_BADGE = 'color:#8a5a00;background:#fff4d6';
const PARTIAL_TEXT = 'color:#8a5a00';

function statusOf(plan, active) {
  const recent = [...state.tasks].filter(task => task.kind === 'batch')
    .sort((a,b) => b.started_at - a.started_at);
  for (const task of recent) {
    const entry = task.plans?.[plan.plan_id] || task.result?.outcomes?.[plan.plan_id];
    if (entry) {
      if (task.reference?.plan_id === plan.plan_id) return {...entry, status:'awaiting_reference'};
      return entry;
    }
    // A single-work retry lists only that work; the others keep their earlier outcome.
    if (task === active && !Object.keys(task.plans || {}).length) return {status:'queued'};
    if (task.status === 'completed' && task.result?.plans?.includes(plan.plan_id))
      return {status:'completed'};
  }
  return {status:'queued'};
}

const badge = status => `<span class="batch-status ${status}"${status === 'partial' ? ` style="${PARTIAL_BADGE}"` : ''}>${t(labels[status] || status)}</span>`;

// What the sentences that were left out mean for the finished work.
export function warningsHTML(entry) {
  if (!entry.warnings?.length) return '';
  return entry.warnings.map(warning => {
    const head = t(warning.stage === 'translate'
      ? '翻译：{count} 句没有取得译文（共 {total} 句），这些句子只有原文字幕、不配音。'
      : '配音：{count} 句没有配音（共 {total} 句），成品里这些句子保留原声。',
    {count: warning.count, total: warning.total});
    const rows = (warning.items || []).map(item => `<li>${esc(item.id)}${item.reason ? `：${esc(item.reason)}` : ''}</li>`).join('');
    const track = warning.track ? `${esc(warning.track)} · ` : '';
    return `<div class="d batch-warning" style="${PARTIAL_TEXT}">${track}${esc(head)}<details><summary>${t('查看这些句子')}</summary><ul>${rows}</ul></details></div>`;
  }).join('');
}

// The four steps a work goes through; the engine names the command it is running.
const STAGE_GROUPS = [
  ['识别', ['create', 'analyze']],
  ['翻译', ['translate']],
  ['配音', ['synthesize']],
  ['导出', ['mix', 'subtitles', 'export']],
];
const RETRYABLE = ['failed', 'partial', 'cancelled', 'interrupted'];

export function stageBarHTML(entry) {
  const status = entry.status || 'queued';
  if (status === 'queued' || status === 'awaiting_reference' && !entry.stage) return '';
  const finished = status === 'completed' || status === 'partial';
  const current = STAGE_GROUPS.findIndex(([, names]) => names.includes(entry.stage));
  const cells = STAGE_GROUPS.map(([label], index) => {
    let mark = 'todo';
    if (finished || index < current) mark = 'done';
    else if (index === current)
      mark = status === 'failed' ? 'failed' : status === 'running' ? 'running' : 'paused';
    return `<span class="stage ${mark}">${esc(t(label))}</span>`;
  });
  return `<div class="stagebar" role="img" aria-label="${esc(t('处理进度'))}">${cells.join('')}</div>`;
}

const retryButton = (id, status, active) => !active && RETRYABLE.includes(status)
  ? `<button class="btn" data-retry-plan="${esc(id)}">${t('重试这个作品')}</button>` : '';

const diagnosticsButton =(id, status) => ['failed', 'partial'].includes(status)
  ? `<button class="btn plain" data-batch-diagnostics="${esc(id)}">${t('导出诊断包')}</button>` : '';

export function renderQueue(active) {
  if (!state.queue.length) return `<div class="card empty"><b>${t('队列是空的')}</b><p>${t('添加一个已解压的作品文件夹开始。')}</p></div>`;
  const entries = state.queue.map(plan => statusOf(plan, active));
  const count = status => entries.filter(e => e.status === status).length;
  const values = {
    done:count('completed'), total:entries.length, failed:count('failed'), waiting:count('queued'),
    partial:count('partial'),
  };
  const summary = values.partial
    ? t('已完成 {done}/{total} · 部分完成 {partial} · 失败 {failed} · 等待 {waiting}', values)
    : t('已完成 {done}/{total} · 失败 {failed} · 等待 {waiting}', values);
  return `<div class="label"><b>${t('队列')}</b><span>${t('按顺序处理，可拖动调整')}</span></div><p class="batch-summary" role="status">${esc(summary)}</p><div class="card q">${state.queue.map((plan,index) => {
    const entry = entries[index], status = entry.status || 'queued';
    return `<div class="row" draggable="${!active}" data-plan-row="${esc(plan.plan_id)}" data-plan-status="${status}"><div class="grow"><div class="t">${esc(plan.folder.split(/[\\/]/).pop())}</div><div class="d">${t('{count} 条音轨',{count:plan.sources.length})} · ${esc(t(contentLabels[plan.edition.output_policy?.content] || '双语版'))} · ${esc(t(plan.layout))}</div>${stageBarHTML(entry)}${entry.error ? `<div class="d batch-error">${esc(t(entry.error))}</div>` : ''}${warningsHTML(entry)}</div>${badge(status)}${status === 'completed' || status === 'partial' ? `<button class="btn" data-batch-output="${esc(plan.output_root)}">${t('打开文件夹')}</button>` : ''}${active ? '' : `${retryButton(plan.plan_id, status, active)}${diagnosticsButton(plan.plan_id, status)}<button class="btn" data-edit-plan="${esc(plan.plan_id)}">${t('编辑')}</button><button class="btn plain" data-remove-plan="${esc(plan.plan_id)}">${t('删除')}</button><button class="btn plain" data-plan-up="${index}"${index===0 ? ' disabled':''}>↑</button>`}</div>`;
  }).join('')}</div>${active ? taskHTML(active) : ''}`;
}

export function renderHistory(history) {
  return [...history].sort((a,b) => b.started_at - a.started_at).map(task => {
    const empty = /队列还是空的|队列没有作品/.test(task.error || '');
    const date = new Date(task.started_at * 1000).toLocaleString();
    const entries = Object.entries(task.plans || task.result?.outcomes || {});
    const body = empty
      ? `<div class="card empty">${t('这次运行没有可执行作品，未处理任何文件。')}</div>`
      : taskHTML(task);
    return `<div class="label"><b>${esc(date)}</b><button class="btn plain small" data-dismiss-batch="${esc(task.id)}">${t('隐藏记录')}</button></div>${body}${empty ? '' : entries.map(([id, entry]) => `<div class="row"><span class="grow">${esc(entry.work || '')}${entry.error ? `<small class="batch-error">${esc(t(entry.error))}</small>` : ''}${warningsHTML(entry)}</span>${badge(entry.status)}${entry.output && ['completed', 'partial'].includes(entry.status) ? `<button class="btn" data-batch-output="${esc(entry.output)}">${t('打开文件夹')}</button>` : ''}${diagnosticsButton(id, entry.status)}</div>`).join('')}${!entries.length && !empty ? (task.result?.outputs || []).map(path => `<div class="row"><span class="grow">${esc(path)}</span><button class="btn" data-batch-output="${esc(path)}">${t('打开文件夹')}</button></div>`).join('') : ''}`;
  }).join('') || `<p class="muted">${t('还没有历史记录')}</p>`;
}
