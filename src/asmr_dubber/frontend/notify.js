import { t } from './session.js';

const supported = () => typeof window !== 'undefined' && 'Notification' in window;

// Ask once, from a click (browsers refuse otherwise).  A refusal is final: no prompt, no nag.
export function requestNotifyPermission() {
  if (!supported() || Notification.permission !== 'default') return;
  try {
    const pending = Notification.requestPermission();
    if (pending && pending.catch) pending.catch(() => {});
  } catch {
    // Some browsers only accept a callback form; the permission simply stays undecided.
  }
}

export function batchSummary(task) {
  const entries = Object.values(task.plans || task.result?.outcomes || {});
  const count = (status) => entries.filter((entry) => entry.status === status).length;
  return {
    done: count('completed'),
    partial: count('partial'),
    failed: count('failed'),
    total: entries.length,
  };
}

export function notifyBatchFinished(task) {
  if (!supported() || Notification.permission !== 'granted') return;
  const values = batchSummary(task);
  const body = values.total
    ? t('已完成 {done} · 部分完成 {partial} · 失败 {failed}（共 {total} 个作品）', values)
    : t(task.status === 'completed' ? '队列处理完成。' : '队列有任务失败，请检查日志后重试。');
  try {
    new Notification(t(task.status === 'failed' || values.failed ? '批量处理结束，有失败' : '批量处理完成'), {
      body,
      tag: `batch-${task.id}`,
    });
  } catch {
    // A notification the system declines is not worth interrupting the app for.
  }
}
