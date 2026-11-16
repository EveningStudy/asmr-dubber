import { api, esc, t, dialog, startTask, notice } from './session.js';

const marks = { ok: '✓', warn: '!', fail: '✗', skip: '–' };
const colors = { ok: '#16733d', warn: '#8a5a00', fail: '#b3261e', skip: 'inherit' };

function checksHTML(checks) {
  return `<div class="card">${checks
    .filter((check) => check.status !== 'skip')
    .map(
      (check) =>
        `<div class="row" data-preflight="${esc(check.status)}"><span style="color:${colors[check.status]};font-weight:700;width:1.5em">${marks[check.status]}</span><div class="grow"><div class="t">${esc(t(check.title))}</div><div class="d">${esc(t(check.message))}</div></div></div>`,
    )
    .join('')}</div>`;
}

// Starts the queue only when the checks pass; a warning asks first, a failure explains.
export async function startBatchChecked() {
  const result = await api('batch/preflight');
  if (!result.ok) {
    dialog('开始前检查未通过', `<p>${t('下面标 ✗ 的问题需要先解决，这次没有处理任何文件。')}</p>${checksHTML(result.checks)}`, '', null);
    return false;
  }
  if (result.checks.some((check) => check.status === 'warn')) {
    dialog('开始前提醒', `<p>${t('下面标 ! 的情况不会阻止处理，但可能让它更慢或更紧张。')}</p>${checksHTML(result.checks)}`, '仍然开始', () =>
      startTask('batch', {}, false),
    );
    return false;
  }
  await startTask('batch', {}, false);
  return true;
}

export async function exportDiagnostics(identifier = '') {
  const result = await api('batch/diagnostics', { identifier });
  const link = document.createElement('a');
  link.href = result.url;
  link.download = 'ASMR-Dubber-diagnostics.zip';
  document.body.append(link);
  link.click();
  link.remove();
  notice('诊断包已生成，不含 API Key；日志里有本机路径，发给别人前看一眼。');
}
