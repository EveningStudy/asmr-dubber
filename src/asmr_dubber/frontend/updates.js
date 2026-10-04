import { $, state, api, esc, t, dialog, startTask, guard } from './session.js';

let release = null;

export function renderUpdate() {
  const box = $('#updateFoot');
  const running = state.tasks.find(
    (task) => task.kind === 'update' && ['queued', 'running', 'cancelling'].includes(task.status),
  );
  if (running) {
    const percent = running.total > 0 ? Math.round((running.current / running.total) * 100) : 0;
    box.textContent = `${t('正在下载新版本')} ${percent}%`;
  } else if (!release) {
    box.textContent = '';
  } else if (release.error) {
    box.innerHTML = `${t('检查更新失败')} · <a href="${esc(release.page)}" target="_blank" rel="noopener">${t('手动下载')}</a>`;
  } else if (release.newer) {
    box.innerHTML = `<button class="btn plain small" id="updateButton">${esc(t('有新版本 {version}', { version: release.latest }))}</button>`;
  } else {
    box.textContent = t('已是最新版本');
  }
}

export async function refreshUpdate() {
  try {
    release = await api('updates/check');
  } catch {
    release = null;
  }
  renderUpdate();
}

export function updateFinished(task) {
  dialog('更新完成', `<p>${esc(t(task.result.message))}</p>`, '', null);
}

function manualLink() {
  return `<p class="stat">${t('下载不动的话，可以')} <a href="${esc(release.page)}" target="_blank" rel="noopener">${t('手动下载')}</a>${t('，解压后覆盖到程序文件夹。')}</p>`;
}

export function initializeUpdates() {
  document.addEventListener(
    'click',
    guard(async (event) => {
      if (!event.target.closest('#updateButton')) return;
      if (!release.automatic) {
        dialog(
          t('有新版本 {version}', { version: release.latest }),
          `<p>${t('这个安装方式不支持自动更新。')}</p>${manualLink()}`,
          '',
          null,
        );
        return;
      }
      dialog(
        t('有新版本 {version}', { version: release.latest }),
        `<p>${t('下载并安装新版本。项目、模型和设置不会受影响。更新期间不要关闭启动窗口。')}</p>${manualLink()}`,
        '下载并安装',
        async () => {
          await startTask('update', {}, false);
        },
      );
    }),
  );
  // The check goes out to GitHub and must never hold up the interface.
  refreshUpdate();
  window.addEventListener('settings-changed', renderUpdate);
}
