import {
  $,
  $$,
  state,
  api,
  esc,
  t,
  options,
  field,
  dialog,
  taskHTML,
  startTask,
  guard,
  notice,
  chooseFile,
  upload,
} from './session.js';
import { referenceDialog } from './project_dialogs.js';
import { openProject } from './projects.js';
import { renderQueue, renderHistory } from './batch_view.js';
import { startBatchChecked, exportDiagnostics } from './batch_preflight.js';
import { requestNotifyPermission } from './notify.js';

let scan = null,
  editing = '',
  selected = new Set(),
  refCountdownTimer = null,
  lastRefKey = '';

function updateRefCountdown(deadlineEpoch) {
  const node = $('#batchRefCountdown');
  if (!node || !deadlineEpoch) return;
  node.textContent = String(Math.max(0, Math.ceil(deadlineEpoch - Date.now() / 1000)));
}
export async function refreshBatch() {
  state.queue = await api('batch/list');
  renderBatch();
  await refreshReference();
}
export async function refreshReference() {
  const value = await api('batch/reference');
  $('#queueReferenceStatus').textContent = value.audio
    ? t('已设置整队外部参考（{count} 个作品）', {count: value.plans.length}) : t('未设置，按作品选择参考');
  $('#clearQueueReference').hidden = !value.audio;
}
async function queueReferenceDialog() {
  const value = await api('batch/reference');
  let external = value.audio ? {path: value.audio, url: value.url} : null;
  dialog('整队共用外部参考', `<p>${t('将用于队列里所有等待作品；情绪仍跟随各句。')}</p><button class="btn" id="pickQueueReference">${t('选择文件')}</button><span id="queueReferenceFile"></span><audio id="queueReferenceAudio" controls style="width:100%"></audio><label>${t('参考文字')}<textarea id="queueReferenceText">${esc(value.text || '')}</textarea></label>${field('外部音频语言', `<select id="queueReferenceLanguage">${options([['跟随项目','auto'],['日语','ja'],['英语','en'],['中文','zh']], value.language || 'auto')}</select>`)}`, '采用', async () => {
    if (!external) throw new Error(t('请选择外部音频。'));
    await api('batch/reference', {external: external.path, text: $('#queueReferenceText').value, language: $('#queueReferenceLanguage').value});
    await refreshReference();
  });
  function preview() {
    $('#queueReferenceFile').textContent = external?.path.split(/[\\/]/).pop() || '';
    $('#queueReferenceAudio').src = external?.url || '';
  }
  $('#pickQueueReference').onclick = guard(async () => {
    const file = await chooseFile('audio/*');
    if (file) { external = await upload(file); preview(); }
  });
  preview();
}
export function renderBatch() {
  const active = state.tasks.find(
    (task) => task.kind === 'batch' && ['queued', 'running', 'cancelling'].includes(task.status),
  );
  const history = state.tasks.filter(
    (task) => task.kind === 'batch' && !task.hidden && !['queued', 'running', 'cancelling'].includes(task.status),
  );
  $('#queueBadge').textContent = state.queue.length || '';
  $('#batchQueue').innerHTML = renderQueue(active);
  $('#runBatch').disabled = !!active || !state.queue.length;
  $('#setQueueReference').disabled = !!active || !state.queue.length;
  const request = active?.reference;
  const refKey = request
    ? `${request.request_id || ''}:${request.project_json || ''}:${request.deadline_epoch || ''}`
    : '';
  if (request) {
    if (refKey !== lastRefKey || !$('#batchRefCountdown')) {
      const secs = request.deadline_epoch
        ? Math.max(0, Math.ceil(request.deadline_epoch - Date.now() / 1000))
        : '';
      $('#batchReference').innerHTML = `<div class="card"><div class="row"><div class="grow"><div class="t">${esc(request.work || '')} · ${t('音色参考')}</div><div class="d">${t('等待时间结束后使用自动推荐的项目片段。')} <span id="batchRefCountdown">${secs}</span> ${t('秒')}</div></div><button class="btn" id="chooseBatchReference">${t('试听并选择')}</button></div></div>`;
      $('#chooseBatchReference').onclick = guard(() => referenceDialog(request.project_json, true));
      lastRefKey = refKey;
    } else {
      updateRefCountdown(request.deadline_epoch);
    }
    if (request.deadline_epoch && !refCountdownTimer) {
      refCountdownTimer = setInterval(() => {
        const currentActive = state.tasks.find(
          (task) => task.kind === 'batch' && ['queued', 'running', 'cancelling'].includes(task.status),
        );
        const currentReq = currentActive?.reference;
        if (!currentReq?.deadline_epoch) {
          clearInterval(refCountdownTimer);
          refCountdownTimer = null;
          return;
        }
        updateRefCountdown(currentReq.deadline_epoch);
      }, 1000);
    }
  } else {
    lastRefKey = '';
    if (refCountdownTimer) {
      clearInterval(refCountdownTimer);
      refCountdownTimer = null;
    }
    $('#batchReference').innerHTML = `<p class="muted">${t('没有需要确认的任务')}</p>`;
  }
  $('#batchHistory').innerHTML = renderHistory(history);
}
function syncTaskFieldVisibility() {
  const content = $('#batchContent')?.value;
  const isSubtitlesOnly = content === 'subtitles';
  const modeNode = $('#batchMode');
  if (modeNode) {
    modeNode.disabled = isSubtitlesOnly;
    if (isSubtitlesOnly) modeNode.value = 'audio';
  }
  const isAudio = isSubtitlesOnly || modeNode?.value === 'audio';
  for (const id of ['batchBackground', 'batchEmbed']) {
    const node = $(`#${id}`);
    if (!node) continue;
    node.disabled = isAudio;
    const row = node.closest('.field');
    if (row) row.hidden = isAudio;
  }
  const cover = $('#batchCover');
  if (cover) cover.hidden = isAudio;
}
function taskFields() {
  const defaults = state.boot.settings;
  return `<div class="card group">${field(
    '成品',
    `<select id="batchContent">${options(
      [
        ['双语版', 'dubbing'],
        ['完全配音版', 'replacement'],
        ['完全配音版 + 双语版', 'both'],
        ['原声 + 字幕，不配音', 'source_subtitles'],
        ['只要字幕文件', 'subtitles'],
      ],
      defaults.autoflow_default_content,
    )}</select>`,
  )}${field(
    '字幕',
    `<select id="batchSubtitle">${options(
      [
        ['双语', 'bilingual'],
        ['仅译文', 'zh'],
        ['仅原文', 'source'],
      ],
      defaults.autoflow_subtitle_language,
    )}</select>`,
  )}${field(
    '格式',
    `<select id="batchMode">${options(
      [
        ['音频', 'audio'],
        ['视频', 'video_normal'],
        ['和谐视频', 'video_harmonized'],
      ],
      defaults.autoflow_default_mode,
    )}</select>`,
  )}${field('视频画面', `<select id="batchBackground"><option value="black">${t('黑色背景')}</option></select>`)}<div class="cover-preview" id="batchCover"><img id="batchCoverImage" hidden alt="${t('视频画面')}"><span id="batchCoverBlack">${t('黑色背景')}</span></div>${field('把字幕烧进视频', `<input type="checkbox" id="batchEmbed"${defaults.autoflow_embed_subtitles ? ' checked' : ''}>`)}${field(
    '多条音轨',
    `<select id="batchLayout">${options(
      [
        ['合并成一个', 'merged'],
        ['每轨单独', 'separate'],
        ['都要', 'both'],
      ],
      defaults.autoflow_default_layout,
    )}</select>`,
  )}${field('重新处理', `<input type="checkbox" id="batchRebuild">`)}</div>`;
}
async function workDialog(identifier = '') {
  editing = identifier;
  scan = null;
  selected.clear();
  dialog(
    identifier ? '编辑队列任务' : '添加作品',
    `<div class="card group"><div class="field"><input type="text" id="workFolder" style="flex:1;max-width:none"><button class="btn" id="chooseWorkFolder">${t('选择文件夹')}</button><button class="btn" id="scanWork">${t('扫描作品')}</button></div>${field('音频版本', '<select id="workEdition"></select>')}${field('包含特典、样本和 Free Talk', `<input type="checkbox" id="workBonus"${state.boot.settings.autoflow_include_bonus ? ' checked' : ''}>`)}</div><div class="row"><div class="more grow" id="trackSummary"></div><button type="button" class="btn plain small" id="toggleAllTracks" hidden></button></div><div class="card group" id="workTracks"></div><div class="more">${t('做成什么')}</div>${taskFields()}<p class="small muted">${t('识别、翻译、配音按“新项目默认值”来。')}</p>`,
    identifier ? '保存修改' : '加入队列',
    async () => {
      if (!scan) throw new Error(t('请先扫描作品。'));
      const sources = scan.source_payloads.filter((_, index) => selected.has(scan.track_items[index].id));
      await api('batch/save', {
        folder: scan.folder,
        edition: $('#workEdition').value,
        sources,
        mode: $('#batchMode').value,
        layout: $('#batchLayout').value,
        background: $('#batchBackground').value,
        embed_subtitles: $('#batchEmbed').checked,
        rebuild: $('#batchRebuild').checked,
        content: $('#batchContent').value,
        subtitle_language: $('#batchSubtitle').value,
        editing,
      });
      await refreshBatch();
    },
  );
  $('#scanWork').onclick = guard(async () => {
    scan = await api('batch/scan', {
      folder: $('#workFolder').value,
      include_bonus: $('#workBonus').checked,
    });
    selected = new Set(scan.track_items.map((track) => track.id));
    renderScan();
  });
  $('#chooseWorkFolder').onclick = guard(async () => {
    const result = await api('files/folder');
    if (result.path) {
      $('#workFolder').value = result.path;
      $('#scanWork').click();
    }
  });
  $('#workEdition').onchange = $('#workBonus').onchange = guard(async () => {
    if (!scan) return;
    const view = await api('batch/edition', {
      folder: scan.folder,
      edition: $('#workEdition').value,
      include_bonus: $('#workBonus').checked,
    });
    Object.assign(scan, view);
    selected = new Set(scan.track_items.map((track) => track.id));
    renderTracks();
  });
  const backgroundSelect = $('#batchBackground');
  let coverGeneration = 0;
  backgroundSelect.onchange = guard(async () => {
    if (!scan) return;
    const generation = ++coverGeneration;
    const result = await api('batch/background', {folder: scan.folder, background: backgroundSelect.value});
    if (generation === coverGeneration && backgroundSelect.isConnected) renderCover(result.preview);
  });
  $('#batchContent').onchange = $('#batchMode').onchange = syncTaskFieldVisibility;
  $('#toggleAllTracks').onclick = () => {
    if (!scan?.track_items?.length) return;
    if (selected.size === scan.track_items.length) selected.clear();
    else selected = new Set(scan.track_items.map((track) => track.id));
    renderTracks();
  };
  syncTaskFieldVisibility();
  if (identifier) {
    scan = await api('batch/edit', { identifier });
    selected = new Set(scan.track_items.map((track) => track.id));
    renderScan();
    $('#workBonus').checked = scan.include_bonus;
    $('#batchMode').value = scan.mode;
    $('#batchLayout').value = scan.layout;
    $('#batchEmbed').checked = scan.embed_subtitles;
    $('#batchRebuild').checked = scan.rebuild;
    $('#batchSubtitle').value = scan.subtitle_language;
    const plan = state.queue.find((plan) => plan.plan_id === identifier);
    $('#batchContent').value =
      plan.edition.output_policy?.content ||
      (scan.source_subtitles_only ? 'subtitles' : scan.subtitles_only ? 'source_subtitles' : 'dubbing');
    syncTaskFieldVisibility();
  }
}
function renderScan() {
  $('#workFolder').value = scan.folder;
  $('#workEdition').innerHTML = options(scan.edition_choices, scan.selected_edition);
  $('#batchBackground').innerHTML = options(scan.background_choices, scan.selected_background);
  renderCover(scan.preview);
  renderTracks();
}
function renderCover(url) {
  $('#batchCoverImage').hidden = !url;
  $('#batchCoverBlack').hidden = !!url;
  if (url) $('#batchCoverImage').src = url;
  else $('#batchCoverImage').removeAttribute('src');
}
function renderTracks() {
  $('#trackSummary').textContent = t('音轨 · 找到 {total} 条，已选 {selected} 条', {
    total: scan.track_items.length,
    selected: selected.size,
  });
  const toggleBtn = $('#toggleAllTracks');
  if (toggleBtn) {
    toggleBtn.hidden = !scan.track_items.length;
    toggleBtn.textContent = t(selected.size === scan.track_items.length ? '取消全选' : '全选');
  }
  $('#workTracks').innerHTML = scan.track_items
    .map(
      (track, index) =>
        `<div class="track${selected.has(track.id) ? '' : ' dim'}" data-track="${esc(track.id)}"><input type="checkbox" data-track-enabled="${esc(track.id)}"${selected.has(track.id) ? ' checked' : ''}><span>${esc(track.path)}<small class="muted">${esc(t(track.category))}</small></span><button class="btn plain" data-track-up="${index}"${index === 0 ? ' disabled' : ''}>↑</button><button class="btn plain" data-track-down="${index}"${index === scan.track_items.length - 1 ? ' disabled' : ''}>↓</button><div><select data-track-subtitle="${esc(track.id)}">${options(
          track.transcript_choices.map((c) => [c.label, c.value]),
          track.transcript_choices.find((c) => c.selected)?.value || '',
        )}</select><div class="track-extra"><select data-track-language="${esc(track.id)}">${options(
          track.language_choices.map((c) => [c.label, c.value]),
          track.language_choices.find((c) => c.selected)?.value,
        )}</select><select data-track-mode="${esc(track.id)}">${options(
          track.timing_choices.filter((c) => !c.disabled).map((c) => [c.label, c.value]),
          track.timing_choices.find((c) => c.selected)?.value,
        )}</select></div></div></div>`,
    )
    .join('');
}
export function initializeBatch() {
  $('#setQueueReference').onclick = guard(queueReferenceDialog);
  $('#clearQueueReference').onclick = guard(async () => {
    await api('batch/reference', {clear: true});
    await refreshReference();
  });
  guard(refreshReference)();
  window.addEventListener('batch-reference-changed', guard(refreshReference));
  window.addEventListener('settings-changed', guard(refreshReference));
  $('#addWork').onclick = guard(() => workDialog());
  $('#runBatch').onclick = guard(async () => {
    requestNotifyPermission();
    $('#runBatch').disabled = true;
    try {
      await startBatchChecked();
    } finally {
      renderBatch();
    }
  });
  document.addEventListener(
    'click',
    guard(async (event) => {
      const diagnostics = event.target.closest('[data-batch-diagnostics]');
      if (diagnostics) {
        await exportDiagnostics(diagnostics.dataset.batchDiagnostics);
        return;
      }
      const retry = event.target.closest('[data-retry-plan]');
      if (retry) {
        // Only this work runs again; its finished stages are kept by the engine.
        requestNotifyPermission();
        await startTask('batch', { plan: retry.dataset.retryPlan }, false);
        return;
      }
      const hide = event.target.closest('[data-dismiss-batch]');
      if (hide) {
        await api('tasks/dismiss', {identifier: hide.dataset.dismissBatch});
        state.tasks = await api('tasks/list');
        renderBatch();
        return;
      }
      const edit = event.target.closest('[data-edit-plan]');
      if (edit) {
        await workDialog(edit.dataset.editPlan);
        return;
      }
      const remove = event.target.closest('[data-remove-plan]');
      if (remove) {
        await api('batch/remove', { identifier: remove.dataset.removePlan });
        await refreshBatch();
        return;
      }
      const up = event.target.closest('[data-plan-up]');
      if (up) {
        const order = state.queue.map((plan) => plan.plan_id),
          index = Number(up.dataset.planUp);
        [order[index - 1], order[index]] = [order[index], order[index - 1]];
        await api('batch/reorder', { order });
        await refreshBatch();
        return;
      }
      const openProj = event.target.closest('[data-open-batch-project]');
      if (openProj) {
        await openProject(openProj.dataset.openBatchProject);
        return;
      }
      const trackMove = event.target.closest('[data-track-up],[data-track-down]');
      if (trackMove) {
        const order = scan.track_items.map((track) => track.id),
          isUp = trackMove.hasAttribute('data-track-up'),
          index = Number(isUp ? trackMove.dataset.trackUp : trackMove.dataset.trackDown),
          swapWith = isUp ? index - 1 : index + 1;
        if (swapWith >= 0 && swapWith < order.length) {
          [order[swapWith], order[index]] = [order[index], order[swapWith]];
          Object.assign(
            scan,
            await api('batch/tracks', { folder: scan.folder, sources: scan.source_payloads, order }),
          );
          renderTracks();
        }
        return;
      }
      const output = event.target.closest('[data-batch-output]');
      if (output) await api('batch/open', { path: output.dataset.batchOutput });
    }),
  );
  document.addEventListener(
    'change',
    guard(async (event) => {
      const node = event.target;
      if (node.dataset.trackEnabled) {
        node.checked ? selected.add(node.dataset.trackEnabled) : selected.delete(node.dataset.trackEnabled);
        renderTracks();
        return;
      }
      const id = node.dataset.trackSubtitle || node.dataset.trackLanguage || node.dataset.trackMode;
      if (!id) return;
      const row = node.closest('[data-track]');
      const result = await api('batch/subtitle', {
        folder: scan.folder,
        sources: scan.source_payloads,
        track: id,
        file: $('[data-track-subtitle]', row).value,
        language: $('[data-track-language]', row).value,
        mode: $('[data-track-mode]', row).value,
      });
      Object.assign(scan, result);
      renderTracks();
    }),
  );
  let dragged = '';
  $('#batchQueue').ondragstart = (event) => {
    dragged = event.target.closest('[data-plan-row]')?.dataset.planRow || '';
  };
  $('#batchQueue').ondragover = (event) => event.preventDefault();
  $('#batchQueue').ondrop = guard(async (event) => {
    event.preventDefault();
    const target = event.target.closest('[data-plan-row]')?.dataset.planRow;
    if (!target || !dragged) return;
    const order = state.queue.map((plan) => plan.plan_id);
    order.splice(order.indexOf(dragged), 1);
    order.splice(order.indexOf(target), 0, dragged);
    await api('batch/reorder', { order });
    await refreshBatch();
  });
}
