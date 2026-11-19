import { renderTranslationPanels, initializeRecovery, initializeTranslateAll } from './project_translation.js';
import {
  $, $$, state, api, esc, t, time, options, field, dialog,
  upload, enqueueSave, flushSaves, startTask, guard, notice, projectBusy,
} from './session.js';
import { renderForms } from './forms.js';
import { matchesRowFilter } from './project_revise.js';
import { transcriptDialog, referenceDialog, reviewDialog } from './project_dialogs.js';
import { initializeReview } from './project_review.js';
import { initializeCompare, renderCompare } from './compare_player.js';
import {
  resetEditorHistory,
  recordFieldUndo,
  recordListUndo,
  syncUndoButtons,
  undoEdit,
  redoEdit,
  playSourceSlice,
  playDubClip,
  checkSourceSliceStop,
  clearSourceSliceStop,
  renderVirtualRows,
  scrollToSentenceIndex,
  renderSentenceControls,
  initializeSentenceEditor,
} from './sentence_editor.js';

const BUSY_CONTROLS = [
  '[data-task],#translateAll,#exportProject,#reviseTranslations,#restoreBackup,#restoreTranslations',
  '#compressTooLong,#disableFailed,#importTranscript,#changeReference,#addSentence,#deleteSentence',
  '#undoSentence,#redoSentence,#sentenceReference,#sentenceControls [data-detail],#rows input[type="checkbox"],#rows select',
  '#rows [data-candidates],#rows [data-take],#rows [data-regenerate],#rows [data-retranslate]',
  '#rows [data-compress],#rows [data-keep-original],#revisionProposals [data-proposal],#revisionProposals [data-proposal-all]',
].join(',');
const filteredSentences = () => (state.project ? state.project.sentences.filter(matchesRowFilter) : []);
let selected = '', importAfterCreate = null, pendingSource = null;

export function syncProjectControls() {
  if (!state.project) return;
  const busy = projectBusy();
  $$(BUSY_CONTROLS).forEach((node) => { node.disabled = busy; });
  $$('#rows [data-row-field="source_text"],#rows [data-row-field="zh_text"]').forEach((cell) => {
    cell.setAttribute('contenteditable', String(!busy));
    cell.contentEditable = String(!busy);
  });
  syncUndoButtons();
}
export async function refreshHome() {
  const projects = await api('projects/list');
  $('#firstSteps').hidden = projects.length > 0;
  $('#projectsRoot').textContent = state.boot.settings.projects_root;
  $('#recentProjects').innerHTML =
    projects
      .map((project) =>
        project.error
          ? `<div class="row muted"><div class="grow"><div class="t">${esc(project.label)}</div><div class="d">${t('项目文件无法读取')} · ${esc(project.error)}</div></div></div>`
          : `<button class="row project-row" data-open="${esc(project.manifest)}"><div class="grow"><div class="t">${esc(project.label)}</div><div class="d">${t(project.source_language)} · ${time(project.duration)}</div></div><div class="p4"><i class="${project.sentences ? 'd' : ''}"></i><i class="${project.translated ? 'd' : ''}"></i><i class="${project.synthesized ? 'd' : ''}"></i><i class="${project.exported ? 'd' : ''}"></i></div><div class="stage">${project.exported ? t('已导出') : t('配音 {done} / {total} 句', { done: project.synthesized, total: project.sentences })}</div><div class="when">${esc(project.updated_at.slice(0, 10))}</div></button>`,
      )
      .join('') || `<div class="row muted">${t('还没有项目')}</div>`;
}
export async function openProject(manifest) {
  await flushSaves({ ignoreError: true });
  state.project = await api('projects/get', { project: manifest });
  resetEditorHistory();
  $('#saveState').textContent = t('修改会自动保存');
  state.filter = 'all';
  const box = $('.tablebox');
  if (box) box.scrollTop = 0;
  selected = state.project.sentences[0]?.id || '';
  $('#sourcePlayer').src = state.project.source_url;
  $('#seek').max = state.project.source.duration_seconds;
  window.dispatchEvent(new CustomEvent('navigate', { detail: { page: 'project' } }));
  renderProject();
}
export function renderProject() {
  if (!state.project) return;
  const project = state.project,
    enabled = project.sentences.filter((sentence) => sentence.enabled);
  const hasTr = (s) => Boolean(s.target_text ?? s.zh_text);
  const translated = enabled.filter(hasTr).length,
    voiced = enabled.filter((sentence) => sentence.tts_file).length;
  $('#projectName').textContent = project.title;
  $('#projectMeta').textContent =
    `${t(project.source_language)} → ${t(project.settings.tts_target_language)} · ${time(project.source.duration_seconds)} · ${t('{count} 句', { count: project.sentences.length })}`;
  $('#asrStat').textContent = t('已识别 {count} 句。重新识别会替换现有句子，译文和配音要重做。', {
    count: project.sentences.length,
  });
  $('#translateRemaining').textContent = t('翻译剩余 {count} 句', { count: enabled.length - translated });
  renderTranslationPanels(project);
  $('#translationStat').textContent = t('已翻译 {done} / {total} 句。', {
    done: translated,
    total: enabled.length,
  });
  $('#synthesizeRemaining').textContent = t('生成剩余 {count} 句', {
    count: enabled.filter((s) => hasTr(s) && !s.tts_file).length,
  });
  $('#dubStat').textContent = t('已完成 {done} / {total} 句。只生成还没有配音、或译文改过的句子。', {
    done: voiced,
    total: enabled.length,
  });
  $('#dubBar').style.width = `${enabled.length ? (voiced / enabled.length) * 100 : 0}%`;
  const undubbed = enabled.filter((s) => hasTr(s) && !s.tts_file).length;
  $('#exportStat').textContent = undubbed
    ? t('还有 {count} 句没有配音，成品里这些句子只有原声。', { count: undubbed })
    : t('所有句子都已配音。');
  $('#referenceLabel').textContent =
    `${t('音色参考')} · ${project.reference_external ? t('外部音频') : project.reference_selected || t('未选择')}`;
  $('#projectDiagnostics').textContent = t(project.diagnostics);
  $('#outputs').innerHTML = Object.entries(project.outputs)
    .map(
      ([key, item]) =>
        `<div class="out"><span class="grow">${esc(item.name)}</span>${/audio|stem|output_file/.test(key) ? `<audio controls preload="none" src="${esc(item.url)}"></audio>` : ''}<a class="btn plain small" href="${esc(item.url)}" download="${esc(item.name)}">${t('下载')}</a></div>`,
    )
    .join('');
  $$('#steps button').forEach((button, index) =>
    button.classList.toggle(
      'done',
      [
        project.sentences.length > 0,
        translated === enabled.length && enabled.length > 0,
        voiced === enabled.length && enabled.length > 0,
        Object.keys(project.outputs).length > 0,
      ][index],
    ),
  );
  renderRows();
  renderForms();
  renderCompare(project);
  syncProjectControls();
}
function renderRows() {
  if (!state.project) return;
  const sentences = filteredSentences();
  renderVirtualRows(sentences, selected);
  renderSentenceControls(selected, sentences.length, syncProjectControls);
}
function saveRows(only = null) {
  const rows = state.project.sentences.map((s) => [
    s.id, s.enabled, s.start_seconds, s.end_seconds, s.source_text, s.target_text ?? s.zh_text ?? '',
    s.original_audio_enabled === null ? 'default' : s.original_audio_enabled ? 'on' : 'off',
    s.original_audio_gain_db, s.chinese_audio_enabled, s.chinese_audio_gain_db, s.speaker_id || '',
  ]);
  $('#saveState').textContent = t('正在保存…');
  return enqueueSave(async () => {
    const result = await api(only ? 'projects/sentence' : 'projects/table', {
      project: state.project.manifest,
      ...(only ? { row: rows.find((row) => row[0] === only) } : { rows }),
      revision: state.project.revision,
    });
    state.project.revision = result.revision;
    state.project.settings = result.settings;
    state.project.outputs = result.outputs;
  });
}
function openCreateDialog(sourcePath, displayName) {
  pendingSource = { path: sourcePath, name: displayName };
  const defaults = state.boot.settings;
  dialog(
    '新建项目',
    `<div class="card group">${field(
      '音频里说的是',
      `<select id="newLanguage">${options(
        [['日语', 'ja'], ['英语', 'en'], ['中文', 'zh']],
        defaults.default_source_language,
      )}</select>`,
    )}${field(
      '配音成',
      `<select id="newTarget">${options(
        [['中文', 'zh'], ['英文', 'en']],
        defaults.tts_target_language,
      )}</select>`,
    )}</div><div class="more">${t('文字从哪来')}</div><div class="card group">${[
      ['自动识别', '用识别模型听出原文', 'auto'],
      ['我有原文字幕', '跳过识别，直接翻译', 'source'],
      ['我有译文字幕', '跳过识别和翻译，直接配音', 'zh'],
      ['我只有不带时间的台本', '先识别出时间，再把台本文字对上去', 'script'],
    ]
      .map(
        ([title, hint, value]) =>
          `<label class="choice"><input type="radio" name="textSource" value="${value}"${value === 'auto' ? ' checked' : ''}><div>${t(title)}<small>${t(hint)}</small></div></label>`,
      )
      .join('')}</div>`,
    '创建',
    async () => {
      importAfterCreate = $('input[name="textSource"]:checked').value;
      await startTask(
        'create',
        {
          source: pendingSource.path,
          source_language: $('#newLanguage').value,
          target_language: $('#newTarget').value,
        },
        false,
      );
    },
    displayName,
  );
}
async function newSource(file) {
  notice(t('正在导入…'));
  const uploaded = await upload(file);
  openCreateDialog(uploaded.path, file.name);
}
export async function finishCreation(task) {
  await openProject(task.result.manifest);
  if (importAfterCreate && importAfterCreate !== 'auto') {
    const kind = importAfterCreate;
    importAfterCreate = null;
    await transcriptDialog(kind === 'zh' ? 'zh' : 'source', kind === 'script' ? 'script_review' : 'estimate');
  } else importAfterCreate = null;
}
export { reviewDialog };
export function initializeProjects() {
  let mode = 'both';
  window.addEventListener('project-changed', renderProject);
  $('#dropNew').onclick = () => $('#mediaFile').click();
  $('#dropNew').onkeydown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      $('#mediaFile').click();
    }
  };
  const pickBtn = $('#pickLocalMedia');
  if (pickBtn) {
    pickBtn.onclick = guard(async () => {
      const picked = await api('files/media');
      if (!picked.native) {
        $('#mediaFile').click();
        return;
      }
      if (picked.path) openCreateDialog(picked.path, picked.name || picked.path);
    });
  }
  $('#mediaFile').onchange = guard(async (event) => {
    if (event.target.files[0]) await newSource(event.target.files[0]);
    event.target.value = '';
  });
  $('#dropNew').ondragover = (event) => { event.preventDefault(); };
  $('#dropNew').ondrop = guard(async (event) => {
    event.preventDefault();
    if (event.dataTransfer.files[0]) await newSource(event.dataTransfer.files[0]);
  });
  $('#openExisting').onclick = guard(async () => {
    dialog('打开项目', field('项目路径', '<input type="text" id="existingManifest">'), '打开项目', async () =>
      openProject($('#existingManifest').value),
    );
  });
  $('#search').oninput = () => {
    const box = $('.tablebox');
    if (box) box.scrollTop = 0;
    renderRows();
  };
  $('#importTranscript').onclick = guard(() => transcriptDialog());
  $('#changeReference').onclick = guard(() => referenceDialog());
  $('#showReview').onclick = guard(reviewDialog);
  $('#openProjectFolder').onclick = guard(() => api('projects/open', { project: state.project.manifest }));
  initializeTranslateAll(dialog, field, () => selected);
  initializeRecovery(saveRows, renderProject, dialog);
  initializeReview({ save: saveRows, render: renderProject, getSelected: () => selected });
  initializeCompare();
  initializeSentenceEditor({
    filteredSentences,
    getSelected: () => selected,
    setSelected: (id) => { selected = id; },
    saveRows,
    renderProject,
    syncBusy: syncProjectControls,
    getListenMode: () => mode,
  });
  window.addEventListener('locate-sentence', (event) => {
    const index = state.project?.sentences.findIndex((s) => s.id === event.detail);
    if (index === undefined || index < 0) return;
    state.filter = 'all';
    $('#search').value = '';
    $$('[data-filter]').forEach((b) => b.classList.toggle('on', b.dataset.filter === 'all'));
    selected = event.detail;
    scrollToSentenceIndex(index);
    renderRows();
    $(`#rows [data-sentence="${CSS.escape(event.detail)}"]`)?.scrollIntoView({ block: 'center' });
  });
  $('#exportProject').onclick = guard(() =>
    startTask('export', { audio: $('#exportAudio').value === 'true', language: $('#exportSubtitle').value }),
  );
  document.addEventListener(
    'click',
    guard(async (event) => {
      const open = event.target.closest('[data-open]');
      if (open) { await openProject(open.dataset.open); return; }
      const task = event.target.closest('[data-task]');
      if (task) { await startTask(task.dataset.task); return; }
      const review = event.target.closest('[data-review-task]');
      if (review) {
        await startTask(review.dataset.reviewTask, {
          window: $('#reviewWindow').value,
          candidate: $('#reviewCandidate').value,
        });
        $('#dialog').classList.remove('on');
        return;
      }
      const filter = event.target.closest('[data-filter]');
      if (filter) {
        state.filter = filter.dataset.filter;
        const box = $('.tablebox');
        if (box) box.scrollTop = 0;
        $$('[data-filter]').forEach((b) => b.classList.toggle('on', b === filter));
        renderRows();
        return;
      }
      const playSrc = event.target.closest('[data-play-source]');
      if (playSrc) {
        const sentence = state.project.sentences.find((s) => s.id === playSrc.dataset.playSource);
        await playSourceSlice(sentence);
        return;
      }
      const play = event.target.closest('[data-play]');
      if (play) {
        const sentence = state.project.sentences.find((s) => s.id === play.dataset.play);
        await playDubClip(sentence);
        return;
      }
      if (event.target.id === 'undoSentence') { await undoEdit(); return; }
      if (event.target.id === 'redoSentence') { await redoEdit(); return; }
      if (event.target.id === 'sentenceReference') {
        await referenceDialog(state.project.manifest, false, selected);
        return;
      }
      if (event.target.id === 'addSentence') {
        if (projectBusy()) throw new Error(t('请等当前任务结束后再操作。'));
        await flushSaves();
        const beforeList = state.project.sentences;
        const beforeSel = selected;
        const last = state.project.sentences.at(-1);
        const start = last?.end_seconds || 0;
        const end = Math.min(state.project.source.duration_seconds, start + 1);
        if (end <= start)
          throw new Error(t('媒体末尾没有剩余空间；请先调整已有句子的结束时间，再添加句子。'));
        selected = `s${Date.now()}`;
        const nextList = [
          ...beforeList,
          {
            id: selected, start_seconds: start, end_seconds: end, source_text: '',
            target_text: t('新句子'),
            enabled: true, original_audio_enabled: null, original_audio_gain_db: 0,
            chinese_audio_enabled: true, chinese_audio_gain_db: 0,
          },
        ];
        recordListUndo(beforeList, nextList, beforeSel, selected);
        state.project.sentences = nextList;
        await saveRows();
        renderProject();
        return;
      }
      if (event.target.id === 'deleteSentence') {
        if (projectBusy()) throw new Error(t('请等当前任务结束后再操作。'));
        const beforeList = state.project.sentences;
        const beforeSel = selected;
        const nextList = beforeList.filter((s) => s.id !== selected);
        selected = nextList[0]?.id || '';
        recordListUndo(beforeList, nextList, beforeSel, selected);
        state.project.sentences = nextList;
        await saveRows();
        renderProject();
        return;
      }
      const row = event.target.closest('tr[data-sentence]');
      if (row && selected !== row.dataset.sentence) {
        selected = row.dataset.sentence;
        const sentence = state.project.sentences.find((s) => s.id === selected);
        clearSourceSliceStop();
        $('#sourcePlayer').currentTime = sentence.start_seconds;
        $$('#rows tr').forEach((r) => r.classList.toggle('sel', r === row));
        renderSentenceControls(selected, filteredSentences().length, syncProjectControls);
      }
    }),
  );
  document.addEventListener(
    'focusout',
    guard(async (event) => {
      if (projectBusy()) return;
      const key = event.target.dataset.rowField;
      if (!key || event.target.type === 'checkbox') return;
      const tr = event.target.closest('tr');
      if (!tr) return;
      const sentence = state.project.sentences.find((s) => s.id === tr.dataset.sentence);
      if (!sentence) return;
      const nextVal = event.target.textContent;
      if (sentence[key] !== nextVal) {
        recordFieldUndo(sentence.id, { [key]: sentence[key] }, { [key]: nextVal });
        sentence[key] = nextVal;
        if (key === 'zh_text') sentence.target_text = nextVal;
        sentence.tts_file = null;
        sentence.tts_url = null;
        sentence.tts_candidates = [];
        await saveRows(sentence.id);
      }
    }),
  );
  document.addEventListener(
    'change',
    guard(async (event) => {
      if (projectBusy()) return;
      const node = event.target,
        key = node.dataset.detail || node.dataset.rowField;
      if (!key) return;
      const sentence = state.project.sentences.find(
        (s) => s.id === (node.dataset.detail ? selected : node.closest('tr').dataset.sentence),
      );
      if (!sentence) return;
      const prevVal = sentence[key];
      const nextVal =
        node.type === 'checkbox'
          ? node.checked
          : key === 'original_audio_enabled'
            ? { default: null, on: true, off: false }[node.value]
            : key === 'speaker_id'
              ? node.value.trim() || null
              : Number(node.value);
      recordFieldUndo(sentence.id, { [key]: prevVal }, { [key]: nextVal });
      sentence[key] = nextVal;
      await saveRows(sentence.id);
      if (key === 'enabled' || key === 'speaker_id' || key === 'start_seconds' || key === 'end_seconds') {
        state.project = await api('projects/get', { project: state.project.manifest });
        renderProject();
      }
    }),
  );
  const source = $('#sourcePlayer'),
    dub = $('#dubPlayer');
  $('#playSource').onclick = guard(async () => {
    if (!source.paused || !dub.paused) {
      clearSourceSliceStop();
      source.pause();
      dub.pause();
      $('#playSource').textContent = '▶';
      return;
    }
    clearSourceSliceStop();
    if (mode !== 'dub') await source.play();
    if (mode !== 'source') {
      const sentence = state.project.sentences.find((s) => s.id === selected);
      const stem = state.project.outputs.chinese_stem_file;
      if (stem) {
        dub.src = stem.url;
        dub.currentTime = source.currentTime;
      } else if (sentence?.tts_url) {
        dub.src = sentence.tts_url;
      }
      if (dub.src) await dub.play();
    }
    $('#playSource').textContent = 'Ⅱ';
  });
  source.ontimeupdate = () => {
    checkSourceSliceStop(source);
    $('#playerTime').textContent = `${time(source.currentTime)} / ${time(source.duration)}`;
    $('#seek').value = source.currentTime;
  };
  source.onended = () => {
    clearSourceSliceStop();
    $('#playSource').textContent = '▶';
  };
  $('#seek').oninput = () => {
    clearSourceSliceStop();
    source.currentTime = Number($('#seek').value);
    if (state.project.outputs.chinese_stem_file) dub.currentTime = source.currentTime;
  };
  $('#listenMode').onclick = (event) => {
    const button = event.target.closest('[data-listen]');
    if (button) {
      mode = button.dataset.listen;
      clearSourceSliceStop();
      source.pause();
      dub.pause();
      $$('[data-listen]').forEach((b) => b.classList.toggle('on', b === button));
      $('#playSource').textContent = '▶';
    }
  };
}
