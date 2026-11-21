import {
  $,
  state,
  api,
  esc,
  t,
  options,
  field,
  dialog,
  chooseFile,
  upload,
  startTask,
  guard,
  notice,
  projectBusy,
  flushSaves,
  stopAllAudio,
} from './session.js';
import { recordListUndo } from './sentence_editor.js';

export const SPEAKER_BUSY_CONTROLS = '#batchSpeakerBtn,[data-spk-edit],[data-spk-reset]';

function speakerSortKey(id) {
  const match = String(id || '').match(/^S(\d+)$/i);
  return match ? [0, Number(match[1]), id] : [1, 0, String(id).toLowerCase()];
}

function speakerTone(speakerId) {
  const value = String(speakerId || '').trim();
  const match = value.match(/^S(\d+)$/i);
  if (match) return String((Number(match[1]) - 1) % 6);
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) hash = (hash * 31 + value.charCodeAt(i)) | 0;
  return String(Math.abs(hash) % 6);
}

export function renderSpeakerFilter(project) {
  const filterEl = $('#speakerFilter');
  const batchBtn = $('#batchSpeakerBtn');
  if (!filterEl || !batchBtn || !project) return;
  const counts = new Map();
  let noneCount = 0;
  for (const s of project.sentences || []) {
    const spk = (s.speaker_id || '').trim();
    if (spk) counts.set(spk, (counts.get(spk) || 0) + 1);
    else noneCount += 1;
  }
  const hasSpeakers = counts.size > 0 || Object.keys(project.speaker_references || {}).length > 0;
  const visible =
    hasSpeakers || Boolean(project.settings?.asr_diarize_speakers) || (state.speakerFilter || 'all') !== 'all';
  filterEl.hidden = !visible;
  batchBtn.hidden = !visible;
  if (!visible) {
    state.speakerFilter = 'all';
    return;
  }
  const ordered = [...counts.keys()].sort((a, b) => {
    const [ga, na, sa] = speakerSortKey(a);
    const [gb, nb, sb] = speakerSortKey(b);
    return ga - gb || na - nb || sa.localeCompare(sb);
  });
  const opts = [[t('全部说话人'), 'all']];
  for (const spk of ordered) {
    opts.push([t('{speaker} · {count} 句', { speaker: spk, count: counts.get(spk) }), spk]);
  }
  if (noneCount > 0) opts.push([t('未标注 · {count} 句', { count: noneCount }), '__none__']);
  const valid = new Set(opts.map(([, v]) => v));
  if (!valid.has(state.speakerFilter)) state.speakerFilter = 'all';
  filterEl.innerHTML = opts
    .map(
      ([label, val]) =>
        `<option value="${esc(val)}"${val === state.speakerFilter ? ' selected' : ''}>${esc(label)}</option>`,
    )
    .join('');
}

export function renderSpeakerVoices(project) {
  const box = $('#speakerVoicesSection');
  if (!box) return;
  const profiles = project?.speaker_profiles || [];
  const show =
    profiles.length >= 2 ||
    (profiles.length >= 1 &&
      (Boolean(project.settings?.asr_diarize_speakers) || profiles.some((p) => p.mode !== 'auto')));
  if (!show) {
    box.hidden = true;
    box.innerHTML = '';
    return;
  }
  box.hidden = false;
  const ttsInfo = state.boot?.choices?.tts?.[project.settings?.tts_backend];
  const supportsRef = ttsInfo ? ttsInfo.reference !== false : true;
  let noteHTML = '';
  if (!supportsRef) {
    noteHTML = `<p class="stat spk-note">${esc(t('当前配音模型不支持声音克隆；切换为 IndexTTS2 等克隆模型后，按说话人指定的音色就会生效。'))}</p>`;
  } else if (profiles.length === 1) {
    noteHTML = `<p class="stat spk-note">${esc(t('当前只识别出 1 位说话人。若是双人音声，可在第 1 步把“说话人数量”设为 2 重新识别，或在左侧表格“说话人”列手动分出 S02 后在此分别选音色。'))}</p>`;
  }
  const rowsHTML = profiles
    .map((p) => {
      const desc =
        p.mode === 'external'
          ? t('外部音色 · {name}', { name: p.audio_name || p.audio_path })
          : p.mode === 'project_sentence'
            ? t('指定原声 · {id}', { id: p.sentence_id })
            : p.sentence_id
              ? t('自动原声 · {id}', { id: p.sentence_id })
              : t('未分配可用原声句');
      const canPlay = (p.mode === 'external' && Boolean(p.audio_url)) || Boolean(p.sentence_id);
      return `<div class="spk-row" data-speaker-id="${esc(p.speaker_id)}"><span class="spk-badge" data-spk-tone="${speakerTone(p.speaker_id)}">${esc(p.speaker_id)}</span><div class="grow spk-meta"><div class="t">${esc(desc)}</div><small>${esc(t('{count} 句', { count: p.sentence_count }))}</small></div><div class="spk-actions"><button type="button" class="btn plain small" data-spk-audition="${esc(p.speaker_id)}"${canPlay ? '' : ' disabled'}>${t('试听')}</button><button type="button" class="btn small" data-spk-edit="${esc(p.speaker_id)}">${t('指定音色')}</button>${p.mode !== 'auto' ? `<button type="button" class="btn plain small" data-spk-reset="${esc(p.speaker_id)}">${t('恢复自动')}</button>` : ''}</div></div>`;
    })
    .join('');
  box.innerHTML = `<div class="more">${t('按说话人指定参考音色')}</div><div class="card group spk-voices">${rowsHTML}</div>${noteHTML}<audio id="speakerVoicePreview" controls preload="none" hidden style="width:100%;margin-top:8px"></audio>`;
}

async function loadSentencePreview(player, sentenceId, manifest, revision, activeCheck) {
  const task = await startTask(
    'preview_reference',
    { project: manifest, revision, sentence: sentenceId },
    false,
  );
  const poll = () =>
    new Promise((resolve) => {
      const timer = setInterval(
        guard(async () => {
          if (!activeCheck()) {
            clearInterval(timer);
            resolve();
            return;
          }
          const latest = await api('tasks/get', { identifier: task.id });
          if (!['queued', 'running', 'cancelling'].includes(latest.status)) {
            clearInterval(timer);
            if (latest.result?.url && activeCheck()) {
              player.hidden = false;
              player.src = latest.result.url;
              await player.play().catch(() => {});
            }
            resolve();
          }
        }),
        350,
      );
    });
  await poll();
}

export async function speakerReferenceDialog(profile) {
  const project = state.project;
  let external =
    profile.mode === 'external' && profile.audio_path
      ? { path: profile.audio_path, name: profile.audio_name, url: profile.audio_url }
      : null;
  const modeChoices = [
    [
      'auto',
      '自动选择该角色原声',
      profile.auto_sentence_id
        ? t('推荐句：{id}', { id: profile.auto_sentence_id })
        : t('按台词时长和清晰度自动挑一句'),
    ],
    ['project_sentence', '指定本项目内某句', t('从原声片段里挑最能代表该角色的一句')],
    ['external', '使用外部参考音频', t('上传外部干声片段，适合两人原声很像或自定义音线')],
  ];
  const radios = modeChoices
    .map(
      ([val, label, hint]) =>
        `<label class="choice"><input type="radio" name="spkRefMode" value="${val}"${profile.mode === val ? ' checked' : ''}><div>${t(label)}<small>${esc(hint)}</small></div></label>`,
    )
    .join('');
  dialog(
    t('{speaker} 的参考音色', { speaker: profile.speaker_id }),
    `<div class="card group">${radios}</div>` +
      `<div id="spkProjBox" class="card group">${field('项目内的句子', `<select id="spkRefSentence">${options(profile.choices || [], profile.sentence_id || '')}</select>`)}</div>` +
      `<div id="spkExtBox" class="card group">${field('外部音频', `<button type="button" class="btn" id="spkPickExternal">${t('选择文件')}</button><span id="spkExternalName">${esc(external?.name || '')}</span>`)}${field('参考文字', `<input type="text" id="spkExternalText" value="${esc(profile.text || '')}" placeholder="${esc(t('可选'))}">`)}${field('外部音频语言', `<select id="spkExternalLang">${options([['跟随项目', 'auto'], ['日语', 'ja'], ['英语', 'en'], ['中文', 'zh']], profile.language || 'auto')}</select>`)}</div>` +
      '<audio id="spkRefPreview" controls preload="none" style="width:100%;margin-top:8px"></audio>',
    '采用',
    async () => {
      await flushSaves();
      const mode = $('input[name="spkRefMode"]:checked').value;
      const latest = await api('projects/get', { project: project.manifest });
      const result = await api('projects/speaker-reference', {
        project: latest.manifest,
        revision: latest.revision,
        speaker_id: profile.speaker_id,
        mode,
        sentence: mode === 'project_sentence' ? $('#spkRefSentence').value : '',
        external: mode === 'external' ? external?.path || '' : '',
        external_name: mode === 'external' ? external?.name || '' : '',
        text: mode === 'external' ? $('#spkExternalText').value : '',
        language: mode === 'external' ? $('#spkExternalLang').value : 'auto',
      });
      state.project = result.project;
      window.dispatchEvent(new Event('project-changed'));
      notice(result.message);
    },
  );
  let gen = 0;
  const syncMode = async () => {
    const mode = $('input[name="spkRefMode"]:checked').value;
    $('#spkProjBox').hidden = mode !== 'project_sentence';
    $('#spkExtBox').hidden = mode !== 'external';
    const player = $('#spkRefPreview');
    const cur = ++gen;
    if (mode === 'external') {
      if (external?.url) player.src = external.url;
      else player.removeAttribute('src');
      return;
    }
    const sid = mode === 'project_sentence' ? $('#spkRefSentence').value : profile.auto_sentence_id;
    if (!sid) {
      player.removeAttribute('src');
      return;
    }
    await loadSentencePreview(
      player,
      sid,
      project.manifest,
      project.revision,
      () => cur === gen && player.isConnected && $('#dialog').classList.contains('on'),
    );
  };
  for (const r of document.querySelectorAll('input[name="spkRefMode"]')) {
    r.onchange = guard(syncMode);
  }
  $('#spkRefSentence').onchange = guard(syncMode);
  $('#spkPickExternal').onclick = guard(async () => {
    const file = await chooseFile('audio/*');
    if (!file) return;
    const uploaded = await upload(file);
    external = { path: uploaded.path, name: file.name, url: uploaded.url };
    $('#spkExternalName').textContent = file.name;
    $('#spkRefPreview').src = uploaded.url;
  });
  await syncMode();
}

function batchSpeakerDialog(hooks) {
  const list = hooks.filteredSentences();
  if (!list.length) throw new Error(t('当前筛选下没有句子。'));
  const existing = [...new Set(['S01', 'S02', ...(state.project?.sentences || []).map((s) => s.speaker_id).filter(Boolean)])];
  dialog(
    '批量改说话人',
    `<p class="stat">${esc(t('将对当前筛选出的 {count} 句统一设置说话人（支持 Ctrl+Z 撤销）。', { count: list.length }))}</p>` +
      `<div class="card group">${field('说话人编号', `<input type="text" id="batchSpkInput" list="batchSpkOptions" value="S01" maxlength="32" placeholder="S01"><datalist id="batchSpkOptions">${existing.map((s) => `<option value="${esc(s)}">`).join('')}</datalist>`)}${field('清空标注', '<input type="checkbox" id="batchSpkClear">')}</div>`,
    '应用',
    async () => {
      if (projectBusy()) throw new Error(t('请等当前任务结束后再操作。'));
      await flushSaves();
      const target = $('#batchSpkClear').checked ? null : $('#batchSpkInput').value.trim() || null;
      const beforeList = state.project.sentences;
      const targetIds = new Set(list.map((s) => s.id));
      const nextList = beforeList.map((s) =>
        targetIds.has(s.id) && (s.speaker_id || null) !== target
          ? { ...s, speaker_id: target, tts_file: null, tts_url: null, tts_candidates: [] }
          : s,
      );
      const curSel = hooks.getSelected();
      recordListUndo(beforeList, nextList, curSel, curSel);
      state.project.sentences = nextList;
      await hooks.saveRows();
      hooks.renderProject();
    },
  );
}

export function initializeSpeakerVoices(hooks) {
  let auditionGen = 0;
  $('#speakerFilter').onchange = () => {
    state.speakerFilter = $('#speakerFilter').value || 'all';
    const box = $('.tablebox');
    if (box) box.scrollTop = 0;
    hooks.renderProject();
  };
  $('#batchSpeakerBtn').onclick = guard(() => batchSpeakerDialog(hooks));
  document.addEventListener(
    'click',
    guard(async (event) => {
      const audBtn = event.target.closest('[data-spk-audition]');
      if (audBtn && state.project) {
        const spk = audBtn.dataset.spkAudition;
        const profile = (state.project.speaker_profiles || []).find((p) => p.speaker_id === spk);
        const player = $('#speakerVoicePreview');
        if (!profile || !player) return;
        stopAllAudio(player);
        const cur = ++auditionGen;
        if (profile.mode === 'external') {
          if (!profile.audio_url) throw new Error(t('外部参考音频不可用或已移动，请重新指定。'));
          player.hidden = false;
          player.src = profile.audio_url;
          await player.play().catch(() => {});
        } else if (profile.sentence_id) {
          await loadSentencePreview(
            player,
            profile.sentence_id,
            state.project.manifest,
            state.project.revision,
            () => cur === auditionGen && player.isConnected,
          );
        }
        return;
      }
      const editBtn = event.target.closest('[data-spk-edit]');
      if (editBtn && state.project) {
        const spk = editBtn.dataset.spkEdit;
        const profile = (state.project.speaker_profiles || []).find((p) => p.speaker_id === spk);
        if (profile) await speakerReferenceDialog(profile);
        return;
      }
      const resetBtn = event.target.closest('[data-spk-reset]');
      if (resetBtn && state.project) {
        await flushSaves();
        const latest = await api('projects/get', { project: state.project.manifest });
        const result = await api('projects/speaker-reference', {
          project: latest.manifest,
          revision: latest.revision,
          speaker_id: resetBtn.dataset.spkReset,
          mode: 'auto',
        });
        state.project = result.project;
        window.dispatchEvent(new Event('project-changed'));
        notice(result.message);
      }
    }),
  );
}
