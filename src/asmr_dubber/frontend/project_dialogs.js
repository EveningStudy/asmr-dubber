import {
  $,
  $$,
  state,
  api,
  t,
  options,
  field,
  dialog,
  chooseFile,
  upload,
  startTask,
  guard,
} from './session.js';

export async function transcriptDialog(kind = 'source', timing = 'estimate') {
  let file = null;
  dialog(
    '导入字幕或台本',
    `<div class="card group">${field(
      '导入内容',
      `<select id="scriptKind">${options(
        [
          ['原文', 'source'],
          ['译文', 'zh'],
        ],
        kind,
      )}</select>`,
    )}${field('选择文件', `<button class="btn" id="pickTranscript">${t('选择文件')}</button><span id="transcriptFilename"></span>`)}${field(
      '纯台本的时间轴',
      `<select id="plainTiming">${options(
        [
          ['按台词长度估算（无需模型，之后手动校对）', 'estimate'],
          ['Qwen3 时间对齐', 'qwen'],
          ['先运行 ASR/翻译，再用大模型校对台本（推荐）', 'script_review'],
        ],
        timing,
      )}</select>`,
    )}</div><label>${t('粘贴台本')}<textarea id="pastedTranscript" rows="6"></textarea></label>`,
    '导入',
    async () => {
      await startTask('import', {
        file: file?.path || null,
        text: $('#pastedTranscript').value,
        script_kind: $('#scriptKind').value,
        timing: $('#plainTiming').value,
      });
    },
  );
  $('#pickTranscript').onclick = guard(async () => {
    const picked = await chooseFile('.srt,.vtt,.ass,.lrc,.txt');
    if (picked) {
      file = await upload(picked);
      $('#transcriptFilename').textContent = picked.name;
    }
  });
  const timingLabels = {
    estimate: '按台词长度估算（无需模型，之后手动校对）',
    qwen: 'Qwen3 时间对齐',
    script_review: '先运行 ASR/翻译，再用大模型校对台本（推荐）',
  };
  const refreshTiming = () => {
    const current = $('#plainTiming').value;
    const available = state.boot.choices.transcript_timing[$('#scriptKind').value];
    $('#plainTiming').innerHTML = options(
      available.map((value) => [timingLabels[value], value]),
      available.includes(current) ? current : available[0],
    );
  };
  $('#scriptKind').onchange = refreshTiming;
  refreshTiming();
}
export async function referenceDialog(manifest = state.project.manifest, batch = false, preferred = '') {
  const project = await api('projects/get', { project: manifest });
  let external =
    !preferred && project.reference_external && project.external_reference_url
      ? { path: project.settings.tts_external_reference_audio, url: project.external_reference_url }
      : null;
  dialog(
    batch ? '试听并选择' : '音色参考',
    `<div class="card group">${field('项目内的句子', `<select id="referenceSentence">${options(project.reference_choices, preferred || project.reference_selected)}</select>`)}${field('开始（秒）', '<input type="number" step="any" id="referenceStart">')}${field('结束（秒）', '<input type="number" step="any" id="referenceEnd">')}</div><label>${t('参考文字')}<textarea id="referenceText"></textarea></label><audio id="referencePreview" controls preload="none" style="width:100%"></audio><div class="card group">${field('外部音频', `<button class="btn" id="pickReference">${t('选择文件')}</button><span id="referenceFileName"></span>`)}${field(
      '外部音频语言',
      `<select id="referenceLanguage">${options(
        [
          ['跟随项目', 'auto'],
          ['日语', 'ja'],
          ['英语', 'en'],
          ['中文', 'zh'],
        ],
        'auto',
      )}</select>`,
    )}</div>`,
    '采用',
    async () => {
      const latest = await api('projects/get', { project: manifest });
      const result = await api('projects/reference', {
        project: manifest,
        revision: latest.revision,
        sentence: $('#referenceSentence').value,
        external: external?.path || '',
        text: $('#referenceText').value,
        language: $('#referenceLanguage').value,
        start: external ? null : Number($('#referenceStart').value),
        end: external ? null : Number($('#referenceEnd').value),
      });
      if (state.project?.manifest === manifest) {
        state.project = result.project;
        window.dispatchEvent(new Event('project-changed'));
      }
    },
  );
  const previewPlayer = $('#referencePreview');
  let previewGeneration = 0;
  async function preview() {
    const generation = ++previewGeneration;
    const sentence = project.sentences.find((s) => s.id === $('#referenceSentence').value);
    if (!sentence) return;
    $('#referenceStart').value = sentence.start_seconds;
    $('#referenceEnd').value = sentence.end_seconds;
    $('#referenceText').value = sentence.source_text;
    const task = await startTask(
      'preview_reference',
      { project: manifest, revision: project.revision, sentence: sentence.id },
      false,
    );
    const interval = setInterval(
      guard(async () => {
        if (
          !previewPlayer.isConnected ||
          !$('#dialog').classList.contains('on') ||
          generation !== previewGeneration
        ) {
          clearInterval(interval);
          return;
        }
        const latest = await api('tasks/get', { identifier: task.id });
        if (!['queued', 'running', 'cancelling'].includes(latest.status)) {
          clearInterval(interval);
          if (latest.result?.url && generation === previewGeneration && previewPlayer.isConnected)
            previewPlayer.src = latest.result.url;
        }
      }),
      400,
    );
  }
  $('#referenceSentence').onchange = guard(async () => {
    external = null;
    $('#referenceFileName').textContent = '';
    await preview();
  });
  $('#pickReference').onclick = guard(async () => {
    const file = await chooseFile('audio/*');
    if (file) {
      external = await upload(file);
      $('#referenceFileName').textContent = file.name;
      $('#referencePreview').src = external.url;
    }
  });
  if (external) {
    $('#referencePreview').src = external.url;
    $('#referenceFileName').textContent = external.path.split(/[\\/]/).pop();
    $('#referenceText').value = project.settings.tts_external_reference_text;
    $('#referenceLanguage').value = project.settings.tts_external_reference_language;
  } else await preview();
}
