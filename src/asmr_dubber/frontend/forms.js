import { $, $$, state, api, esc, t, options, field, enqueueSave, upload, chooseFile, guard, startTask } from './session.js';

export function renderForms() {
  if (!state.boot) return;
  $$('.schema').forEach(box => {
    const {panels, part, scope} = box.dataset;
    const values = scope === 'proj' ? state.project?.settings : state.boot.settings;
    if (!values) { box.innerHTML = ''; return; }
    const language = scope === 'proj' ? state.project.source_language : values.default_source_language;
    const fields = state.boot.parameters.filter(parameter => panels.split(',').includes(parameter.panel)
      && (scope !== 'proj' || parameter.scope === 'project')
      && (part === 'all' || parameter.basic === (part === 'basic'))
      && Object.entries(parameter.visible_when).every(([key, allowed]) => allowed.includes(values[key])));
    const groups = new Map();
    for (const parameter of fields) {
      const key = parameter.basic ? 'basic' : parameter.group;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(parameter);
    }
    const open = new Set($$('details[open]',box).map(node => node.dataset.group));
    box.innerHTML = [...groups].map(([group, list]) => {
      const body = list.map(parameter => control(parameter, values, scope, language)).join('');
      return group === 'basic' || panels === 'general' ? `<div class="card group">${body}</div>` : `<details class="card group" data-group="${esc(group)}"${open.has(group)?' open':''}><summary>${esc(t(group))}</summary><div>${body}</div></details>`;
    }).join('');
  });
}
function control(parameter, values, scope, language) {
  const key = parameter.key, value = values[key], id = `${scope}-${key}-${Math.random().toString(36).slice(2)}`;
  let choices = parameter.options, input;
  if (key === 'asr_backend' && language !== 'ja') choices = choices.filter(item => ['faster_whisper','generic_asr_api'].includes(item[1]));
  if (key === 'asr_model') choices = (state.boot.choices.asr[values.asr_backend]?.models || []).map(model => [model,model]);
  if (key === 'tts_model') choices = (state.boot.choices.tts[values.tts_backend]?.models || []).map(model => [model,model]);
  if (key === 'tts_voice') choices = (state.boot.choices.tts[values.tts_backend]?.voices || []).map(voice => [voice,voice]);
  if (key === 'translation_model') choices = (state.boot.choices.translation[values.translation_provider]?.models || []).map(model => [model,model]);
  if (key === 'asr_vad_mode' && language !== 'ja') choices = choices.filter(item => item[1] !== 'asmr');
  if (choices?.length && !choices.some(item => String(item[1]) === String(value)) && value != null) choices = [[String(value),value], ...choices];
  const data = `id="${id}" data-param="${key}" data-scope="${scope}"`;
  if (parameter.type === 'boolean') input = `<button type="button" class="switch${value?' on':''}" role="switch" aria-checked="${!!value}" aria-label="${esc(t(parameter.label))}" ${data}></button>`;
  else if (choices?.length && ['asr_model','tts_model','translation_model','tts_voice'].includes(key)) input = `<input type="text" list="${id}-options" value="${esc(value)}" ${data}><datalist id="${id}-options">${options(choices,value)}</datalist>`;
  else if (choices?.length) input = `<select aria-label="${esc(t(parameter.label))}" ${data}>${options(choices, value)}</select>`;
  else if (['number','integer'].includes(parameter.type)) input = `<span class="numf"><input type="number" value="${esc(value)}" step="${parameter.type==='integer'?'1':'any'}"${parameter.minimum!=null?` min="${parameter.minimum}"`:''}${parameter.maximum!=null?` max="${parameter.maximum}"`:''} ${data}></span>`;
  else if (parameter.type === 'array' || key.includes('prompt') || key.endsWith('params') || key.endsWith('extra_body') || key === 'autoflow_timestamp_footer') input = `<textarea class="mono" rows="3" ${data}>${esc(parameter.type==='array'?JSON.stringify(value):value)}</textarea>`;
  else input = `<input type="text" value="${esc(value)}" ${data}>`;
  if (key.endsWith('_audio')) input += `<button class="btn small" data-reference-upload="${key}" data-scope="${scope}">${t('选择文件')}</button>`;
  if (key === 'projects_root') input += `<button class="btn small" data-folder="${id}">${t('更改')}</button>`;
  if (key === 'tts_voice' && values.tts_backend === 'edge_tts') input += `<button class="btn small" data-preview-voice="${esc(value)}">${t('试听')}</button>`;
  return `<div class="field${input.startsWith('<textarea')?' col':''}"><label class="fl" for="${id}">${esc(t(parameter.label))}${parameter.hint?`<small>${esc(t(parameter.hint))}</small>`:''}</label><span class="control">${input}</span></div>`;
}
export function saveParameters(changes, scope) {
  return enqueueSave(async () => {
    const request = {changes};
    if (scope === 'proj') { request.project = state.project.manifest; request.revision = state.project.revision; }
    const result = await api('settings/update', request);
    if (scope === 'proj') {
      const latest = await api('projects/get',{project:state.project.manifest});
      state.project = latest;
    } else state.boot.settings = result;
    window.dispatchEvent(new CustomEvent('settings-changed',{detail:changes}));
    renderForms();
  });
}
export function initializeForms() {
  document.addEventListener('change', guard(async event => {
    const node = event.target;
    if (!node.dataset.param) return;
    const parameter = state.boot.parameters.find(item => item.key === node.dataset.param);
    let value = node.value;
    if (parameter.type === 'array') value = JSON.parse(value);
    else if (['number','integer'].includes(parameter.type)) value = Number(value);
    if (parameter.nullable && value === '') value = null;
    await saveParameters({[parameter.key]:value},node.dataset.scope);
  }));
  document.addEventListener('click', guard(async event => {
    const preview = event.target.closest('[data-preview-voice]');
    if (preview) { const task = await startTask('preview_edge',{voice:preview.dataset.previewVoice},false); window.dispatchEvent(new CustomEvent('preview-audio',{detail:task.id})); return; }
    const toggle = event.target.closest('button[data-param]');
    if (toggle) { await saveParameters({[toggle.dataset.param]:toggle.getAttribute('aria-checked')!=='true'},toggle.dataset.scope); return; }
    const uploadButton = event.target.closest('[data-reference-upload]');
    if (uploadButton) {
      const file = await chooseFile('audio/*'); if (!file) return;
      const owned = await upload(file);
      const stored = await api('settings/reference-upload',{path:owned.path});
      await saveParameters({[uploadButton.dataset.referenceUpload]:stored.path},uploadButton.dataset.scope);
    }
    const folder = event.target.closest('[data-folder]');
    if (folder) { const result = await api('files/folder'); if (result.path) { const input = document.getElementById(folder.dataset.folder); input.value=result.path; input.dispatchEvent(new Event('change',{bubbles:true})); } }
  }));
}
