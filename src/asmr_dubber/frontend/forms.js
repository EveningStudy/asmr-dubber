import {
  $,
  $$,
  state,
  api,
  esc,
  t,
  options,
  field,
  enqueueSave,
  upload,
  chooseFile,
  guard,
  startTask,
} from './session.js';

export function renderForms() {
  if (!state.boot) return;
  $$('.schema').forEach((box) => {
    const { panels, part, scope } = box.dataset;
    const values = scope === 'proj' ? state.project?.settings : state.boot.settings;
    if (!values) {
      box.innerHTML = '';
      return;
    }
    const language = scope === 'proj' ? state.project.source_language : values.default_source_language;
    const conditions = {
      ...values,
      source_language: language,
      asmr_vad_ready: state.boot.choices.asmr_vad_ready,
    };
    const matches = (rule) =>
      Object.entries(rule).every(([key, allowed]) => allowed.includes(conditions[key]));
    const fields = state.boot.parameters.filter(
      (parameter) =>
        panels.split(',').includes(parameter.panel) &&
        (scope !== 'proj' || parameter.scope === 'project') &&
        (part === 'all' || parameter.basic === (part === 'basic')) &&
        matches(parameter.visible_when) &&
        (!parameter.visible_any || parameter.visible_any.some(matches)),
    );
    const groups = new Map();
    for (const parameter of fields) {
      const key = parameter.basic ? 'basic' : parameter.group;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(parameter);
    }
    const open = new Set($$('details[open]', box).map((node) => node.dataset.group));
    box.innerHTML = [...groups]
      .map(([group, list]) => {
        const body = list.map((parameter) => control(parameter, values, scope, language)).join('');
        return group === 'basic' || panels === 'general'
          ? `<div class="card group">${body}</div>`
          : `<details class="card group" data-group="${esc(group)}"${open.has(group) ? ' open' : ''}><summary>${esc(t(group))}</summary><div>${body}</div></details>`;
      })
      .join('');
  });
}
function control(parameter, values, scope, language) {
  const key = parameter.key,
    id = `${scope}-${key}-${Math.random().toString(36).slice(2)}`;
  const mapping = parameter.display_conditions || parameter.display_values;
  const value = mapping
    ? Object.entries(mapping).find(([label, fields]) =>
        Object.entries(fields).every(([name, v]) => values[name] === v),
      )?.[0]
    : values[key];
  let choices = parameter.options,
    input;
  if (parameter.options_source) {
    const source = parameter.options_source;
    const list = state.boot.choices[source.domain][values[source.dependency]]?.[source.attribute] || [];
    choices = (Array.isArray(list) ? list : list[language] || []).map((v) => [v, v]);
  }
  if (parameter.display_options) choices = parameter.display_options;
  if (parameter.options_when) {
    const conditions = {
      ...values,
      source_language: language,
      asmr_vad_ready: state.boot.choices.asmr_vad_ready,
    };
    choices = choices.filter(
      ([, v]) =>
        !parameter.options_when[v] ||
        Object.entries(parameter.options_when[v]).every(([k, allowed]) => allowed.includes(conditions[k])),
    );
  }
  if (parameter.type === 'array' && choices?.length)
    choices = [
      ...choices,
      ...(value || []).filter((v) => !choices.some((item) => item[1] === v)).map((v) => [v, v]),
    ];
  else if (choices?.length && !choices.some((item) => String(item[1]) === String(value)) && value != null)
    choices = [[String(value), value], ...choices];
  const data = `id="${id}" data-param="${key}" data-scope="${scope}"`;
  if (parameter.type === 'boolean' && !parameter.display_options)
    input = `<button type="button" class="switch${value ? ' on' : ''}" role="switch" aria-checked="${!!value}" aria-label="${esc(t(parameter.label))}" ${data}></button>`;
  else if (choices?.length && parameter.custom)
    input = `<input type="text" list="${id}-options" value="${esc(value)}" ${data}><datalist id="${id}-options">${options(choices, value)}</datalist>`;
  else if (choices?.length && parameter.type === 'array')
    input = `<select multiple ${data}>${choices.map(([label, v]) => `<option value="${esc(v)}"${value.includes(v) ? ' selected' : ''}>${esc(t(label))}</option>`).join('')}</select>`;
  else if (choices?.length)
    input = `<select aria-label="${esc(t(parameter.label))}" ${data}>${options(choices, value)}</select>`;
  else if (['number', 'integer'].includes(parameter.type))
    input = `<span class="numf"><input type="number" value="${esc(value)}" step="${parameter.type === 'integer' ? '1' : 'any'}"${parameter.minimum != null ? ` min="${parameter.minimum}"` : ''}${parameter.maximum != null ? ` max="${parameter.maximum}"` : ''} ${data}></span>`;
  else if (
    parameter.type === 'array' ||
    key.includes('prompt') ||
    key.endsWith('params') ||
    key.endsWith('extra_body') ||
    key === 'autoflow_timestamp_footer'
  )
    input = `<textarea class="mono" rows="3"${key.startsWith('translation_prompt') ? ` placeholder="${esc(state.boot.choices.translation_prompts[key.endsWith('_ja') ? 'ja' : key.endsWith('_en') ? 'en' : key.endsWith('_zh') ? 'zh' : language] || '')}"` : ''} ${data}>${esc(parameter.type === 'array' ? JSON.stringify(value) : value)}</textarea>`;
  else input = `<input type="text" value="${esc(value)}" ${data}>`;
  if (key.endsWith('_audio'))
    input += `<button class="btn small" data-reference-upload="${key}" data-scope="${scope}">${t('选择文件')}</button>`;
  if (key === 'projects_root') input += `<button class="btn small" data-folder="${id}">${t('更改')}</button>`;
  if (key === 'tts_voice' && values.tts_backend === 'edge_tts')
    input += `<button class="btn small" data-preview-voice="${esc(value)}">${t('试听')}</button>`;
  if (key.startsWith('translation_prompt'))
    input += `<button class="btn small" data-reset-param="${key}" data-scope="${scope}">${t('恢复内置 Prompt')}</button>`;
  return `<div class="field${input.startsWith('<textarea') ? ' col' : ''}"><label class="fl" for="${id}">${esc(t(parameter.label))}${parameter.hint ? `<small>${esc(t(parameter.hint))}</small>` : ''}</label><span class="control">${input}</span></div>`;
}
export function saveParameters(changes, scope) {
  return enqueueSave(async () => {
    const request = { changes };
    if (scope === 'proj') {
      request.project = state.project.manifest;
      request.revision = state.project.revision;
    }
    const result = await api('settings/update', request);
    if (scope === 'proj') {
      const latest = await api('projects/get', { project: state.project.manifest });
      state.project = latest;
    } else state.boot.settings = result;
    window.dispatchEvent(new CustomEvent('settings-changed', { detail: changes }));
    renderForms();
  });
}
export function initializeForms() {
  document.addEventListener(
    'change',
    guard(async (event) => {
      const node = event.target;
      if (!node.dataset.param) return;
      const parameter = state.boot.parameters.find((item) => item.key === node.dataset.param);
      let value = node.value;
      if (parameter.type === 'array')
        value = node.multiple ? [...node.selectedOptions].map((option) => option.value) : JSON.parse(value);
      else if (['number', 'integer'].includes(parameter.type)) {
        // An emptied number box is not zero: clear nullable values, otherwise restore the saved one.
        if (value.trim() === '') {
          if (!parameter.nullable) return renderForms();
          value = null;
        } else value = Number(value);
      }
      if (parameter.nullable && value === '') value = null;
      await saveParameters({ [parameter.key]: value }, node.dataset.scope);
    }),
  );
  document.addEventListener(
    'click',
    guard(async (event) => {
      const reset = event.target.closest('[data-reset-param]');
      if (reset) {
        await saveParameters({ [reset.dataset.resetParam]: '' }, reset.dataset.scope);
        return;
      }
      const preview = event.target.closest('[data-preview-voice]');
      if (preview) {
        const task = await startTask('preview_edge', { voice: preview.dataset.previewVoice }, false);
        window.dispatchEvent(new CustomEvent('preview-audio', { detail: task.id }));
        return;
      }
      const toggle = event.target.closest('button[data-param]');
      if (toggle) {
        await saveParameters(
          { [toggle.dataset.param]: toggle.getAttribute('aria-checked') !== 'true' },
          toggle.dataset.scope,
        );
        return;
      }
      const uploadButton = event.target.closest('[data-reference-upload]');
      if (uploadButton) {
        const file = await chooseFile('audio/*');
        if (!file) return;
        const owned = await upload(file);
        const stored = await api('settings/reference-upload', { path: owned.path });
        await saveParameters(
          { [uploadButton.dataset.referenceUpload]: stored.path },
          uploadButton.dataset.scope,
        );
      }
      const folder = event.target.closest('[data-folder]');
      if (folder) {
        const result = await api('files/folder');
        if (result.path) {
          const input = document.getElementById(folder.dataset.folder);
          input.value = result.path;
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    }),
  );
}
