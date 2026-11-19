import { $, $$, state, api, esc, t, guard, notice } from './session.js';
import { saveParameters, renderForms } from './forms.js';

const GROUPS = [
  ['quality', '质量档位'],
  ['output', '成品类型'],
];
const same = (a, b) =>
  a === b || (typeof a === 'number' && typeof b === 'number' && Math.abs(a - b) < 1e-9) || String(a) === String(b);

function valuesOf(scope) {
  return scope === 'proj' ? state.project?.settings : state.boot.settings;
}

// A project can only take project-scoped settings; global defaults take all of them.
export function presetChanges(preset, scope) {
  const keys = scope === 'proj' ? preset.project_keys : Object.keys(preset.changes);
  return Object.fromEntries(keys.map((key) => [key, preset.changes[key]]));
}

export function presetDifferences(preset, scope, values = valuesOf(scope)) {
  if (!values) return [];
  return Object.entries(presetChanges(preset, scope))
    .filter(([key, wanted]) => !same(values[key], wanted))
    .map(([key, wanted]) => ({ key, current: values[key], wanted }));
}

function shown(parameter, value) {
  if (typeof value === 'boolean') return t(value ? '开' : '关');
  const choice = parameter?.options?.find((item) => same(item[1], value));
  if (choice) return t(choice[0]);
  return value === '' || value == null ? t('空') : String(value);
}

function groupHTML(scope, [group, title], values) {
  const presets = state.boot.presets.filter((preset) => preset.group === group);
  const scored = presets.map((preset) => ({ preset, diff: presetDifferences(preset, scope, values) }));
  const exact = scored.find((item) => item.diff.length === 0);
  const nearest = [...scored].sort((a, b) => a.diff.length - b.diff.length)[0];
  const buttons = scored
    .map(
      ({ preset }) =>
        `<button data-preset="${esc(preset.id)}" data-preset-scope="${scope}" title="${esc(t(preset.description))}"${
          exact?.preset === preset ? ' class="on" aria-pressed="true"' : ' aria-pressed="false"'
        }>${esc(t(preset.label))}</button>`,
    )
    .join('');
  let detail = '';
  if (!exact && nearest) {
    const rows = nearest.diff
      .map(({ key, current, wanted }) => {
        const parameter = state.boot.parameters.find((item) => item.key === key);
        return `<li><span>${esc(t(parameter?.label || key))}</span><b>${esc(shown(parameter, current))}</b> → ${esc(
          shown(parameter, wanted),
        )}</li>`;
      })
      .join('');
    detail = `<details class="pr-diff"><summary>${esc(
      t('与“{name}”不同的参数：{count} 项', { name: t(nearest.preset.label), count: nearest.diff.length }),
    )}</summary><ul>${rows}</ul></details>`;
  }
  return `<div class="pr-group"><span class="pr-label">${esc(t(title))}</span><div class="seg">${buttons}</div>${detail}</div>`;
}

export function renderPresets() {
  if (!state.boot?.presets) return;
  $$('.presetbox').forEach((box) => {
    const scope = box.dataset.scope;
    const values = valuesOf(scope);
    const dynamic = $('.pr-dyn', box);
    if (!values) {
      dynamic.innerHTML = '';
      return;
    }
    const query = (state.paramQuery[scope] || '').trim();
    const page = $(scope === 'proj' ? '#project' : '#settings');
    page.classList.toggle('searching', !!query);
    const count = query ? $$('.schema .field', page).length : 0;
    dynamic.innerHTML =
      GROUPS.map((group) => groupHTML(scope, group, values)).join('') +
      (query
        ? `<p class="stat" role="status">${esc(count ? t('{count} 项匹配', { count }) : t('没有匹配的参数'))}</p>`
        : '');
  });
}

async function applyPreset(id, scope) {
  const preset = state.boot.presets.find((item) => item.id === id);
  if (!preset) return;
  if (scope === 'proj' && !state.project) throw new Error(t('请先新建或打开项目。'));
  const catalog = preset.requires || preset.prefer_requires ? await api('models/list') : { items: [] };
  const stateOf = (identifier) => catalog.items.find((entry) => entry.id === identifier)?.state;
  const changes = presetChanges(preset, scope);
  // The preferred backend is a separate install: switch to it only when it is ready.
  const preferred = preset.prefer_requires && stateOf(preset.prefer_requires) === 'ready';
  if (preferred) Object.assign(changes, preset.prefer);
  await saveParameters(changes, scope);
  const missingRequired = Boolean(
    preset.requires && stateOf(preset.requires) && stateOf(preset.requires) !== 'ready',
  );
  const missingPrefer = Boolean(preset.prefer_requires && !preferred);
  const parts = [t('已应用预设：{name}', { name: t(preset.label) })];
  if (missingRequired)
    parts.push(t('“{name}”需要先在“模型”页安装人声分离模型，否则无法导出。', { name: t(preset.label) }));
  if (missingPrefer)
    parts.push(
      t('“{name}”推荐配合 IndexTTS-2.5，配音时长更贴合原句；在“模型”页装好后再点一次此预设即可切换。', {
        name: t(preset.label),
      }),
    );
  notice(parts.join('\n'), missingRequired, { sticky: missingRequired });
}

export function initializePresets() {
  window.addEventListener('forms-rendered', renderPresets);
  document.addEventListener('input', (event) => {
    const field = event.target.closest('.pr-search');
    if (!field) return;
    const scope = field.closest('.presetbox')?.dataset.scope || 'def';
    state.paramQuery[scope] = field.value;
    renderForms();
  });
  document.addEventListener(
    'click',
    guard(async (event) => {
      const button = event.target.closest('[data-preset]');
      if (button) await applyPreset(button.dataset.preset, button.dataset.presetScope);
    }),
  );
}
