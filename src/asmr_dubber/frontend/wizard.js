import { $, state, api, esc, t, size, startTask, guard, notice, localize } from './session.js';
import { saveParameters } from './forms.js';
import { renderKeys } from './settings.js';

const TITLES = [
  ['检测硬件', '先看看这台电脑适合装什么。'],
  ['下载模型', '识别和配音模型在本机运行，只需下载一次。断线后再点一次会接着下。'],
  ['填写翻译 Key', '翻译靠云端大模型。填一个 API Key 并测试，已有译文字幕的话可以跳过。'],
];
const ACTIVE = ['queued', 'running', 'cancelling'];
let step = 1,
  recommendation = null,
  catalog = null,
  signature = '',
  testTaskId = '',
  testError = '';

const isOpen = () => $('#wizard').classList.contains('on');

async function load() {
  [recommendation, catalog] = await Promise.all([api('onboarding/recommendation'), api('models/list')]);
}

export async function openWizard() {
  step = 1;
  testTaskId = testError = '';
  $('#wizard').classList.add('on');
  await load();
  renderWizard();
}

async function finish(startProject) {
  await saveParameters({ onboarding_completed: true }, 'def');
  $('#wizard').classList.remove('on');
  if (!startProject) return;
  window.dispatchEvent(new CustomEvent('navigate', { detail: { page: 'home' } }));
  setTimeout(() => $('#dropNew').click(), 0);
}

const item = (id) => catalog?.items.find((entry) => entry.id === id);
const downloadTask = (id) =>
  [...state.tasks].reverse().find((task) => task.kind === 'download' && task.request.model === id);

function hardwareHTML() {
  const hardware = catalog.hardware;
  const gpu = hardware.gpu ? `${esc(hardware.gpu)}${hardware.vram_gb ? ` · ${hardware.vram_gb.toFixed(1)} GB` : ''}` : esc(t('没有检测到 NVIDIA 显卡'));
  const memory = hardware.memory_gb ? `${hardware.memory_gb.toFixed(1)} GB` : esc(t('未检测'));
  return `<div class="card device wz-device"><div><span>${t('显卡')}</span><b>${gpu}</b></div><div><span>${t('内存')}</span><b>${memory}</b></div><div><span>${t('磁盘')}</span><b>${size(catalog.free_bytes)}</b></div></div>`;
}

function recommendedIds() {
  return [recommendation.asr.id, recommendation.tts.id].filter(Boolean);
}

function stepOne() {
  const names = recommendedIds()
    .map((id) => `<li>${esc(t(item(id)?.name || id))}${item(id)?.disk_gb ? ` <small class="muted">${item(id).disk_gb} GB</small>` : ''}</li>`)
    .join('');
  const note = [recommendation.asr.note && t('推荐的识别模型在这台电脑上不合适（{note}），改用轻量的 Parakeet。', { note: t(recommendation.asr.note) }),
    !recommendation.tts.id && t('这台电脑跑不动本机的 IndexTTS2，配音请用 Edge TTS（免费、联网）或云端配音。')]
    .filter(Boolean)
    .map((text) => `<p class="stat">${esc(text)}</p>`)
    .join('');
  return `${hardwareHTML()}<div class="label"><b>${esc(t('推荐配置'))}：${esc(t(recommendation.title))}</b></div><div class="card wz-card"><p>${esc(t(recommendation.detail))}</p><p class="stat">${esc(t('下一步会下载：'))}</p><ul class="wz-list">${names}</ul>${note}</div>`;
}

function modelRow(id) {
  const entry = item(id);
  if (!entry) return '';
  const task = downloadTask(id);
  const running = task && ACTIVE.includes(task.status);
  const ratio = task && task.total > 0 ? Math.min(100, (task.current / task.total) * 100) : 0;
  const failed = task && ['failed', 'cancelled', 'interrupted'].includes(task.status);
  const busy = state.tasks.some((other) => ACTIVE.includes(other.status) && other.kind !== 'diagnostic');
  const action = running
    ? `<button class="btn" data-cancel="${task.id}">${t('暂停')}</button>`
    : entry.state === 'ready'
      ? `<span class="tag rec">${t('已安装')}</span>`
      : `<button class="btn primary" data-download-model="${id}"${busy ? ' disabled' : ''}>${t(failed ? '继续' : '下载')}</button>`;
  const detail = running ? task.message : failed ? task.error || task.message : t(entry.state);
  return `<div class="row mrow"><div class="grow"><div class="t">${esc(t(entry.name))}</div><div class="d">${esc(t(entry.description))}</div>${running ? `<div class="bar"><i style="width:${ratio}%"></i></div>` : ''}<div class="d">${esc(t(detail || ''))}</div></div><div class="size num">${entry.disk_gb ? `${entry.disk_gb} GB` : ''}</div><div class="act">${action}</div></div>`;
}

function stepTwo() {
  const ids = recommendedIds();
  const rows = ids.map(modelRow).join('');
  const speech = `<div class="row"><div class="grow"><div class="t">Edge TTS</div><div class="d">${t('微软在线语音，免费，要联网。音色固定，不能模仿原声。')}</div></div><span class="muted small">${t('已内置')}</span></div>`;
  return `<div class="card">${rows}${recommendation.tts.id ? '' : speech}</div><div class="row wz-source"><span class="muted small">${t('下载来源')}</span><div class="seg"><button data-source="modelscope"${state.boot.settings.download_source === 'modelscope' ? ' class="on"' : ''}>${t('国内镜像')}</button><button data-source="original"${state.boot.settings.download_source === 'original' ? ' class="on"' : ''}>${t('海外源')}</button></div></div><p class="stat">${t('也可以先跳过，到“模型”页再下载。')}</p>`;
}

function testResult() {
  const task = state.tasks.find((entry) => entry.id === testTaskId);
  if (testError) return `<p class="wz-result bad" role="status">${esc(testError)}</p>`;
  if (!task) return '';
  if (ACTIVE.includes(task.status)) return `<p class="wz-result" role="status">${t('测试中…')}</p>`;
  if (task.status === 'completed')
    return `<p class="wz-result ok" role="status">${esc(t(task.result?.message || '连接正常'))}</p>`;
  return `<p class="wz-result bad" role="status">${esc(t(task.error || task.message || '测试失败'))}</p>`;
}

function stepThree() {
  const providers = state.boot.choices.translation;
  const current = $('#wizardProvider')?.value || state.boot.settings.translation_provider;
  const available = state.boot.keys.translation_available?.[current];
  const choices = Object.entries(providers)
    .map(([id, provider]) => `<option value="${esc(id)}"${id === current ? ' selected' : ''}>${esc(t(provider.label))}</option>`)
    .join('');
  const saved = $('#wizardKey')?.value || '';
  return `<div class="card group"><div class="field"><label class="fl" for="wizardProvider">${t('翻译服务')}</label><select id="wizardProvider">${choices}</select></div><div class="field"><label class="fl" for="wizardKey">API Key<small>${esc(available ? t('已保存，留空就用已保存的。') : t('只保存在本机。'))}</small></label><input type="password" id="wizardKey" autocomplete="off" value="${esc(saved)}"></div></div><div class="row wz-test"><button class="btn" id="wizardTest">${t('保存并测试')}</button></div><div id="wizardResult">${testResult()}</div>`;
}

export function renderWizard() {
  if (!isOpen() || !catalog) return;
  const [title, description] = TITLES[step - 1];
  $('#wizardTitle').textContent = t(title);
  $('#wizardDesc').textContent = t(description);
  $('#wizardSteps').innerHTML = TITLES.map(
    ([name], index) =>
      `<span class="wz-step${index + 1 === step ? ' on' : ''}${index + 1 < step ? ' done' : ''}"><i>${index + 1}</i>${esc(t(name))}</span>`,
  ).join('');
  const body = $('#wizardBody');
  // Step 3 holds typed text; re-render it only when the step itself changed.
  if (step === 3 && body.dataset.step === '3') $('#wizardResult').innerHTML = testResult();
  else body.innerHTML = [stepOne, stepTwo, stepThree][step - 1]();
  body.dataset.step = String(step);
  $('#wizardBack').hidden = step === 1;
  $('#wizardNext').textContent = t(step === 3 ? '完成，开始新建项目' : '下一步');
  $('#wizardSkip').textContent = t(step === 3 ? '先不用，直接进入' : '稍后再说');
  localize($('#wizard'));
}

// Called on every task poll: refresh model state once a download changes, update progress.
export async function wizardTick() {
  if (!isOpen()) return;
  const next = JSON.stringify(
    state.tasks.filter((task) => task.kind === 'download').map((task) => [task.id, task.status]),
  );
  if (next !== signature) {
    signature = next;
    if (step < 3) await load();
  }
  renderWizard();
}

async function saveAndTest() {
  testError = '';
  const provider = $('#wizardProvider').value;
  const key = $('#wizardKey').value;
  if (provider !== state.boot.settings.translation_provider)
    await saveParameters({ translation_provider: provider }, 'def');
  try {
    if (key) await api('settings/key', { kind: 'translation', name: provider, value: key });
    $('#wizardKey').value = '';
    state.boot.keys = await api('settings/keys');
    renderKeys();
    if (!state.boot.keys.translation_available?.[provider]) throw new Error(t('还没有填写 API Key。'));
    testTaskId = (await startTask('diagnostic', { service: 'translation' }, false)).id;
  } catch (error) {
    testError = error.message;
  }
  $('#wizardBody').dataset.step = '';
  renderWizard();
}

export function initializeWizard() {
  $('#wizardNext').onclick = guard(async () => {
    if (step < 3) {
      step += 1;
      $('#wizardBody').dataset.step = '';
      renderWizard();
    } else await finish(true);
  });
  $('#wizardBack').onclick = () => {
    step -= 1;
    $('#wizardBody').dataset.step = '';
    renderWizard();
  };
  $('#wizardSkip').onclick = guard(() => finish(step === 3));
  $('#wizardBody').addEventListener(
    'click',
    guard(async (event) => {
      if (event.target.closest('#wizardTest')) await saveAndTest();
    }),
  );
  $('#wizardBody').addEventListener('change', (event) => {
    if (event.target.id !== 'wizardProvider') return;
    $('#wizardBody').dataset.step = '';
    renderWizard();
  });
  $('#reopenWizard').onclick = guard(openWizard);
  window.addEventListener('settings-changed', () => {
    $('#wizardBody').dataset.step = '';
    renderWizard();
  });
  if (state.boot.onboarding_required ?? !state.boot.settings.onboarding_completed) guard(openWizard)();
}
