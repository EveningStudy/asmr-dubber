export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
export const state = {boot: null, project: null, tasks: [], queue: [], page: 'home', step: 1, filter: 'all', language: 'zh', saveChain: Promise.resolve(), saveError: null, seenTasks: new Set()};
export const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let templateSource, templates=[];
function translatedText(source) {
  const locales=state.boot?.locales||{};
  if(locales[source])return locales[source];
  const prefix=source.match(/^(\[\d+\/\d+\]\s*)(.*)$/);
  if(prefix)return prefix[1]+translatedText(prefix[2]);
  if(templateSource!==locales){
    templateSource=locales;
    templates=Object.entries(locales).filter(([key])=>/\{\w+\}/.test(key)).sort((a,b)=>b[0].length-a[0].length).map(([key,text])=>{
      const names=[...key.matchAll(/\{(\w+)\}/g)].map(match=>match[1]);
      const pattern=key.split(/(\{\w+\})/).map(part=>/^\{\w+\}$/.test(part)?'(.+?)':part.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('');
      return {pattern:new RegExp(`^${pattern}$`),names,text};
    });
  }
  for(const template of templates){
    const match=source.match(template.pattern);if(!match)continue;
    return template.text.replace(/\{(\w+)\}/g,(_,name)=>{const value=match[template.names.indexOf(name)+1];return locales[value]||value;});
  }
  return source.split(' · ').map(part=>locales[part]||part).join(' · ');
}
export function t(source, values = {}) {
  const label = state.boot?.labels[source] || source;
  let text = state.language === 'en' ? state.boot?.locales[source] || String(label).split('\n').map(translatedText).join('\n') : label;
  for (const [key, value] of Object.entries(values)) text = text.replaceAll(`{${key}}`, String(value));
  return text;
}
export function localize(root = document) {
  $$('[data-i18n]', root).forEach(node => {
    node.dataset.i18nSource ||= node.textContent;
    node.textContent = t(node.dataset.i18nSource);
  });
  $$('[data-placeholder]', root).forEach(node => node.placeholder = t(node.dataset.placeholder));
  document.documentElement.lang = state.language === 'en' ? 'en' : 'zh-CN';
  $$('[data-language]').forEach(button => button.classList.toggle('on', button.dataset.language === state.language));
}
export async function api(route, data = {}) {
  const response = await fetch(`/api/${route}`, {method:'POST', headers:{'Content-Type':'application/json','X-ASMR-Token':$('meta[name="asmr-session"]').content}, body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || result.error || response.statusText);
  return result;
}
export async function upload(file) {
  const response = await fetch('/api/uploads', {method:'POST', headers:{'X-ASMR-Token':$('meta[name="asmr-session"]').content,'X-Filename':encodeURIComponent(file.name)}, body:file});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}
let noticeTimer;
export function notice(message, error = false) {
  clearTimeout(noticeTimer);
  const box = $('#notice'); box.textContent = t(String(message)); box.hidden = false; box.classList.toggle('error', error);
  if (!error) noticeTimer = setTimeout(() => box.hidden = true, 5000);
}
export function guard(action) {
  return async event => { try { await action(event); } catch (error) { notice(error.message, true); } };
}
export function enqueueSave(action) {
  const operation = state.saveChain.catch(() => {}).then(action);
  state.saveChain = operation;
  operation.then(() => { state.saveError = null; $('#saveState').textContent = t('修改会自动保存'); }, error => { state.saveError = error; $('#saveState').textContent = t('保存失败，草稿已保留'); notice(error.message, true); });
  return operation;
}
export async function flushSaves() { await state.saveChain; if (state.saveError) throw state.saveError; }
export const time = value => { const n = Number(value || 0); return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toFixed(1).padStart(4,'0')}`; };
export const size = bytes => `${(bytes / 1024**3).toFixed(2)} GB`;
export function options(items, selected) {
  return items.map(([label, value]) => `<option value="${esc(value)}"${String(value)===String(selected)?' selected':''}>${esc(t(label))}</option>`).join('');
}
export function field(label, control, hint = '') {
  return `<div class="field"><div class="fl">${esc(t(label))}${hint ? `<small>${esc(t(hint))}</small>` : ''}</div>${control}</div>`;
}
export function dialog(title, body, submitLabel, submit, description = '') {
  $('#dialogTitle').textContent = t(title); $('#dialogDescription').textContent = description;
  $('#dialogBody').innerHTML = body; $('#dialogSubmit').textContent = t(submitLabel);
  $('#dialogSubmit').hidden = !submit;
  $('#dialogSubmit').onclick = guard(async () => {
    const button = $('#dialogSubmit'); button.disabled = true;
    try { await submit(); $('#dialog').classList.remove('on'); } finally { button.disabled = false; }
  });
  $('#dialog').classList.add('on');
  localize($('#dialog'));
}
export async function chooseFile(accept = '') {
  return new Promise(resolve => {
    const input = document.createElement('input'); input.type = 'file'; input.accept = accept;
    input.addEventListener('change', () => { resolve(input.files[0] || null); input.remove(); }, {once:true});
    input.addEventListener('cancel', () => { resolve(null); input.remove(); }, {once:true});
    input.hidden = true; document.body.append(input); input.click();
  });
}
export function taskHTML(task) {
  const active = ['queued','running','cancelling'].includes(task.status);
  const ratio = task.total > 0 ? Math.min(100, task.current/task.total*100) : 0;
  return `<div class="card taskbox"><div class="row"><div class="grow"><div class="t">${esc(t(task.kind))} · ${esc(t(task.status))}</div><div class="d">${esc(t(task.error || task.message))}</div><div class="bar"><i style="width:${ratio}%"></i></div></div>${active ? `<button class="btn" data-cancel="${task.id}">${t('暂停')}</button>` : ['cancelled','failed','interrupted'].includes(task.status) ? `<button class="btn" data-resume="${task.id}">${t('继续')}</button>` : ''}</div><details id="task-log-${task.id}"${document.getElementById(`task-log-${task.id}`)?.open?' open':''}><summary>${t('日志')}</summary><pre class="log">${esc(task.logs.join('\n'))}</pre></details></div>`;
}
export async function startTask(kind, extra = {}, project = true) {
  await flushSaves();
  const request = {kind, ...extra};
  if (project) { if (!state.project) throw new Error(t('请先新建或打开项目。')); request.project = state.project.manifest; request.revision = state.project.revision; }
  const task = await api('tasks/start', request);
  state.tasks.push(task); window.dispatchEvent(new Event('tasks-changed'));
  return task;
}
export function projectBusy() { return state.tasks.some(task => task.request.project === state.project?.manifest && ['queued','running','cancelling'].includes(task.status)); }
