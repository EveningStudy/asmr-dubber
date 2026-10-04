import { $, $$, state, api, esc, t, size, startTask, taskHTML, guard, notice } from './session.js';
import { saveParameters } from './forms.js';

let catalog=null;
export async function refreshModels() {catalog=await api('models/list');renderModels();}
export function renderModels() {
  if(!catalog)return;
  const gpu=catalog.hardware.gpu||t('处理器'),vram=catalog.hardware.vram_gb;
  $('#deviceFoot').innerHTML=`<b>${esc(gpu)}${vram?` · ${vram.toFixed(1)} GB`:''}</b>${t('磁盘可用 {size}',{size:size(catalog.free_bytes)})} · ${t('版本')} ${state.boot.version}`;
  $('#modelDevice').innerHTML=`<div><span>${t('显卡')}</span><b>${esc(gpu)}${vram?` · ${vram.toFixed(1)} GB`:''}</b></div><div><span>${t('磁盘')}</span><b>${size(catalog.free_bytes)}</b></div><div><span>${t('模型已占用')}</span><b>${size(catalog.used_bytes)}</b></div><div style="margin-left:auto"><button class="btn plain" id="importModels">${t('导入离线模型包')}</button></div>`;
  const groups=[...new Set(catalog.items.map(item=>item.group))];
  $('#modelGroups').innerHTML=groups.map(group=>`<div class="label"><b>${t(group)}</b></div><div class="card">${catalog.items.filter(item=>item.group===group).map(item=>{
    const task=[...state.tasks].reverse().find(task=>task.kind==='download'&&(task.request.model===item.id||task.request.model==='recommended'));
    const running=task&&['queued','running','cancelling'].includes(task.status);
    return `<div class="row mrow"><div class="grow"><div class="t">${esc(item.name)}${item.recommended?` <span class="tag rec">${t('推荐')}</span>`:''}</div><div class="d">${esc(t(item.description))}</div><div class="tags">${item.vram_gb?`<span class="tag">${t('显存')} ${item.vram_gb} GB</span>`:''}<span class="tag">${esc(t(item.state))}</span></div></div><div class="size num">${item.disk_gb?`${item.disk_gb} GB`:t('未知')}</div><div class="act">${running?`<button class="btn" data-cancel="${task.id}">${t('暂停')}</button>`:item.state==='ready'?`${t('已安装')}<button class="btn plain small" data-remove-model="${item.id}">${t('删除')}</button>`:`<button class="btn" data-download-model="${item.id}">${t(task&&['failed','cancelled','interrupted'].includes(task.status)?'继续':'下载')}</button>`}</div></div>`;
  }).join('')}</div>`).join('');
  $('#starterDescription').textContent=t('Parakeet 1.1B 负责识别，IndexTTS2 负责模仿原声音色配音。');
  $('#asrCheck').classList.toggle('ok',catalog.items.some(item=>item.group==='识别'&&item.state==='ready'));
  $$('[data-source]').forEach(button=>button.classList.toggle('on',button.dataset.source===state.boot.settings.download_source));
  $('#modelTasks').innerHTML=state.tasks.filter(task=>['download','import_models','repair'].includes(task.kind)).map(taskHTML).join('');
  $('#importModels').onclick=guard(async()=>{await startTask('import_models',{},false);notice(t('请把离线模型包放在程序目录的 model-packs 文件夹。'));});
}
export function initializeModels() {
  $('#starterBtn').onclick=guard(()=>startTask('download',{model:'recommended'},false));
  document.addEventListener('click',guard(async event=>{
    const download=event.target.closest('[data-download-model]');if(download){const prior=[...state.tasks].reverse().find(task=>task.kind==='download'&&task.request.model===download.dataset.downloadModel&&['failed','cancelled','interrupted'].includes(task.status));if(prior)await api('tasks/resume',{identifier:prior.id});else await startTask('download',{model:download.dataset.downloadModel},false);return;}
    const remove=event.target.closest('[data-remove-model]');if(remove){await api('models/remove',{model:remove.dataset.removeModel});await refreshModels();return;}
    const source=event.target.closest('[data-source]');if(source){await saveParameters({download_source:source.dataset.source},'def');renderModels();}
  }));
}
