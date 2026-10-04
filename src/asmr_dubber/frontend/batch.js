import { $, $$, state, api, esc, t, options, field, dialog, startTask, taskHTML, guard, notice } from './session.js';
import { referenceDialog } from './projects.js';

let scan=null, editing='', selected=new Set();
export async function refreshBatch() { state.queue=await api('batch/list');renderBatch(); }
export function renderBatch() {
  const active=state.tasks.find(task=>task.kind==='batch'&&['queued','running','cancelling'].includes(task.status));
  const history=state.tasks.filter(task=>task.kind==='batch'&&!['queued','running','cancelling'].includes(task.status));
  const completedPlans=new Set(history.filter(task=>task.status==='completed'&&task.result?.exit_code===0).flatMap(task=>task.result.plans||[]));
  $('#queueBadge').textContent=state.queue.length||'';
  $('#batchQueue').innerHTML=state.queue.length?`<div class="label"><b>${t('队列')}</b><span>${t('按顺序处理，可拖动调整')}</span></div><div class="card q">${state.queue.map((plan,index)=>`<div class="row" draggable="${!active}" data-plan-row="${plan.plan_id}"><div class="grow"><div class="t">${esc(plan.folder.split(/[\\/]/).pop())}</div><div class="d">${t('{count} 条音轨',{count:plan.sources.length})} · ${esc(t(plan.edition.output_policy?.content||'dubbing'))} · ${esc(t(plan.layout))}</div></div><div class="prog">${completedPlans.has(plan.plan_id)?t('已完成'):active?t('正在处理…'):t('等待中')}</div>${active?'':`<button class="btn" data-edit-plan="${plan.plan_id}">${t('编辑')}</button><button class="btn plain" data-remove-plan="${plan.plan_id}">${t('删除')}</button><button class="btn plain" data-plan-up="${index}"${index===0?' disabled':''}>↑</button>`}</div>`).join('')}</div>${active?taskHTML(active):''}`:`<div class="card empty"><b>${t('队列是空的')}</b><p>${t('添加一个已解压的作品文件夹开始。')}</p></div>`;
  $('#runBatch').disabled=!!active||!state.queue.length;
  const request=active?.reference;
  $('#batchReference').innerHTML=request?`<div class="card"><div class="row"><div class="grow"><div class="t">${esc(request.work||'')} · ${t('音色参考')}</div><div class="d">${t('等待时间结束后使用自动推荐的项目片段。')} ${request.deadline_epoch?Math.max(0,Math.ceil(request.deadline_epoch-Date.now()/1000)):''} ${t('秒')}</div></div><button class="btn" id="chooseBatchReference">${t('试听并选择')}</button></div></div>`:`<p class="muted">${t('没有需要确认的任务')}</p>`;
  if(request)$('#chooseBatchReference').onclick=guard(()=>referenceDialog(request.project_json,true));
  $('#batchHistory').innerHTML=history.map(task=>`${taskHTML(task)}${(task.result?.outputs||[]).map(path=>`<div class="row"><span class="grow">${esc(path)}</span><button class="btn" data-batch-output="${esc(path)}">${t('打开文件夹')}</button></div>`).join('')}`).join('');
}
function taskFields() {
  const defaults=state.boot.settings;
  return `<div class="card group">${field('成品',`<select id="batchContent">${options([['双语版','dubbing'],['替换版（实验性）','replacement'],['双语版 + 替换版','both'],['原声 + 字幕，不配音','source_subtitles'],['只要字幕文件','subtitles']],'dubbing')}</select>`)}${field('字幕',`<select id="batchSubtitle">${options([['双语','bilingual'],['仅译文','zh'],['仅原文','source']],defaults.autoflow_subtitle_language)}</select>`)}${field('格式',`<select id="batchMode">${options([['音频','audio'],['视频','video_normal'],['和谐视频','video_harmonized']],defaults.autoflow_default_mode)}</select>`)}${field('视频画面',`<select id="batchBackground"><option value="black">${t('黑色背景')}</option></select>`)}${field('把字幕烧进视频',`<input type="checkbox" id="batchEmbed"${defaults.autoflow_embed_subtitles?' checked':''}>`)}${field('多条音轨',`<select id="batchLayout">${options([['合并成一个','merged'],['每轨单独','separate'],['都要','both']],defaults.autoflow_default_layout)}</select>`)}${field('重新处理',`<input type="checkbox" id="batchRebuild">`)}</div>`;
}
async function workDialog(identifier='') {
  editing=identifier;scan=null;selected.clear();
  dialog(identifier?'编辑队列任务':'添加作品',`<div class="card group"><div class="field"><input type="text" id="workFolder" style="flex:1;max-width:none"><button class="btn" id="chooseWorkFolder">${t('选择文件夹')}</button><button class="btn" id="scanWork">${t('扫描作品')}</button></div>${field('音频版本','<select id="workEdition"></select>')}${field('包含特典、样本和 Free Talk',`<input type="checkbox" id="workBonus"${state.boot.settings.autoflow_include_bonus?' checked':''}>`)}</div><div class="more" id="trackSummary"></div><div class="card group" id="workTracks"></div><div class="more">${t('做成什么')}</div>${taskFields()}<p class="small muted">${t('识别、翻译、配音按“新项目默认值”来。')}</p>`,'加入队列',async()=>{
    if(!scan)throw new Error(t('请先扫描作品。'));
    const sources=scan.source_payloads.filter((_,index)=>selected.has(scan.track_items[index].id));
    await api('batch/save',{folder:scan.folder,edition:$('#workEdition').value,sources,mode:$('#batchMode').value,layout:$('#batchLayout').value,background:$('#batchBackground').value,embed_subtitles:$('#batchEmbed').checked,rebuild:$('#batchRebuild').checked,content:$('#batchContent').value,subtitle_language:$('#batchSubtitle').value,editing});
    await refreshBatch();
  });
  $('#scanWork').onclick=guard(async()=>{scan=await api('batch/scan',{folder:$('#workFolder').value,include_bonus:$('#workBonus').checked});selected=new Set(scan.track_items.map(track=>track.id));renderScan();});
  $('#chooseWorkFolder').onclick=guard(async()=>{const result=await api('files/folder');if(result.path){$('#workFolder').value=result.path;$('#scanWork').click();}});
  $('#workEdition').onchange=$('#workBonus').onchange=guard(async()=>{if(!scan)return;const view=await api('batch/edition',{folder:scan.folder,edition:$('#workEdition').value,include_bonus:$('#workBonus').checked});Object.assign(scan,view);selected=new Set(scan.track_items.map(track=>track.id));renderTracks();});
  if(identifier){
    scan=await api('batch/edit',{identifier});selected=new Set(scan.track_items.map(track=>track.id));renderScan();
    $('#workBonus').checked=scan.include_bonus;$('#batchMode').value=scan.mode;$('#batchLayout').value=scan.layout;$('#batchEmbed').checked=scan.embed_subtitles;$('#batchRebuild').checked=scan.rebuild;$('#batchSubtitle').value=scan.subtitle_language;
    const plan=state.queue.find(plan=>plan.plan_id===identifier);$('#batchContent').value=plan.edition.output_policy?.content||(scan.source_subtitles_only?'source_subtitles':scan.subtitles_only?'subtitles':'dubbing');
  }
}
function renderScan() {$('#workFolder').value=scan.folder;$('#workEdition').innerHTML=options(scan.edition_choices,scan.selected_edition);$('#batchBackground').innerHTML=options(scan.background_choices,scan.selected_background);renderTracks();}
function renderTracks() {
  $('#trackSummary').textContent=t('音轨 · 找到 {total} 条，已选 {selected} 条',{total:scan.track_items.length,selected:selected.size});
  $('#workTracks').innerHTML=scan.track_items.map((track,index)=>`<div class="track${selected.has(track.id)?'':' dim'}" data-track="${esc(track.id)}"><input type="checkbox" data-track-enabled="${esc(track.id)}"${selected.has(track.id)?' checked':''}><span>${esc(track.path)}<small class="muted">${esc(t(track.category))}</small></span><button class="btn plain" data-track-up="${index}"${index===0?' disabled':''}>↑</button><div><select data-track-subtitle="${esc(track.id)}">${options(track.transcript_choices.map(c=>[c.label,c.value]),track.transcript_choices.find(c=>c.selected)?.value||'')}</select><div class="track-extra"><select data-track-language="${esc(track.id)}">${options(track.language_choices.map(c=>[c.label,c.value]),track.language_choices.find(c=>c.selected)?.value)}</select><select data-track-mode="${esc(track.id)}">${options(track.timing_choices.filter(c=>!c.disabled).map(c=>[c.label,c.value]),track.timing_choices.find(c=>c.selected)?.value)}</select></div></div></div>`).join('');
}
export function initializeBatch() {
  $('#addWork').onclick=guard(()=>workDialog());$('#runBatch').onclick=guard(()=>startTask('batch',{},false));
  document.addEventListener('click',guard(async event=>{
    const edit=event.target.closest('[data-edit-plan]');if(edit){await workDialog(edit.dataset.editPlan);return;}
    const remove=event.target.closest('[data-remove-plan]');if(remove){await api('batch/remove',{identifier:remove.dataset.removePlan});await refreshBatch();return;}
    const up=event.target.closest('[data-plan-up]');if(up){const order=state.queue.map(plan=>plan.plan_id),index=Number(up.dataset.planUp);[order[index-1],order[index]]=[order[index],order[index-1]];await api('batch/reorder',{order});await refreshBatch();return;}
    const track=event.target.closest('[data-track-up]');if(track){const order=scan.track_items.map(track=>track.id),index=Number(track.dataset.trackUp);[order[index-1],order[index]]=[order[index],order[index-1]];Object.assign(scan,await api('batch/tracks',{folder:scan.folder,sources:scan.source_payloads,order}));renderTracks();return;}
    const output=event.target.closest('[data-batch-output]');if(output)await api('batch/open',{path:output.dataset.batchOutput});
  }));
  document.addEventListener('change',guard(async event=>{
    const node=event.target;
    if(node.dataset.trackEnabled){node.checked?selected.add(node.dataset.trackEnabled):selected.delete(node.dataset.trackEnabled);renderTracks();return;}
    const id=node.dataset.trackSubtitle||node.dataset.trackLanguage||node.dataset.trackMode;if(!id)return;
    const row=node.closest('[data-track]');const result=await api('batch/subtitle',{folder:scan.folder,sources:scan.source_payloads,track:id,file:$('[data-track-subtitle]',row).value,language:$('[data-track-language]',row).value,mode:$('[data-track-mode]',row).value});Object.assign(scan,result);renderTracks();
  }));
  let dragged='';
  $('#batchQueue').ondragstart=event=>{dragged=event.target.closest('[data-plan-row]')?.dataset.planRow||'';};
  $('#batchQueue').ondragover=event=>event.preventDefault();
  $('#batchQueue').ondrop=guard(async event=>{event.preventDefault();const target=event.target.closest('[data-plan-row]')?.dataset.planRow;if(!target||!dragged)return;const order=state.queue.map(plan=>plan.plan_id);order.splice(order.indexOf(dragged),1);order.splice(order.indexOf(target),0,dragged);await api('batch/reorder',{order});await refreshBatch();});
}
