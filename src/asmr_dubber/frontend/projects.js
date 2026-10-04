import { $, $$, state, api, esc, t, time, options, field, dialog, chooseFile, upload, enqueueSave, flushSaves, startTask, guard, notice, projectBusy } from './session.js';
import { renderForms } from './forms.js';

let selected = '', rowPage = 0, importAfterCreate = null, pendingSource = null;
export async function refreshHome() {
  const projects = await api('projects/list');
  $('#firstSteps').hidden = projects.length > 0;
  $('#projectsRoot').textContent = state.boot.settings.projects_root;
  $('#recentProjects').innerHTML = projects.map(project => `<button class="row project-row" data-open="${esc(project.manifest)}"><div class="grow"><div class="t">${esc(project.label)}</div><div class="d">${t(project.source_language)} · ${time(project.duration)}</div></div><div class="p4"><i class="${project.sentences?'d':''}"></i><i class="${project.translated?'d':''}"></i><i class="${project.synthesized?'d':''}"></i><i class="${project.exported?'d':''}"></i></div><div class="stage">${project.exported?t('已导出'):t('配音 {done} / {total} 句',{done:project.synthesized,total:project.sentences})}</div><div class="when">${esc(project.updated_at.slice(0,10))}</div></button>`).join('') || `<div class="row muted">${t('还没有项目')}</div>`;
}
export async function openProject(manifest) {
  await flushSaves();
  state.project = await api('projects/get',{project:manifest});
  state.filter='all'; rowPage=0; selected = state.project.sentences[0]?.id || '';
  $('#sourcePlayer').src = state.project.source_url;
  $('#seek').max = state.project.source.duration_seconds;
  window.dispatchEvent(new CustomEvent('navigate',{detail:{page:'project'}}));
  renderProject();
}
export function renderProject() {
  if (!state.project) return;
  const project = state.project, enabled = project.sentences.filter(sentence=>sentence.enabled);
  const translated = enabled.filter(sentence=>sentence.zh_text).length, voiced=enabled.filter(sentence=>sentence.tts_file).length;
  $('#projectName').textContent = project.title;
  $('#projectMeta').textContent = `${t(project.source_language)} → ${t(project.settings.tts_target_language)} · ${time(project.source.duration_seconds)} · ${t('{count} 句',{count:project.sentences.length})}`;
  $('#asrStat').textContent = t('已识别 {count} 句。重新识别会替换现有句子，译文和配音要重做。',{count:project.sentences.length});
  $('#translateRemaining').textContent=t('翻译剩余 {count} 句',{count:enabled.length-translated});
  $('#translationStat').textContent=t('已翻译 {done} / {total} 句。',{done:translated,total:enabled.length});
  $('#synthesizeRemaining').textContent=t('生成剩余 {count} 句',{count:enabled.filter(s=>s.zh_text&&!s.tts_file).length});
  $('#dubStat').textContent=t('已完成 {done} / {total} 句。只生成还没有配音、或译文改过的句子。',{done:voiced,total:enabled.length});
  $('#dubBar').style.width=`${enabled.length?voiced/enabled.length*100:0}%`;
  $('#exportStat').textContent=t('还有 {count} 句没有配音，成品里这些句子只有原声。',{count:enabled.length-voiced});
  $('#referenceLabel').textContent = `${t('音色参考')} · ${project.reference_external?t('外部音频'):project.reference_selected||t('未选择')}`;
  $('#projectDiagnostics').textContent=project.diagnostics;
  $('#outputs').innerHTML=Object.entries(project.outputs).map(([key,item])=>`<div class="out"><span class="grow">${esc(item.name)}</span>${/audio|stem|output_file/.test(key)?`<audio controls preload="none" src="${item.url}"></audio>`:''}<a class="btn plain small" href="${item.url}" download="${esc(item.name)}">${t('显示')}</a></div>`).join('');
  $$('#steps button').forEach((button,index)=>button.classList.toggle('done',[project.sentences.length>0,translated===enabled.length&&enabled.length>0,voiced===enabled.length&&enabled.length>0,Object.keys(project.outputs).length>0][index]));
  $$('[data-task],#translateAll,#exportProject').forEach(button=>button.disabled=projectBusy());
  renderRows(); renderForms();
}
function renderRows() {
  if (!state.project) return;
  const query=$('#search').value.toLowerCase();
  const sentences=state.project.sentences.filter(s=>(`${s.source_text} ${s.zh_text}`.toLowerCase().includes(query)) && (state.filter==='all' || state.filter==='translation'&&!s.zh_text || state.filter==='tts'&&!s.tts_file || state.filter==='review'&&s.script_review_note));
  const page=sentences.slice(rowPage*50,(rowPage+1)*50);
  $('#rows').innerHTML=page.map(sentence=>`<tr data-sentence="${esc(sentence.id)}" class="${!sentence.enabled?'off':''} ${selected===sentence.id?'sel':''}"><td><input type="checkbox" data-row-field="enabled"${sentence.enabled?' checked':''} aria-label="${t('参与配音')}"></td><td class="id">${esc(sentence.id)}</td><td class="time">${time(sentence.start_seconds)} – ${time(sentence.end_seconds)}</td><td contenteditable="${!projectBusy()}" data-row-field="source_text">${esc(sentence.source_text)}</td><td contenteditable="${!projectBusy()}" data-row-field="zh_text">${esc(sentence.zh_text)}</td><td class="dub">${sentence.tts_url?`<button class="play" data-play="${esc(sentence.id)}" aria-label="${t('试听')}">▶</button>${Number(sentence.tts_duration_seconds||0).toFixed(1)} ${t('秒')}`:t(sentence.enabled?'未生成':'不配音')}</td></tr>`).join('');
  renderControls();
}
function renderControls() {
  const sentence=state.project.sentences.find(s=>s.id===selected);
  $('#sentenceControls').innerHTML=`<div class="tools"><button class="btn" id="addSentence">${t('新增句子')}</button>${sentence?`<button class="btn" id="deleteSentence">${t('删除句子')}</button><button class="btn plain" id="sentenceReference">${t('设为项目音色参考')}</button>`:''}<span class="saved">${state.project.sentences.length>50?`<button class="btn" id="prevRows">‹</button> ${rowPage+1} / ${Math.ceil(state.project.sentences.length/50)} <button class="btn" id="nextRows">›</button>`:''}</span></div>${sentence?`<details><summary>${t('所选句子')} · ${esc(sentence.id)}</summary><div class="card group">${field('开始（秒）',`<input type="number" step="any" data-detail="start_seconds" value="${sentence.start_seconds}">`)}${field('结束（秒）',`<input type="number" step="any" data-detail="end_seconds" value="${sentence.end_seconds}">`)}${field('原声开关（仅分离时）',`<select data-detail="original_audio_enabled">${options([['默认','default'],['开启','on'],['关闭','off']],sentence.original_audio_enabled===null?'default':sentence.original_audio_enabled?'on':'off')}</select>`)}${field('原声音量微调（dB）',`<input type="number" step="any" min="-60" max="12" data-detail="original_audio_gain_db" value="${sentence.original_audio_gain_db}">`)}${field('播放中文配音',`<input type="checkbox" data-detail="chinese_audio_enabled"${sentence.chinese_audio_enabled?' checked':''}>`)}${field('中文音量微调（dB）',`<input type="number" step="any" min="-60" max="12" data-detail="chinese_audio_gain_db" value="${sentence.chinese_audio_gain_db}">`)}</div><p class="stat">${esc(sentence.script_review_note||sentence.error||'')}</p></details>`:''}`;
}
function saveRows(only=null) {
  const rows=state.project.sentences.map(s=>[s.id,s.enabled,s.start_seconds,s.end_seconds,s.source_text,s.zh_text,s.original_audio_enabled===null?'default':s.original_audio_enabled?'on':'off',s.original_audio_gain_db,s.chinese_audio_enabled,s.chinese_audio_gain_db]);
  $('#saveState').textContent=t('正在保存…');
  return enqueueSave(async()=>{
    const result=await api(only?'projects/sentence':'projects/table',{project:state.project.manifest,...(only?{row:rows.find(row=>row[0]===only)}:{rows}),revision:state.project.revision});
    state.project.revision=result.revision; state.project.settings=result.settings; state.project.outputs=result.outputs;
  });
}
async function newSource(file) {
  notice(t('正在导入…')); pendingSource=await upload(file);
  const defaults=state.boot.settings;
  dialog('新建项目',`<div class="card group">${field('音频里说的是',`<select id="newLanguage">${options([['日语','ja'],['英语','en'],['中文','zh']],defaults.default_source_language)}</select>`)}${field('配音成',`<select id="newTarget">${options([['中文','zh'],['英文','en']],defaults.tts_target_language)}</select>`)}</div><div class="more">${t('文字从哪来')}</div><div class="card group">${[['自动识别','用识别模型听出原文','auto'],['我有原文字幕','跳过识别，直接翻译','source'],['我有中文字幕','跳过识别和翻译，直接配音','zh'],['我只有不带时间的台本','先识别出时间，再把台本文字对上去','script']].map(([title,hint,value])=>`<label class="choice"><input type="radio" name="textSource" value="${value}"${value==='auto'?' checked':''}><div>${t(title)}<small>${t(hint)}</small></div></label>`).join('')}</div>`,'创建',async()=>{
    importAfterCreate=$('input[name="textSource"]:checked').value;
    await startTask('create',{source:pendingSource.path,source_language:$('#newLanguage').value,target_language:$('#newTarget').value},false);
  },file.name);
}
export async function finishCreation(task) {
  await openProject(task.result.manifest);
  if (importAfterCreate && importAfterCreate!=='auto') {
    const kind=importAfterCreate; importAfterCreate=null;
    await transcriptDialog(kind==='zh'?'zh':'source',kind==='script'?'script_review':'estimate');
  } else importAfterCreate=null;
}
async function transcriptDialog(kind='source',timing='estimate') {
  let file=null;
  dialog('导入字幕或台本',`<div class="card group">${field('导入内容',`<select id="scriptKind">${options([['原文','source'],['译文','zh']],kind)}</select>`)}${field('选择文件',`<button class="btn" id="pickTranscript">${t('选择文件')}</button><span id="transcriptFilename"></span>`)}${field('纯台本的时间轴',`<select id="plainTiming">${options([['按台词长度估算（无需模型，之后手动校对）','estimate'],['Qwen3 时间对齐','qwen'],['先运行 ASR/翻译，再用大模型校对台本（推荐）','script_review']],timing)}</select>`)}</div><label>${t('粘贴台本')}<textarea id="pastedTranscript" rows="6"></textarea></label>`,'导入',async()=>{ await startTask('import',{file:file?.path||null,text:$('#pastedTranscript').value,script_kind:$('#scriptKind').value,timing:$('#plainTiming').value}); });
  $('#pickTranscript').onclick=guard(async()=>{const picked=await chooseFile('.srt,.vtt,.ass,.lrc,.txt');if(picked){file=await upload(picked);$('#transcriptFilename').textContent=picked.name;}});
  const timingLabels={estimate:'按台词长度估算（无需模型，之后手动校对）',qwen:'Qwen3 时间对齐',script_review:'先运行 ASR/翻译，再用大模型校对台本（推荐）'};
  const refreshTiming=()=>{const current=$('#plainTiming').value;const available=state.boot.choices.transcript_timing[$('#scriptKind').value];$('#plainTiming').innerHTML=options(available.map(value=>[timingLabels[value],value]),available.includes(current)?current:available[0]);};
  $('#scriptKind').onchange=refreshTiming;refreshTiming();
}
export async function referenceDialog(manifest=state.project.manifest, batch=false, preferred='') {
  const project=await api('projects/get',{project:manifest});
  let external=!preferred&&project.reference_external&&project.external_reference_url?{path:project.settings.tts_external_reference_audio,url:project.external_reference_url}:null;
  dialog(batch?'试听并选择':'音色参考',`<div class="card group">${field('项目内的句子',`<select id="referenceSentence">${options(project.reference_choices,preferred||project.reference_selected)}</select>`)}${field('开始（秒）','<input type="number" step="any" id="referenceStart">')}${field('结束（秒）','<input type="number" step="any" id="referenceEnd">')}</div><label>${t('参考文字')}<textarea id="referenceText"></textarea></label><audio id="referencePreview" controls preload="none" style="width:100%"></audio><div class="card group">${field('外部音频',`<button class="btn" id="pickReference">${t('选择文件')}</button><span id="referenceFileName"></span>`)}${field('外部音频语言',`<select id="referenceLanguage">${options([['跟随项目','auto'],['日语','ja'],['英语','en'],['中文','zh']],'auto')}</select>`)}</div>`,'采用',async()=>{
    const latest=await api('projects/get',{project:manifest});
    const result=await api('projects/reference',{project:manifest,revision:latest.revision,sentence:$('#referenceSentence').value,external:external?.path||'',text:$('#referenceText').value,language:$('#referenceLanguage').value,start:external?null:Number($('#referenceStart').value),end:external?null:Number($('#referenceEnd').value)});
    if(state.project?.manifest===manifest){state.project=result.project;renderProject();}
  });
  const previewPlayer=$('#referencePreview');let previewGeneration=0;
  async function preview() {
    const generation=++previewGeneration;
    const sentence=project.sentences.find(s=>s.id===$('#referenceSentence').value);if(!sentence)return;
    $('#referenceStart').value=sentence.start_seconds;$('#referenceEnd').value=sentence.end_seconds;$('#referenceText').value=sentence.source_text;
    const task=await startTask('preview_reference',{project:manifest,revision:project.revision,sentence:sentence.id},false);
    const interval=setInterval(guard(async()=>{if(!previewPlayer.isConnected||!$('#dialog').classList.contains('on')||generation!==previewGeneration){clearInterval(interval);return;}const latest=await api('tasks/get',{identifier:task.id});if(!['queued','running','cancelling'].includes(latest.status)){clearInterval(interval);if(latest.result?.url&&generation===previewGeneration&&previewPlayer.isConnected)previewPlayer.src=latest.result.url;}}),400);
  }
  $('#referenceSentence').onchange=guard(async()=>{external=null;$('#referenceFileName').textContent='';await preview();});
  $('#pickReference').onclick=guard(async()=>{const file=await chooseFile('audio/*');if(file){external=await upload(file);$('#referenceFileName').textContent=file.name;$('#referencePreview').src=external.url;}});
  if(external){$('#referencePreview').src=external.url;$('#referenceFileName').textContent=external.path.split(/[\\/]/).pop();$('#referenceText').value=project.settings.tts_external_reference_text;$('#referenceLanguage').value=project.settings.tts_external_reference_language;}else await preview();
}
export async function reviewDialog() {
  const overview=await api('review/get',{project:state.project.manifest});
  dialog('音频复核提案：试听、采纳与撤销',`<p>${esc(overview.status)}</p><div class="card group">${field('复核音频片段',`<select id="reviewWindow">${options(overview.choices,overview.choices[0]?.[1])}</select>`)}${field('候选文字','<select id="reviewCandidate"></select>')}</div><div id="reviewDiff"></div><audio controls id="reviewAudio" style="width:100%"></audio><div class="review-actions">${[['accept_review','采纳所选候选'],['keep_review','保留并确认主稿'],['undo_review','撤销上一次采纳或确认'],['unlock_review','解除人工确认锁定（允许重新识别）'],['retry_review','只重试音频复核（不重跑完整主 ASR）'],['align_review','文字确认后：仅重新对齐时间（Qwen3）']].map(([kind,label])=>`<button class="btn" data-review-task="${kind}">${t(label)}</button>`).join('')}</div>`,'',null);
  async function details() {if(!$('#reviewWindow').value)return;const result=await api('review/get',{project:state.project.manifest,window:$('#reviewWindow').value,candidate:$('#reviewCandidate').value}); const choice=$('#reviewCandidate').value;$('#reviewCandidate').innerHTML=options(result.candidates,choice);$('#reviewDiff').innerHTML=result.diff;$('#reviewAudio').src=result.audio;}
  $('#reviewWindow').onchange=guard(async()=>{$('#reviewCandidate').innerHTML='';await details();});$('#reviewCandidate').onchange=guard(details);
  $('#reviewDiff').insertAdjacentHTML('beforebegin',`<p class="stat">${t('实验性，效果可能不如单模型')} · ${t('普通单模型识别不需要使用')}</p>`);
  await details();
}
export function initializeProjects() {
  $('#dropNew').onclick=()=>$('#mediaFile').click();$('#dropNew').onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('#mediaFile').click();}};
  $('#mediaFile').onchange=guard(async event=>{if(event.target.files[0])await newSource(event.target.files[0]);event.target.value='';});
  $('#dropNew').ondragover=event=>{event.preventDefault();};$('#dropNew').ondrop=guard(async event=>{event.preventDefault();if(event.dataTransfer.files[0])await newSource(event.dataTransfer.files[0]);});
  $('#openExisting').onclick=guard(async()=>{dialog('打开项目',field('项目路径','<input type="text" id="existingManifest">'),'打开项目',async()=>openProject($('#existingManifest').value));});
  $('#search').oninput=()=>{rowPage=0;renderRows();};$('#importTranscript').onclick=guard(()=>transcriptDialog());$('#changeReference').onclick=guard(()=>referenceDialog());$('#showReview').onclick=guard(reviewDialog);
  $('#openProjectFolder').onclick=guard(()=>api('projects/open',{project:state.project.manifest}));
  $('#translateAll').onclick=guard(()=>startTask('translate',{force:true}));$('#exportProject').onclick=guard(()=>startTask('export',{audio:$('#exportAudio').value==='true',language:$('#exportSubtitle').value}));
  document.addEventListener('click',guard(async event=>{
    const open=event.target.closest('[data-open]');if(open){await openProject(open.dataset.open);return;}
    const task=event.target.closest('[data-task]');if(task){await startTask(task.dataset.task);return;}
    const review=event.target.closest('[data-review-task]');if(review){await startTask(review.dataset.reviewTask,{window:$('#reviewWindow').value,candidate:$('#reviewCandidate').value});$('#dialog').classList.remove('on');return;}
    const filter=event.target.closest('[data-filter]');if(filter){state.filter=filter.dataset.filter;rowPage=0;$$('[data-filter]').forEach(b=>b.classList.toggle('on',b===filter));renderRows();return;}
    const play=event.target.closest('[data-play]');if(play){const sentence=state.project.sentences.find(s=>s.id===play.dataset.play);$('#dubPlayer').src=sentence.tts_url;await $('#dubPlayer').play();return;}
    if(event.target.id==='prevRows'){rowPage=Math.max(0,rowPage-1);renderRows();return;}if(event.target.id==='nextRows'){rowPage++;renderRows();return;}
    if(event.target.id==='sentenceReference'){await referenceDialog(state.project.manifest,false,selected);return;}
    if(event.target.id==='addSentence'){await flushSaves();const last=state.project.sentences.at(-1);const start=last?.end_seconds||0;const end=Math.min(state.project.source.duration_seconds,start+1);if(end<=start)throw new Error(t('媒体末尾没有剩余空间；请先调整已有句子的结束时间，再添加句子。'));selected=`s${Date.now()}`;state.project.sentences.push({id:selected,start_seconds:start,end_seconds:end,source_text:'',zh_text:t('新句子'),enabled:true,original_audio_enabled:null,original_audio_gain_db:0,chinese_audio_enabled:true,chinese_audio_gain_db:0});await saveRows();renderRows();return;}
    if(event.target.id==='deleteSentence'){state.project.sentences=state.project.sentences.filter(s=>s.id!==selected);selected=state.project.sentences[0]?.id||'';await saveRows();renderRows();return;}
    const row=event.target.closest('tr[data-sentence]');if(row&&selected!==row.dataset.sentence){selected=row.dataset.sentence;const sentence=state.project.sentences.find(s=>s.id===selected);$('#sourcePlayer').currentTime=sentence.start_seconds;$$('#rows tr').forEach(r=>r.classList.toggle('sel',r===row));renderControls();}
  }));
  document.addEventListener('focusout',guard(async event=>{const key=event.target.dataset.rowField;if(!key||event.target.type==='checkbox')return;const sentence=state.project.sentences.find(s=>s.id===event.target.closest('tr').dataset.sentence);if(sentence[key]!==event.target.textContent){sentence[key]=event.target.textContent;sentence.tts_file=null;sentence.tts_url=null;await saveRows(sentence.id);}}));
  document.addEventListener('change',guard(async event=>{const node=event.target,key=node.dataset.detail||node.dataset.rowField;if(!key)return;const sentence=state.project.sentences.find(s=>s.id===(node.dataset.detail?selected:node.closest('tr').dataset.sentence));sentence[key]=node.type==='checkbox'?node.checked:key==='original_audio_enabled'?({default:null,on:true,off:false}[node.value]):Number(node.value);await saveRows(sentence.id);}));
  const source=$('#sourcePlayer'),dub=$('#dubPlayer');let mode='both';
  $('#playSource').onclick=guard(async()=>{if(!source.paused||!dub.paused){source.pause();dub.pause();$('#playSource').textContent='▶';return;}if(mode!=='dub')await source.play();if(mode!=='source'){const sentence=state.project.sentences.find(s=>s.id===selected);const stem=state.project.outputs.chinese_stem_file;if(stem){dub.src=stem.url;dub.currentTime=source.currentTime;}else if(sentence?.tts_url){dub.src=sentence.tts_url;}if(dub.src)await dub.play();}$('#playSource').textContent='Ⅱ';});
  source.ontimeupdate=()=>{$('#playerTime').textContent=`${time(source.currentTime)} / ${time(source.duration)}`;$('#seek').value=source.currentTime;};source.onended=()=>{$('#playSource').textContent='▶';};$('#seek').oninput=()=>{source.currentTime=Number($('#seek').value);if(state.project.outputs.chinese_stem_file)dub.currentTime=source.currentTime;};
  $('#listenMode').onclick=event=>{const button=event.target.closest('[data-listen]');if(button){mode=button.dataset.listen;source.pause();dub.pause();$$('[data-listen]').forEach(b=>b.classList.toggle('on',b===button));$('#playSource').textContent='▶';}};
}
