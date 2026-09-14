"""Bounded native sentence editor; full drafts travel only on explicit actions."""

# Embedded HTML/CSS/JS follows the same formatting convention as AutoFlow UI assets.
# ruff: noqa: E501

TEMPLATE = """
<section aria-label="句子校对表格" class="sentence-editor">
  <div class="editor-toolbar">
    <button type="button" data-action="prev">上一页</button>
    <span data-role="page" aria-live="polite"></span>
    <button type="button" data-action="next">下一页</button>
    <label>跳到页 <input data-role="jump" type="number" min="1" value="1" aria-label="跳到页"></label>
    <button type="button" data-action="jump">跳转</button>
    <label>句子 ID <input data-role="find" type="text" aria-label="查找句子 ID"></label>
    <button type="button" data-action="find">定位</button>
    <button type="button" data-action="add">添加句子</button>
  </div>
  <div class="editor-scroll">
    <table><colgroup><col style="width:100px"><col style="width:110px"><col style="width:100px"><col style="width:100px"><col style="width:350px"><col style="width:350px"></colgroup>
      <thead><tr><th>句子 ID</th><th>启用中文处理</th><th>开始（秒）</th><th>结束（秒）</th><th>原文</th><th>中文译文</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <p data-role="status" aria-live="polite">尚无句子。</p>
</section>
"""

CSS = """
.sentence-editor { min-width:0; }
.editor-toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:8px 0; }
.editor-toolbar label { display:flex; gap:4px; align-items:center; }
.editor-toolbar input { width:90px; }
.editor-scroll { overflow:auto; max-height:600px; }
table { table-layout:fixed; border-collapse:collapse; width:1110px; }
th { position:sticky; top:0; background:var(--block-background-fill); z-index:1; }
th, td { border:1px solid var(--border-color-primary); padding:5px; vertical-align:top; }
textarea, input { background:var(--input-background-fill); color:var(--body-text-color); border:1px solid var(--border-color-primary); border-radius:4px; padding:5px; }
td input:not([type=checkbox]), td textarea { width:100%; box-sizing:border-box; font:inherit; }
td input[type=checkbox] { appearance:auto !important; width:18px; height:18px; accent-color:var(--color-accent); }
textarea { height:64px; resize:vertical; white-space:pre-wrap; }
button { padding:6px 10px; border:1px solid var(--border-color-primary); border-radius:5px; background:var(--button-secondary-background-fill); color:var(--body-text-color); }
button:disabled { opacity:.4; }
[data-role=status] { color:var(--body-text-color-subdued); margin-top:6px; }
"""

JS = r"""
const size = 50;
let page = 0;
let rows = [];
const text = value => value === true ? 'true' : value === false ? 'false' : String(value ?? '');
const enabled = value => ['true','1','yes','是'].includes(text(value).trim().toLowerCase());
const status = message => { element.querySelector('[data-role=status]').textContent = message; };
const publish = () => {
  // The template is static: changing its value does not replace the active cell.
  // A shallow outer copy notifies Gradio's input binding, with no server event.
  props.value = rows.slice();
  status('编辑内容在当前页面。翻页和切换页签保留草稿；请点击“保存校对表格”，并查看上方保存结果。');
};
const render = () => {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.max(0, Math.min(page, pages - 1));
  element.querySelector('[data-role=page]').textContent = `第 ${page+1}/${pages} 页 · 共 ${rows.length} 句 · 每页 ${size} 句`;
  element.querySelector('[data-role=jump]').value = String(page+1);
  element.querySelector('[data-action=prev]').disabled = page === 0;
  element.querySelector('[data-action=next]').disabled = page === pages-1;
  const body = element.querySelector('tbody');
  body.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (let i=page*size; i<Math.min(rows.length,(page+1)*size); i++) {
    const tr = document.createElement('tr'); tr.dataset.row = String(i);
    for(let col=0; col<6; col++) {
      const td = document.createElement('td');
      if(col===0) { td.textContent = text(rows[i][col]); }
      else {
        const field = document.createElement(col>=4 ? 'textarea' : 'input');
        field.dataset.row = String(i); field.dataset.col = String(col);
        field.setAttribute('aria-label', `${text(rows[i][0])} ${['ID','启用中文处理','开始秒数','结束秒数','原文','中文译文'][col]}`);
        if(col===1) {field.type='checkbox';field.checked=enabled(rows[i][col]);}
        else {field.value=text(rows[i][col]);if(col<4)field.inputMode='decimal';}
        td.append(field);
      }
      tr.append(td);
    }
    fragment.append(tr);
  }
  body.append(fragment);
};
element.addEventListener('input', event => {
  const field=event.target;
  if(field.dataset.row === undefined || field.dataset.col === undefined)return;
  const i=Number(field.dataset.row), col=Number(field.dataset.col);
  if(!rows[i])return;
  rows[i]=rows[i].slice();
  rows[i][col]=col===1 ? (field.checked ? 'true':'false') : field.value;
  publish();
});
element.addEventListener('click', event => {
  const action=event.target.closest('button[data-action]')?.dataset.action;
  if(!action)return;
  if(action==='prev')page--;
  else if(action==='next')page++;
  else if(action==='jump') {
    const requested=Number(element.querySelector('[data-role=jump]').value);
    if(!Number.isInteger(requested)||requested<1) {status('请输入有效页码。');return;}
    page=requested-1;
  } else if(action==='find') {
    const query=element.querySelector('[data-role=find]').value.trim();
    const index=rows.findIndex(row=>text(row[0])===query);
    if(index<0){status('没有找到这个句子 ID。');return;}
    page=Math.floor(index/size);
  } else if(action==='add') {
    const used=new Set(rows.map(row=>text(row[0])));
    let number=rows.length+1, id;
    do {id='s'+String(number++).padStart(6,'0');} while(used.has(id));
    const start=rows.length ? Number(rows[rows.length-1][3])||0 : 0;
    rows.push([id,'true',String(start),String(start+1),'','']);
    page=Math.floor((rows.length-1)/size);publish();
  }
  render();
  element.querySelector('.editor-scroll').scrollTop=0;
});
const receive = () => {
  rows=(Array.isArray(props.value) ? props.value : []).map(row=>row.map(text));
  render();status(rows.length ? '已载入项目句子。修改后请保存。' : '尚无句子。');
};
receive();watch('value',receive);
"""
