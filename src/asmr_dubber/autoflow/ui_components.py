from __future__ import annotations

# ruff: noqa: E501

TRACK_LIST_TEMPLATE = r"""
<div class="af-sort-list" role="list" aria-label="本次将处理的音轨">
  {{#each value}}
  <article class="af-sort-item" draggable="true" data-item-id="{{id}}" role="listitem">
    <div class="af-drag-handle" title="拖动调整顺序" aria-hidden="true">⠿</div>
    <div class="af-item-body">
      <div class="af-item-heading">
        <strong><span class="af-position">{{position}}</span>. {{title}}</strong>
        <span class="af-badge">{{category}}</span>
      </div>
      <div class="af-path">{{path}}</div>
      <div class="af-track-controls">
        <label>
          <span>字幕文件</span>
          <select data-role="subtitle">
            {{#each transcript_choices}}
            <option value="{{value}}" data-language="{{language}}" {{#if selected}}selected{{/if}}>
              {{label}}
            </option>
            {{/each}}
          </select>
        </label>
        <label>
          <span>字幕语言</span>
          <select data-role="language" {{#if has_subtitle}}{{else}}disabled{{/if}}>
            {{#each language_choices}}
            <option value="{{value}}" {{#if selected}}selected{{/if}}>{{label}}</option>
            {{/each}}
          </select>
        </label>
      </div>
    </div>
    <div class="af-order-buttons" aria-label="调整音轨顺序">
      <button type="button" data-action="up" title="上移" {{#if can_move_up}}{{else}}disabled{{/if}}>↑</button>
      <button type="button" data-action="down" title="下移" {{#if can_move_down}}{{else}}disabled{{/if}}>↓</button>
    </div>
  </article>
  {{/each}}
  {{#if value.length}}{{else}}
  <div class="af-empty">扫描作品后，这里会列出本次要处理的音轨。</div>
  {{/if}}
</div>
"""

TRACK_LIST_CSS = r"""
.af-sort-list { display: grid; gap: .65rem; }
.af-sort-item {
  display: grid;
  grid-template-columns: 1.25rem minmax(0, 1fr) auto;
  gap: .7rem;
  align-items: start;
  padding: .8rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 9px;
  background: var(--block-background-fill);
  transition: border-color .15s ease, box-shadow .15s ease, opacity .15s ease;
}
.af-sort-item:hover { border-color: var(--color-accent); }
.af-sort-item.af-dragging { opacity: .55; box-shadow: 0 8px 24px rgba(0,0,0,.12); }
.af-drag-handle { cursor: grab; color: var(--body-text-color-subdued); font-size: 1.25rem; line-height: 1.4; }
.af-drag-handle:active { cursor: grabbing; }
.af-item-body { min-width: 0; }
.af-item-heading { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.af-position { color: var(--color-accent); }
.af-badge {
  padding: .1rem .4rem;
  border-radius: 999px;
  background: var(--background-fill-secondary);
  color: var(--body-text-color-subdued);
  font-size: .78rem;
}
.af-path { margin-top: .2rem; color: var(--body-text-color-subdued); overflow-wrap: anywhere; }
.af-track-controls { display: grid; grid-template-columns: minmax(0, 2fr) minmax(7rem, 1fr); gap: .65rem; margin-top: .65rem; }
.af-track-controls label { display: grid; gap: .25rem; color: var(--body-text-color-subdued); font-size: .82rem; }
.af-track-controls select {
  width: 100%;
  min-height: 2.35rem;
  padding: .35rem .5rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 7px;
  background: var(--input-background-fill);
  color: var(--body-text-color);
}
.af-order-buttons { display: grid; gap: .3rem; }
.af-order-buttons button {
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 6px;
  background: var(--button-secondary-background-fill);
  color: var(--body-text-color);
}
.af-order-buttons button:disabled { opacity: .35; cursor: default; }
.af-empty { padding: 1rem; color: var(--body-text-color-subdued); text-align: center; }
@media (max-width: 640px) {
  .af-sort-item { grid-template-columns: 1rem minmax(0, 1fr); }
  .af-order-buttons { grid-column: 2; grid-template-columns: repeat(2, 2rem); }
  .af-track-controls { grid-template-columns: 1fr; }
}
"""

TRACK_LIST_JS = r"""
let draggedItem = null;
const itemSelector = '.af-sort-item';
const emitOrder = (list) => {
  const order = Array.from(list.querySelectorAll(itemSelector)).map((item) => item.dataset.itemId);
  trigger('track_reorder', {order});
};
const bindTrackList = () => {
  const list = element.querySelector('.af-sort-list');
  if (!list) return;
  list.ondragstart = (event) => {
    if (event.target.closest('button, select')) {
      event.preventDefault();
      return;
    }
    const item = event.target.closest(itemSelector);
    if (!item || !list.contains(item)) return;
    draggedItem = item;
    item.classList.add('af-dragging');
    if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
  };
  list.ondragover = (event) => {
    const item = event.target.closest(itemSelector);
    if (!draggedItem || !item || draggedItem === item) return;
    event.preventDefault();
    const rect = item.getBoundingClientRect();
    const after = event.clientY > rect.top + rect.height / 2;
    list.insertBefore(draggedItem, after ? item.nextSibling : item);
  };
  list.ondrop = (event) => {
    if (!draggedItem) return;
    event.preventDefault();
    emitOrder(list);
  };
  list.ondragend = () => {
    draggedItem?.classList.remove('af-dragging');
    draggedItem = null;
  };
  list.onclick = (event) => {
    const button = event.target.closest('[data-action]');
    const item = button?.closest(itemSelector);
    if (!button || !item || !list.contains(item)) return;
    const action = button.dataset.action;
    if (action === 'up' && item.previousElementSibling) {
      list.insertBefore(item, item.previousElementSibling);
      emitOrder(list);
    } else if (action === 'down' && item.nextElementSibling) {
      list.insertBefore(item.nextElementSibling, item);
      emitOrder(list);
    }
  };
  list.onchange = (event) => {
    const control = event.target.closest('[data-role]');
    const item = control?.closest(itemSelector);
    if (!control || !item || !list.contains(item)) return;
    const subtitle = item.querySelector('[data-role="subtitle"]');
    const language = item.querySelector('[data-role="language"]');
    if (control.dataset.role === 'subtitle') {
      const selected = subtitle.selectedOptions[0];
      language.disabled = !subtitle.value;
      if (subtitle.value && selected?.dataset.language) language.value = selected.dataset.language;
    }
    trigger('track_subtitle', {
      track_id: item.dataset.itemId,
      transcript: subtitle.value,
      language: language.value
    });
  };
};
bindTrackList();
watch('value', bindTrackList);
"""


QUEUE_LIST_TEMPLATE = r"""
<div class="af-queue-list" role="list" aria-label="处理队列">
  {{#each value}}
  <article class="af-queue-item {{#if reference_ready}}af-awaiting-reference{{/if}}" draggable="true" data-plan-id="{{id}}" data-reference-request-id="{{reference_request_id}}" role="listitem">
    <div class="af-drag-handle" title="拖动调整顺序" aria-hidden="true">⠿</div>
    <div class="af-queue-body">
      <div class="af-queue-heading">
        <strong><span class="af-position">{{position}}</span>. {{work}}</strong>
        {{#if rebuild}}<span class="af-warning-badge">将重新处理</span>{{/if}}
        {{#if reference_ready}}<span class="af-reference-badge">等待参考音频</span>{{/if}}
      </div>
      <div class="af-queue-meta">{{tracks}} 条音轨 · {{mode}} · {{layout}} · {{titles}}</div>
      <div class="af-path">{{output}}</div>
      {{#if reference_status}}<div class="af-reference-status">{{reference_status}}</div>{{/if}}
      <div class="af-queue-actions">
        <button type="button" data-action="edit">编辑选项</button>
        <button type="button" data-action="restart">{{rebuild_label}}</button>
        <button type="button" data-action="remove" class="af-danger">移除</button>
      </div>
    </div>
    <div class="af-order-buttons" aria-label="调整队列顺序">
      <button type="button" data-action="up" title="上移" {{#if can_move_up}}{{else}}disabled{{/if}}>↑</button>
      <button type="button" data-action="down" title="下移" {{#if can_move_down}}{{else}}disabled{{/if}}>↓</button>
    </div>
  </article>
  {{/each}}
  {{#if value.length}}{{else}}
  <div class="af-empty">队列为空。扫描作品并确认选项后，可以依次加入多个任务。</div>
  {{/if}}
</div>
"""

QUEUE_LIST_CSS = (
    TRACK_LIST_CSS
    + r"""
.af-queue-list { display: grid; gap: .65rem; }
.af-queue-item {
  display: grid;
  grid-template-columns: 1.25rem minmax(0, 1fr) auto;
  gap: .7rem;
  align-items: start;
  padding: .8rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 9px;
  background: var(--block-background-fill);
  transition: border-color .15s ease, box-shadow .15s ease, opacity .15s ease;
}
.af-queue-item:hover { border-color: var(--color-accent); }
.af-queue-item.af-dragging { opacity: .55; box-shadow: 0 8px 24px rgba(0,0,0,.12); }
.af-queue-body { min-width: 0; }
.af-queue-heading { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.af-queue-meta { margin-top: .25rem; color: var(--body-text-color-subdued); }
.af-warning-badge { padding: .1rem .45rem; border-radius: 999px; background: #fff0d6; color: #8a4b00; font-size: .78rem; }
.af-reference-badge { padding: .1rem .45rem; border-radius: 999px; background: #e8f2ff; color: #175cd3; font-size: .78rem; }
.af-queue-item.af-awaiting-reference { border-color: color-mix(in srgb, var(--color-accent) 65%, var(--border-color-primary)); }
.af-reference-status { margin-top: .55rem; padding: .55rem .65rem; border-radius: 7px; background: var(--background-fill-secondary); color: var(--body-text-color); }
.af-reference-dialog { max-width: min(30rem, calc(100vw - 2rem)); border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 1.1rem; background: var(--block-background-fill); color: var(--body-text-color); box-shadow: 0 18px 60px rgba(0,0,0,.24); }
.af-reference-dialog::backdrop { background: rgba(0,0,0,.36); }
.af-reference-dialog h3 { margin: 0 0 .45rem; font-size: 1.05rem; }
.af-reference-dialog p { margin: 0; color: var(--body-text-color-subdued); line-height: 1.55; }
.af-reference-dialog-actions { display: flex; justify-content: flex-end; gap: .55rem; margin-top: 1rem; }
.af-reference-dialog-actions button { min-height: 2.2rem; padding: .35rem .8rem; border: 1px solid var(--border-color-primary); border-radius: 7px; }
.af-reference-dialog-actions .af-reference-primary { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); }
.af-queue-actions { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .65rem; }
.af-queue-actions button {
  min-height: 2.1rem;
  padding: .3rem .7rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 7px;
  background: var(--button-secondary-background-fill);
  color: var(--body-text-color);
}
.af-queue-actions .af-danger { color: #b42318; }
@media (max-width: 640px) {
  .af-queue-item { grid-template-columns: 1rem minmax(0, 1fr); }
  .af-queue-item > .af-order-buttons { grid-column: 2; grid-template-columns: repeat(2, 2rem); }
}
"""
)

QUEUE_LIST_JS = r"""
let draggedQueueItem = null;
let lastReferenceNotice = '';
const queueItemSelector = '.af-queue-item';
const emitQueueOrder = (list) => {
  const order = Array.from(list.querySelectorAll(queueItemSelector)).map((item) => item.dataset.planId);
  trigger('queue_reorder', {order});
};
const bindQueueList = () => {
  const list = element.querySelector('.af-queue-list');
  if (!list) return;
  list.ondragstart = (event) => {
    if (event.target.closest('button')) {
      event.preventDefault();
      return;
    }
    const item = event.target.closest(queueItemSelector);
    if (!item || !list.contains(item)) return;
    draggedQueueItem = item;
    item.classList.add('af-dragging');
    if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
  };
  list.ondragover = (event) => {
    const item = event.target.closest(queueItemSelector);
    if (!draggedQueueItem || !item || draggedQueueItem === item) return;
    event.preventDefault();
    const rect = item.getBoundingClientRect();
    const after = event.clientY > rect.top + rect.height / 2;
    list.insertBefore(draggedQueueItem, after ? item.nextSibling : item);
  };
  list.ondrop = (event) => {
    if (!draggedQueueItem) return;
    event.preventDefault();
    emitQueueOrder(list);
  };
  list.ondragend = () => {
    draggedQueueItem?.classList.remove('af-dragging');
    draggedQueueItem = null;
  };
  list.onclick = (event) => {
    const button = event.target.closest('[data-action]');
    const item = button?.closest(queueItemSelector);
    if (!button || !item || !list.contains(item)) return;
    const action = button.dataset.action;
    if (action === 'up' && item.previousElementSibling) {
      list.insertBefore(item, item.previousElementSibling);
      emitQueueOrder(list);
    } else if (action === 'down' && item.nextElementSibling) {
      list.insertBefore(item.nextElementSibling, item);
      emitQueueOrder(list);
    } else if (action === 'edit') {
      trigger('queue_edit', {plan_id: item.dataset.planId});
      setTimeout(() => element.ownerDocument.querySelector('#autoflow-options')?.scrollIntoView({behavior: 'smooth'}), 250);
    } else if (action === 'restart') {
      trigger('queue_restart', {plan_id: item.dataset.planId});
    } else if (action === 'remove') {
      trigger('queue_remove', {plan_id: item.dataset.planId});
    }
  };
  try {
    const pending = Array.from(list.querySelectorAll(queueItemSelector)).find(
      (item) => Boolean(item.dataset.referenceRequestId)
    );
    const requestId = pending?.dataset.referenceRequestId || '';
    if (requestId && requestId !== lastReferenceNotice) {
      lastReferenceNotice = requestId;
      const doc = element.ownerDocument;
      let dialog = doc.getElementById('autoflow-reference-dialog');
      if (!dialog) {
        dialog = doc.createElement('dialog');
        dialog.id = 'autoflow-reference-dialog';
        dialog.className = 'af-reference-dialog';
        doc.body.appendChild(dialog);
      }
      const heading = doc.createElement('h3');
      heading.textContent = '可以选择参考音频了';
      const message = doc.createElement('p');
      const work = pending.querySelector('.af-queue-heading strong')?.textContent?.trim() || '当前作品';
      message.textContent = `${work} 已完成识别和翻译。你可以选择项目片段或导入外部音频；不操作会按设置自动继续。`;
      const actions = doc.createElement('div');
      actions.className = 'af-reference-dialog-actions';
      const later = doc.createElement('button');
      later.type = 'button';
      later.textContent = '暂不选择';
      later.onclick = () => dialog.close();
      const choose = doc.createElement('button');
      choose.type = 'button';
      choose.className = 'af-reference-primary';
      choose.textContent = '前往选择';
      choose.onclick = () => {
        dialog.close();
        const panel = doc.querySelector('#autoflow-reference-panel');
        panel?.scrollIntoView({behavior: 'smooth', block: 'center'});
        const toggle = panel?.querySelector('button[aria-expanded="false"]');
        try { toggle?.click(); } catch (_error) { /* visible queue state remains available */ }
      };
      actions.append(later, choose);
      dialog.replaceChildren(heading, message, actions);
      try {
        if (typeof dialog.showModal === 'function') dialog.showModal();
      } catch (_error) {
        // Browser dialogs are best-effort. The queue card and selector remain
        // visible, and the backend timeout continues independently.
      }
    }
  } catch (_error) {
    // UI notification failures must never affect the running task.
  }
};
bindQueueList();
watch('value', bindQueueList);
"""
