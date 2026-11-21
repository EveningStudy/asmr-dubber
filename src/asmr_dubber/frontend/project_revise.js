import { needsCheck } from './project_review.js';
import { $, state, api, t, esc, field, dialog, guard, notice, startTask, flushSaves, projectBusy } from './session.js';

// The sentence table's search box and filter buttons decide which rows are visible.
export function matchesRowFilter(sentence) {
  const query = $('#search').value.toLowerCase();
  const spk = state.speakerFilter || 'all';
  const spkMatch =
    spk === 'all' || (spk === '__none__' ? !sentence.speaker_id : sentence.speaker_id === spk);
  return (
    spkMatch &&
    `${sentence.id} ${sentence.speaker_id ?? ''} ${sentence.source_text} ${sentence.target_text ?? ''}`
      .toLowerCase()
      .includes(query) &&
    (state.filter === 'all' ||
      (state.filter === 'translation' && sentence.enabled && !sentence.target_text) ||
      (state.filter === 'tts' && sentence.enabled && !sentence.tts_file) ||
      (state.filter === 'review' && sentence.script_review_note) ||
      (state.filter === 'check' && needsCheck(sentence)))
  );
}

// Only enabled sentences that already have a translation can be revised.
const revisable = (sentence) => sentence.enabled && sentence.target_text;

function scopes(selectedId) {
  const sentences = state.project.sentences.filter(revisable);
  return [
    ['selected', '当前选中的句子', sentences.filter((s) => s.id === selectedId)],
    ['filtered', '当前表格筛选结果', sentences.filter(matchesRowFilter)],
    ['all', '全部已翻译的句子', sentences],
  ];
}

function reviseDialog(selectedId) {
  const options = scopes(selectedId);
  const first = options.find(([, , items]) => items.length)?.[0] || 'selected';
  const choices = options
    .map(
      ([value, label, items]) =>
        `<label class="choice"><input type="radio" name="reviseScope" value="${value}"${value === first ? ' checked' : ''}><div>${t(label)} · ${items.length}</div></label>`,
    )
    .join('');
  dialog(
    '按要求改译',
    `<p>${t('把选中的句子连同你的要求发给翻译模型。改动不会直接生效，会列成“旧译文 → 新译文”，由你逐句接受或拒绝。')}</p>` +
      `<div class="card group">${choices}</div>` +
      `<div class="card group"><div class="field col"><div class="fl">${t('你的要求')}</div><textarea id="reviseInstruction" rows="4" maxlength="2000" data-placeholder="例如：不要出现“亲爱的”；语气更温柔一点"></textarea></div>` +
      `${field('也包括我手动改过的句子', '<input type="checkbox" id="reviseEdited">')}</div>`,
    '开始改译',
    async () => {
      const instruction = $('#reviseInstruction').value.trim();
      if (!instruction) throw new Error(t('请先写下改译要求。'));
      const value = $('input[name="reviseScope"]:checked').value;
      const ids = options.find(([name]) => name === value)[2].map((s) => s.id);
      if (!ids.length) throw new Error(t('这个范围里没有已翻译的句子。'));
      await startTask('revise_translations', {
        sentence_ids: ids,
        instruction,
        include_edited: $('#reviseEdited').checked,
      });
    },
  );
}

export function renderProposals(project) {
  const box = $('#revisionProposals');
  const proposals = project.translation_proposals || [];
  $('#reviseTranslations').disabled = projectBusy();
  box.hidden = !proposals.length;
  if (box.hidden) return;
  const sources = new Map(project.sentences.map((s) => [s.id, s.source_text]));
  const disabled = projectBusy() ? ' disabled' : '';
  const items = proposals
    .map(
      (p) =>
        `<li title="${esc(p.instruction)}"><div class="head"><button class="btn plain small" data-locate="${esc(p.id)}">${esc(p.id)}</button><span class="src">${esc(sources.get(p.id) || '')}</span></div>` +
        `<div class="old"><small>${t('旧译文')}</small><del>${esc(p.old_zh)}</del></div>` +
        `<div class="new"><small>${t('新译文')}</small>${p.kind === 'sound' ? `<i>${t('不配音')}</i>` : esc(p.new_zh)}</div>` +
        `<div class="actions"><button class="btn small" data-proposal="accept" data-id="${esc(p.id)}"${disabled}>${t('接受')}</button><button class="btn plain small" data-proposal="reject" data-id="${esc(p.id)}"${disabled}>${t('拒绝')}</button></div></li>`,
    )
    .join('');
  box.innerHTML =
    `<div class="t">${t('改译提议（{count}）', { count: proposals.length })}</div>` +
    `<div class="actions"><button class="btn primary small" data-proposal-all="accept"${disabled}>${t('全部接受')}</button><button class="btn small" data-proposal-all="reject"${disabled}>${t('全部拒绝')}</button></div>` +
    `<ul class="proposals">${items}</ul>`;
}

async function resolve(accept, reject) {
  await flushSaves();
  const result = await api('projects/proposals', {
    project: state.project.manifest,
    revision: state.project.revision,
    accept,
    reject,
  });
  state.project = result;
  window.dispatchEvent(new Event('project-changed'));
  if (result.proposal_result.stale)
    notice(t('{count} 条提议因为译文已改动而被丢弃，没有应用。', { count: result.proposal_result.stale }), true);
}

export function initializeRevise(getSelected) {
  $('#reviseTranslations').onclick = guard(() => {
    if (projectBusy()) throw new Error(t('请等当前任务结束后再操作。'));
    reviseDialog(getSelected());
  });
  $('#revisionProposals').addEventListener(
    'click',
    guard(async (event) => {
      const all = event.target.closest('[data-proposal-all]');
      const one = event.target.closest('[data-proposal]');
      if (!all && !one) return;
      const ids = all ? state.project.translation_proposals.map((p) => p.id) : [one.dataset.id];
      const accepting = (all || one).dataset[all ? 'proposalAll' : 'proposal'] === 'accept';
      await resolve(accepting ? ids : [], accepting ? [] : ids);
    }),
  );
}
