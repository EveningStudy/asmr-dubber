import { $, $$, state, t, time, esc } from './session.js';

// Original, bilingual mix and full-replacement mix play together; only the chosen one is
// audible, so switching never loses the playing position.
const LABELS = { source: '原声', bilingual: '双语版', replacement: '完全配音版' };
const DRIFT_SECONDS = 0.12;
let signature = '',
  players = new Map(),
  active = 'source',
  playing = false,
  position = 0,
  loopedId = '',
  timer = null;

const master = () => players.get(active);
const selectedSentence = () => state.project?.sentences.find((s) => s.id === state.selectedSentence);

function stop() {
  playing = false;
  clearInterval(timer);
  players.forEach((audio) => audio.pause());
  $('#comparePlay').textContent = '▶';
}
function teardown() {
  stop();
  players.forEach((audio) => audio.removeAttribute('src'));
  players = new Map();
  position = 0;
}
function seek(seconds) {
  position = Math.max(0, seconds);
  players.forEach((audio) => {
    audio.currentTime = position;
  });
  showTime();
}
function showTime() {
  const total = Number($('#compareSeek').max) || 0;
  $('#compareTime').textContent = `${time(position)} / ${time(total)}`;
  $('#compareSeek').value = position;
}
function showLoopInfo() {
  const sentence = selectedSentence();
  $('#compareInfo').textContent = !$('#compareLoop').checked
    ? ''
    : sentence
      ? t('循环 {id}：{start} – {end}', {
          id: sentence.id,
          start: time(sentence.start_seconds),
          end: time(sentence.end_seconds),
        })
      : t('请先在句子表里选中一句。');
}
function tick() {
  const lead = master();
  if (!lead) return;
  position = lead.currentTime;
  const sentence = $('#compareLoop').checked ? selectedSentence() : null;
  if (sentence) {
    const fresh = loopedId !== sentence.id;
    loopedId = sentence.id;
    if (fresh || position >= sentence.end_seconds || position < sentence.start_seconds - 0.05) {
      seek(sentence.start_seconds);
      showLoopInfo();
      return;
    }
  }
  players.forEach((audio) => {
    if (audio !== lead && Math.abs(audio.currentTime - position) > DRIFT_SECONDS) audio.currentTime = position;
  });
  showTime();
  showLoopInfo();
  if (lead.ended) stop();
}
async function play() {
  // The editor's own players would otherwise play over the comparison.
  $('#sourcePlayer')?.pause();
  $('#dubPlayer')?.pause();
  $('#playSource').textContent = '▶';
  const sentence = $('#compareLoop').checked ? selectedSentence() : null;
  if (sentence && (position < sentence.start_seconds || position >= sentence.end_seconds))
    seek(sentence.start_seconds);
  players.forEach((audio, key) => {
    audio.muted = key !== active;
    audio.currentTime = position;
  });
  await Promise.all([...players.values()].map((audio) => audio.play()));
  playing = true;
  $('#comparePlay').textContent = 'Ⅱ';
  clearInterval(timer);
  timer = setInterval(tick, 80);
}
function renderButtons(tracks) {
  $('#compareTracks').innerHTML = tracks
    .map((track) => {
      const cls = [track.key === active && 'on', track.stale && 'stale'].filter(Boolean).join(' ');
      const tip = track.stale ? ` title="${esc(t('句子或设置已改动，需重新导出更新此版本'))}"` : '';
      return `<button data-compare="${esc(track.key)}" class="${cls}"${tip}>${esc(t(LABELS[track.key]))}${track.stale ? esc(t('（旧）')) : ''}</button>`;
    })
    .join('');
}

export function renderCompare(project) {
  const tracks = project.compare_tracks || [];
  const usable = tracks.length > 1;
  $('#comparePlayer').hidden = !usable;
  $('#compareEmpty').hidden = usable;
  const next = usable ? tracks.map((track) => `${track.key}:${track.url}:${track.modified ?? ''}`).join('|') : '';
  if (next !== signature) {
    teardown();
    signature = next;
    if (!tracks.some((track) => track.key === active)) active = 'source';
    for (const track of usable ? tracks : []) {
      const audio = new Audio();
      audio.preload = 'auto';
      audio.muted = track.key !== active;
      audio.src = track.url;
      players.set(track.key, audio);
    }
    $('#compareSeek').max = project.source.duration_seconds;
    showTime();
  }
  renderButtons(usable ? tracks : []);
  showLoopInfo();
}

export function initializeCompare() {
  $('#comparePlay').onclick = () => (playing ? stop() : play().catch(() => stop()));
  $('#compareSeek').oninput = () => seek(Number($('#compareSeek').value));
  $('#compareLoop').onchange = () => {
    loopedId = '';
    const sentence = $('#compareLoop').checked ? selectedSentence() : null;
    if (sentence) seek(sentence.start_seconds);
    showLoopInfo();
  };
  $('#compareTracks').onclick = (event) => {
    const button = event.target.closest('[data-compare]');
    if (!button || !players.has(button.dataset.compare)) return;
    position = master()?.currentTime ?? position;
    active = button.dataset.compare;
    players.forEach((audio, key) => {
      audio.muted = key !== active;
      if (Math.abs(audio.currentTime - position) > DRIFT_SECONDS) audio.currentTime = position;
    });
    $$('[data-compare]').forEach((node) => node.classList.toggle('on', node === button));
  };
}
