import { $, t } from './session.js';

// Wording that depends on where the project is: the first recognition is not a
// "re-run", and a subtitles-only export does not need any dubbing.
const hasTranslation = (s) => Boolean(s.target_text ?? s.zh_text);
const neverVoiced = (s) =>
  s.chinese_audio_enabled === false || ['skipped_tts', 'skipped_sound'].includes(s.status);

export function refreshStepHints(project) {
  const count = project.sentences.length;
  const analyze = $('#analyzeButton');
  if (analyze) {
    analyze.textContent = t(count ? '重新识别' : '开始识别');
    analyze.classList.toggle('primary', !count);
  }
  $('#asrStat').textContent = count
    ? t('已识别 {count} 句。重新识别会替换现有句子，译文和配音要重做。', { count })
    : t('还没有识别。识别完成后可以在表格里逐句修改；已有字幕或台本可以直接导入。');
  $('#exportStat').textContent = exportSummary(project);
}

function exportSummary(project) {
  const enabled = project.sentences.filter((s) => s.enabled);
  if (!project.sentences.length) return t('还没有句子，请先识别或导入台本。');
  const untranslated = enabled.filter((s) => !hasTranslation(s) && !neverVoiced(s)).length;
  if ($('#exportAudio')?.value === 'false') {
    const language = $('#exportSubtitle')?.value;
    if (language === 'none') return t('选了“只要字幕”，请在“字幕”里选要哪种字幕。');
    if (language === 'source' || !untranslated) return t('只导出字幕文件（SRT 和 LRC），不需要配音。');
    return t('只导出字幕文件（SRT 和 LRC），不需要配音。还有 {count} 句没有译文，字幕里会用原文代替。', {
      count: untranslated,
    });
  }
  if (!enabled.some(hasTranslation)) return t('还没有已翻译的句子，导出成品里将只有原声。');
  const undubbed = enabled.filter((s) => !neverVoiced(s) && !s.tts_file).length;
  return undubbed
    ? t('还有 {count} 句没有配音，成品里这些句子只有原声。', { count: undubbed })
    : t('所有句子都已配音。');
}

// Re-recognition replaces every sentence; translations and dubs cannot be recovered.
export function confirmAnalyze(project) {
  const kept = project.sentences.filter((s) => hasTranslation(s) || s.tts_file).length;
  return (
    !kept ||
    window.confirm(
      t('重新识别会替换现有的 {count} 句，已有的译文和配音会全部清空，无法撤销。确定继续吗？', {
        count: project.sentences.length,
      }),
    )
  );
}

export function exportRequest() {
  const audio = $('#exportAudio').value === 'true';
  const language = $('#exportSubtitle').value;
  if (!audio && language === 'none') throw new Error(t('选了“只要字幕”，请在“字幕”里选要哪种字幕。'));
  return { audio, language };
}

export function initializeStepHints(refresh) {
  $('#exportAudio').addEventListener('change', refresh);
  $('#exportSubtitle').addEventListener('change', refresh);
  // From the translation step straight to a subtitles-only export.
  $('#subtitlesOnlyShortcut').onclick = () => {
    $('#exportAudio').value = 'false';
    if ($('#exportSubtitle').value === 'none') $('#exportSubtitle').value = 'bilingual';
    window.dispatchEvent(new CustomEvent('set-step', { detail: 4 }));
    refresh();
  };
}
