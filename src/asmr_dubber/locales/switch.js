() => {
  if (window.asmrLanguage) return;
  const catalog = __CATALOG__;
  const originals = new WeakMap();
  const attributeOriginals = new WeakMap();
  let language = 'zh';
  try { language = localStorage.getItem('asmr.ui.language') === 'en' ? 'en' : 'zh'; } catch (_) {}
  const translate = (value) => {
    if (language !== 'en') return value;
    const trimmed = value.trim();
    const result = catalog[trimmed];
    if (result) return value.replace(trimmed, result);
    const templates = [
      [/^音频识别、翻译、中文配音与字幕制作 · (.+)$/, 'Speech recognition, translation, Chinese dubbing & subtitles · $1'],
      [/^第 (\d+)\/(\d+) 页 · 共 (\d+) 句 · 每页 (\d+) 句$/, 'Page $1/$2 · $3 sentences · $4 per page'],
      [/^适合 CUDA（建议 (.+) GB）$/, 'CUDA compatible ($1 GB recommended)'],
      [/^约 (.+) GB$/, 'About $1 GB'],
      [/^缺少 (.+) 运行时$/, 'Missing $1 runtime'],
      [/^系统：\s*(.+)$/, 'System: $1'],
      [/^内存：\s*(.+)$/, 'Memory: $1'],
      [/^推荐设备：\s*(.+)$/, 'Recommended device: $1'],
      [/^设置已保存：(.*)$/, 'Settings saved: $1'],
      [/^字幕每行上限：(\d+) 字符。$/, 'Subtitle line limit: $1 characters.'],
      [/^当前项目字幕每行上限：(\d+) 字符；需要重新生成字幕，已有文件不会自动改写。$/, 'Project subtitle line limit: $1 characters. Regenerate subtitles; existing files are not rewritten automatically.'],
      [/^(.+) · 驱动 ([\d.]+) · 计算能力 ([\d.]+)$/, '$1 · driver $2 · compute capability $3'],
    ];
    for (const [pattern, replacement] of templates) {
      if (pattern.test(trimmed)) return value.replace(trimmed, trimmed.replace(pattern, replacement));
    }
    const backendStatus = trimmed.match(/^\[(不可用|待确认|外部)\] (.+)$/);
    if (backendStatus && catalog[backendStatus[2]]) {
      return '[' + ({'不可用':'Unavailable','待确认':'Unverified','外部':'External'})[backendStatus[1]] + '] ' + catalog[backendStatus[2]];
    }
    if (trimmed.endsWith(' ·') && catalog[trimmed.slice(0,-2)]) {
      return catalog[trimmed.slice(0,-2)] + ' · ';
    }
    return value;
  };
  // Only presentation surfaces. Never edit values, contenteditable cells, source
  // text, translations, filenames, logs, prompts, or API payloads.
  const scope = 'button,label,legend,summary,h1,h2,h3,h4,[role="tab"],[role="option"],'+
    '.prose,[data-testid="block-info"],.info-text,.upload-text,'+
    '.sentence-editor th,.editor-toolbar,[data-role="status"],'+
    '#asmr-dubber-product-marker,.empty-state,.af-empty,.af-warning-badge,.af-reference-badge';
  const protectedArea = 'textarea,input,pre,code,[contenteditable="true"],'+
    '.sentence-editor tbody,.file-preview,[data-role="subtitle"],.af-path,[data-no-i18n]';
  let pending = false;
  let observer;
  function render() {
    pending = false;
    observer?.disconnect();
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    // Read-only application status gets a translated presentation alongside the
    // original control. Never replace a bound value or translate editable data.
    const statusLabels = new Set(['设置状态', 'Settings status', '分离密钥状态',
      'Separation key status', '实验分离安装状态', 'Experimental separation installation status']);
    document.querySelectorAll('textarea[readonly],textarea[disabled]').forEach(field => {
      const label = field.closest('label')?.querySelector('[data-testid="block-info"]')?.textContent.trim();
      if (!statusLabels.has(label)) return;
      const source = field.value;
      const translated = source.split('\n').map(translate).join('\n');
      let preview = field.parentElement.querySelector(':scope > .asmr-status-translation');
      if (language === 'en' && translated !== source) {
        if (!preview) {
          preview = document.createElement('div');
          preview.className = 'asmr-status-translation';
          preview.dataset.noI18n = '';
          preview.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;';
          field.after(preview);
        }
        preview.textContent = translated;
        preview.hidden = false;
        field.dataset.asmrTranslated = 'true';
        field.hidden = true;
      } else if (field.dataset.asmrTranslated) {
        field.hidden = false;
        delete field.dataset.asmrTranslated;
        if (preview) preview.hidden = true;
      }
    });
    document.querySelectorAll('.sentence-editor select option').forEach(option => {
      const previous = originals.get(option);
      const source = previous && option.textContent === previous.rendered ? previous.source : option.textContent;
      const rendered = translate(source);
      originals.set(option, {source, rendered});
      if (option.textContent !== rendered) option.textContent = rendered;
    });
    // Walk the document once. Per-scope walkers revisited nested Gradio trees
    // many times, making ordinary tab clicks feel queued on large pages.
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const parent = node.parentElement;
      if (!parent?.closest(scope) || parent.closest(protectedArea)) continue;
      const previous = originals.get(node);
      const source = previous && node.nodeValue === previous.rendered ? previous.source : node.nodeValue;
      const rendered = translate(source);
      originals.set(node, {source, rendered});
      if (node.nodeValue !== rendered) node.nodeValue = rendered;
    }
    // Accessibility labels and placeholders are presentation, not form values.
    document.querySelectorAll('[aria-label],[title],[placeholder]').forEach(el => {
      if (el.closest('[data-no-i18n]')) return;
      const saved = attributeOriginals.get(el) || {};
      for (const name of ['aria-label', 'title', 'placeholder']) {
        if (!el.hasAttribute(name)) continue;
        const value = el.getAttribute(name);
        const source = saved[name] && value === saved[name].rendered ? saved[name].source : value;
        const rendered = translate(source);
        saved[name] = {source, rendered};
        if (rendered !== value) el.setAttribute(name, rendered);
      }
      attributeOriginals.set(el, saved);
    });
    observer?.observe(document.body, {subtree:true, childList:true, characterData:true});
  }
  function setLanguage(value) {
    language = value === 'en' ? 'en' : 'zh';
    try { localStorage.setItem('asmr.ui.language', language); } catch (_) {}
    render();
    document.dispatchEvent(new CustomEvent('asmr-language-change', {detail:language}));
  }
  window.asmrLanguage = {set:setLanguage, get:() => language, translate};
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-ui-language]');
    if (button) setLanguage(button.dataset.uiLanguage);
  });
  observer = new MutationObserver(mutations => {
    const relevant = mutations.some(mutation => {
      if (mutation.type === 'characterData') return mutation.target.parentElement?.closest(scope);
      return [...mutation.addedNodes].some(node =>
        node.nodeType === Node.ELEMENT_NODE &&
        (node.matches?.(scope) || node.querySelector?.(scope))
      );
    });
    if (relevant && !pending) { pending = true; requestAnimationFrame(render); }
  });
  render();
}
