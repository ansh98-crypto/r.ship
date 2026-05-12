from pathlib import Path

path = Path('index.html')
if not path.exists():
    raise SystemExit('index.html not found. Run this script in your r.ship project folder.')

html = path.read_text(encoding='utf-8')

css_marker = "/* ── Theme grid ── */"
css_add = """
/* ── Master answer ── */
.answer-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.answer-card h3 {
  font-family: 'Instrument Serif', serif;
  font-size: 22px;
  font-weight: 400;
  margin-bottom: 8px;
}
.answer-card .answer-body {
  font-size: 14px;
  color: var(--text);
  line-height: 1.65;
  white-space: pre-wrap;
}
.answer-meta {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
"""
if '.answer-card' not in html:
    html = html.replace(css_marker, css_add + "\n" + css_marker)

start = html.find('async function doSearch() {')
if start == -1:
    raise SystemExit('Could not find doSearch() function.')

next_marker = html.find('// ── ADD CONTACT', start)
if next_marker == -1:
    raise SystemExit('Could not find end marker after doSearch().')

new_do_search = r'''async function doSearch() {
  const q = $('searchInput').value.trim(); if(!q) return;
  $('searchResults').innerHTML = '<p style="color:var(--hint);font-size:13px">Thinking…</p>';
  $('intentBadge').innerHTML = '';

  try {
    const data = await api('/api/search', { method:'POST', body:JSON.stringify({query:q}) });
    const { results=[], intent, answer, mode, answer_contacts=[] } = data;

    if(intent?.intent) {
      $('intentBadge').innerHTML = `<span class="badge badge-accent">✦ ${escapeHtml(intent.intent)}</span>`;
    }

    let html = '';

    // New v3 behavior: show synthesized AI answer first.
    if(answer) {
      html += `
        <div class="answer-card">
          <h3>Answer</h3>
          <div class="answer-body">${formatAnswer(answer)}</div>
          <div class="answer-meta">
            ${mode ? `<span class="badge badge-accent">${escapeHtml(mode)}</span>` : ''}
            ${answer_contacts?.length ? `<span class="badge badge-ok">Used: ${answer_contacts.map(escapeHtml).join(', ')}</span>` : ''}
          </div>
        </div>
      `;
    }

    if(!results.length && !answer) {
      $('searchResults').innerHTML = `<div class="empty">No results for "${escapeHtml(q)}"</div>`;
      return;
    }

    if(results.length) {
      html += `
        <p class="sl-hdr">Relevant people / supporting memory</p>
        ${results.map(r => {
          const c = r.contact; const log = r.matched_log;
          const words = q.split(/\s+/).filter(w=>w.length>2);
          let snippet = log ? escapeHtml(log.text.slice(0,220)+(log.text.length>220?'…':'')) : '';
          words.forEach(w => {
            const safe = w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
            snippet = snippet.replace(new RegExp(safe,'gi'), m=>`<mark>${m}</mark>`);
          });
          return `
            <div class="r-card" onclick="openContact('${c.id}')">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                ${av(c,36)}
                <div style="flex:1">
                  <span style="font-size:14px;font-weight:600">${escapeHtml(c.name)}</span>
                  ${c.role?`<span style="font-size:12px;color:var(--muted);margin-left:6px">${escapeHtml(c.role)}</span>`:''}
                </div>
                <span style="font-size:11px;color:var(--hint)">${Math.round((r.score||0)*100)}% match</span>
              </div>
              ${snippet?`<p style="font-size:13px;color:var(--muted);line-height:1.5">${snippet}</p>`:''}
              ${log?.ai_summary?`<div style="margin-top:6px"><span class="badge badge-accent">✦ ${escapeHtml(log.ai_summary)}</span></div>`:''}
            </div>
          `;
        }).join('')}
      `;
    }

    $('searchResults').innerHTML = html;
  } catch(e) {
    $('searchResults').innerHTML=`<p style="color:var(--warn)">Error: ${escapeHtml(e.message)}</p>`;
  }
}

function escapeHtml(s='') {
  return String(s)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'",'&#039;');
}

function formatAnswer(text='') {
  return escapeHtml(text)
    .replace(/\n\s*[-•]\s+/g, '\n• ')
    .replace(/\n/g, '<br>');
}

'''

html = html[:start] + new_do_search + html[next_marker:]
path.write_text(html, encoding='utf-8')
print('Updated index.html successfully.')
