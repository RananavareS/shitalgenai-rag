// =============================================================
//  ShitalGenAI — RAG Edition (rag.js)
//  Handles: auth, document upload, chat with RAG context
// =============================================================

const CONFIG = {
  MAX_TOKENS:           4096,
  MAX_MEMORY_MESSAGES:  40,
};

const MODELS = [
  // ── Groq (free, fast) ──────────────────────────────────────────
  { id: 'llama-3.3-70b-versatile', label: '⚡ Llama 3.3 70B (Groq - Free)', default: true },
  { id: 'llama-3.1-8b-instant',    label: '⚡ Llama 3.1 8B (Groq - Fastest)' },

  // ── Anthropic Claude (best for coding) ─────────────────────────
  { id: 'claude-sonnet-4-5',  label: '🧠 Claude Sonnet 4.5 (Best for code)' },
  { id: 'claude-opus-4-1',    label: '🧠 Claude Opus 4.1 (Most powerful)'   },
  { id: 'claude-3-5-haiku',   label: '🧠 Claude 3.5 Haiku (Fast)'           },

  // ── OpenAI GPT ──────────────────────────────────────────────────
  { id: 'gpt-4o',       label: '🤖 GPT-4o' },
  { id: 'gpt-4o-mini',  label: '🤖 GPT-4o Mini (Fast)' },

  // ── Local Ollama (free, runs on your PC, best for code) ─────────
  { id: 'qwen2.5-coder:7b',      label: '💻 Qwen2.5 Coder 7B (Local - Free)' },
  { id: 'qwen2.5-coder:14b',     label: '💻 Qwen2.5 Coder 14B (Local - Free)' },
  { id: 'deepseek-coder-v2:16b', label: '💻 DeepSeek Coder V2 (Local - Free)' },
  { id: 'codellama:13b',         label: '💻 CodeLlama 13B (Local - Free)' },
];

// =============================================================
//  HELPERS
// =============================================================
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function now() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
const $ = id => document.getElementById(id);

// =============================================================
//  AUTH (same as original)
// =============================================================
const AUTH = {
  USER_DAILY_TOKEN_LIMIT: 4000,

  currentEmail() {
    const e = sessionStorage.getItem('sga_email');
    return e ? e.toLowerCase().trim() : null;
  },
  isAdmin()    { return sessionStorage.getItem('sga_role') === 'admin'; },
  isPro()      { return sessionStorage.getItem('sga_pro') === '1'; },
  isLoggedIn() { return this.isAdmin() || !!this.currentEmail(); },

  loginUser(email) {
    if (!email) return;
    sessionStorage.setItem('sga_email', email.toLowerCase().trim());
    sessionStorage.removeItem('sga_role');
  },
  async loginAdmin(pw) {
    try {
      const res = await fetch('/api/admin-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pw }),
      });
      const data = await res.json();
      if (data.ok) {
        sessionStorage.setItem('sga_role', 'admin');
        sessionStorage.removeItem('sga_email');
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },
  logout() {
    ['sga_email','sga_role','sga_pro','sga_payment_id'].forEach(k => sessionStorage.removeItem(k));
  },

  todayKey()   { return new Date().toISOString().split('T')[0]; },
  storageKey() {
    const e = this.currentEmail();
    return e ? 'sga_tok_' + e.replace(/[^a-z0-9]/gi, '_') : null;
  },
  getUsedToday() {
    const k = this.storageKey(); if (!k) return 0;
    const d = JSON.parse(localStorage.getItem(k) || '{}');
    return Math.max(0, Math.min(d[this.todayKey()] || 0, this.USER_DAILY_TOKEN_LIMIT));
  },
  addUsage(tokens) {
    const k = this.storageKey(); if (!k) return;
    const d = JSON.parse(localStorage.getItem(k) || '{}');
    const t = this.todayKey();
    Object.keys(d).forEach(day => { if (day < t) delete d[day]; });
    d[t] = Math.min(this.USER_DAILY_TOKEN_LIMIT, (d[t] || 0) + tokens);
    localStorage.setItem(k, JSON.stringify(d));
    this.updateBadge();
  },
  remaining() {
    return (this.isAdmin() || this.isPro())
      ? Infinity
      : Math.max(0, this.USER_DAILY_TOKEN_LIMIT - this.getUsedToday());
  },
  canSend() { return this.isAdmin() || this.isPro() || this.remaining() >= 200; },

  updateBadge() {
    const badge    = $('tokenBadge');     if (!badge) return;
    const userBar  = $('userInfoBar');
    const loginBtn = $('adminLoginBtn');
    const logoutBtn= $('adminLogoutBtn');
    const signBtn  = $('signOutBtn');

    if (this.isAdmin()) {
      if (userBar)   userBar.innerHTML       = `<span class="user-chip">👑 Admin</span>`;
      badge.innerHTML                        = `<div class="token-admin">Unlimited tokens</div>`;
      if (loginBtn)  loginBtn.style.display  = 'none';
      if (logoutBtn) logoutBtn.style.display = 'block';
      if (signBtn)   signBtn.style.display   = 'none';
    } else if (this.currentEmail()) {
      const used = this.getUsedToday(), lim = this.USER_DAILY_TOKEN_LIMIT;
      const pct  = Math.min(100, Math.round(used / lim * 100));
      const left = Math.max(0, lim - used);
      if (userBar) userBar.innerHTML = `<span class="user-chip">✉ ${escHtml(this.currentEmail())}</span>`;
      badge.innerHTML = `
        <div class="token-bar-wrap">
          <div class="token-bar-track"><div class="token-bar-fill" style="width:${pct}%"></div></div>
          <div class="token-numbers"><span>${left.toLocaleString()} left</span><span>${lim.toLocaleString()}/day</span></div>
        </div>`;
      if (loginBtn)  loginBtn.style.display  = 'none';
      if (logoutBtn) logoutBtn.style.display = 'none';
      if (signBtn)   signBtn.style.display   = 'block';
    } else {
      if (userBar)  userBar.innerHTML        = '';
      badge.innerHTML                        = '';
      if (loginBtn)  loginBtn.style.display  = 'block';
      if (logoutBtn) logoutBtn.style.display = 'none';
      if (signBtn)   signBtn.style.display   = 'none';
    }
  },
};

// =============================================================
//  TOAST
// =============================================================
function showToast(msg, borderColor = '#6366f1') {
  if (!document.body) return;
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.borderLeft = `3px solid ${borderColor}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// =============================================================
//  LOGIN FLOW
// =============================================================
function submitEmailLogin() {
  const input = $('loginEmail');
  const err   = $('loginEmailError');
  const email = (input?.value || '').trim();
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    if (err) err.textContent = 'Please enter a valid email address.';
    return;
  }
  AUTH.loginUser(email);
  enterApp();
}

async function showAdminLoginFromScreen() {
  const pw = prompt('Enter admin password:');
  if (pw === null) return;
  if (await AUTH.loginAdmin(pw)) {
    enterApp();
  } else {
    alert('Incorrect password.');
  }
}

function enterApp() {
  $('loginScreen').style.display = 'none';
  $('appScreen').style.display   = '';
  AUTH.updateBadge();
  populateModels();
  loadDocumentList();
  updateChunkBadge();
  renderHistory();
  fetchDeepLakeHistory();   // load full history from DeepLake
  wireAttachmentInputs();   // wire file/folder/image inputs
}

function signOut() {
  AUTH.logout();
  location.reload();
}

// =============================================================
//  MODELS
// =============================================================
function populateModels() {
  const sel = $('modelSelect'); if (!sel) return;
  sel.innerHTML = '';
  MODELS.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id; opt.textContent = m.label;
    if (m.default) opt.selected = true;
    sel.appendChild(opt);
  });
}

// =============================================================
//  DOCUMENT MANAGEMENT
// =============================================================
let docList = []; // local mirror of server doc list

async function loadDocumentList() {
  try {
    const res  = await fetch('/api/documents');
    docList    = await res.json();
  } catch {
    docList = [];
  }
  renderDocList();
  updateChunkBadge();
}

function renderDocList() {
  const el = $('docList'); if (!el) return;
  if (docList.length === 0) {
    el.innerHTML = '<div class="doc-empty">No documents yet</div>';
    return;
  }
  el.innerHTML = docList.map(d => `
    <div class="doc-item" id="doc-${escHtml(d.id)}">
      <span class="doc-icon">${docIcon(d.name)}</span>
      <div class="doc-info">
        <div class="doc-name" title="${escHtml(d.name)}">${escHtml(d.name)}</div>
        <div class="doc-chunks">${d.chunk_count} chunks</div>
      </div>
      <button class="doc-del" title="Remove" onclick="deleteDoc('${escHtml(d.id)}')">×</button>
    </div>`).join('');
}

function docIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  const map = { pdf:'📄', docx:'📝', doc:'📝', txt:'📃', md:'📑',
                py:'🐍', js:'🟨', ts:'🔷', html:'🌐', css:'🎨',
                json:'📋', csv:'📊', ipynb:'📓' };
  return map[ext] || '📄';
}

async function deleteDoc(docId) {
  try {
    await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    docList = docList.filter(d => d.id !== docId);
    renderDocList();
    updateChunkBadge();
    showToast('Document removed.', '#ef4444');
  } catch (e) {
    showToast('Failed to remove document.', '#ef4444');
  }
}

function updateChunkBadge() {
  const totalChunks = docList.reduce((sum, d) => sum + d.chunk_count, 0);
  const cc = $('chunkCount'); if (cc) cc.textContent = totalChunks;
  const rb = $('ragBadge'); if (rb) rb.textContent = `${docList.length} doc${docList.length !== 1 ? 's' : ''}`;
}

// File upload
async function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async e => {
      try {
        const base64 = e.target.result.split(',')[1];
        const res = await fetch('/api/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, data: base64 }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        resolve(data.doc);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error('File read error'));
    reader.readAsDataURL(file);
  });
}

async function handleFiles(files) {
  if (!files || files.length === 0) return;
  const prog  = $('uploadProgress');
  const fill  = $('progressFill');
  const label = $('progressLabel');
  const drop  = $('uploadDrop');
  if (prog) prog.style.display = 'block';
  if (drop) drop.style.display = 'none';

  for (let i = 0; i < files.length; i++) {
    const pct = Math.round(((i) / files.length) * 100);
    if (fill)  fill.style.width = pct + '%';
    if (label) label.textContent = `Processing ${files[i].name}…`;
    try {
      const doc = await uploadFile(files[i]);
      docList.push(doc);
      renderDocList();
      updateChunkBadge();
      showToast(`✅ ${doc.name} — ${doc.chunk_count} chunks indexed`, '#22c55e');
    } catch (err) {
      showToast(`❌ ${files[i].name}: ${err.message}`, '#ef4444');
    }
  }

  if (fill)  fill.style.width = '100%';
  if (label) label.textContent = 'Done!';
  setTimeout(() => {
    if (prog) prog.style.display = 'none';
    if (drop) drop.style.display = 'flex';
    if (fill) fill.style.width   = '0%';
  }, 1200);
}

// =============================================================
//  CHAT STATE
// =============================================================
const Memory = {
  messages: [],
  sessions: [],
  session:  { id: Date.now().toString(), title: 'RAG Chat', createdAt: new Date().toISOString() },

  add(role, content) {
    this.messages.push({ role, content });
    if (this.messages.length > CONFIG.MAX_MEMORY_MESSAGES)
      this.messages.splice(0, this.messages.length - CONFIG.MAX_MEMORY_MESSAGES);
    this.updateStats();
  },

  buildSystemPrompt(mode) {
    const modeConfig = {
      ask:       { instruction: 'Answer the user\'s question based on the provided document context. Be precise and cite relevant parts.' },
      summarize: { instruction: 'Summarize the key information from the provided documents. Structure with clear sections.' },
      extract:   { instruction: 'Extract and list specific facts, data, or entities the user asks about from the documents.' },
      compare:   { instruction: 'Compare and contrast information found across the provided document chunks.' },
      explain:   { instruction: 'Explain the concepts in the provided document context in simple, clear language.' },
    };
    const cfg = modeConfig[mode] || modeConfig.ask;
    return `You are ShitalGenAI RAG Assistant — an expert at reading, analyzing, and answering questions about documents.\nTask: ${cfg.instruction}\nIMPORTANT: Base your answer primarily on the RELEVANT DOCUMENT CONTEXT provided. If no relevant context is found, say so clearly and answer from general knowledge if possible.`;
  },

  saveSession() {
    const key  = 'sga_rag_sessions';
    const list = JSON.parse(localStorage.getItem(key) || '[]');
    const data = { ...this.session, messages: this.messages, lastUpdated: new Date().toISOString() };
    const idx  = list.findIndex(s => s.id === this.session.id);
    if (idx >= 0) list[idx] = data; else list.unshift(data);
    if (list.length > 10) list.length = 10;
    localStorage.setItem(key, JSON.stringify(list));
    this.sessions = list;
  },

  loadSessions() {
    this.sessions = JSON.parse(localStorage.getItem('sga_rag_sessions') || '[]');
    return this.sessions;
  },

  loadSession(id) {
    const s = this.sessions.find(s => s.id === id);
    if (!s) return false;
    this.session  = { id: s.id, title: s.title, createdAt: s.createdAt };
    this.messages = s.messages || [];
    return s;
  },

  reset() {
    this.messages = [];
    this.session  = { id: Date.now().toString(), title: 'RAG Chat', createdAt: new Date().toISOString() };
    this.updateStats();
  },

  updateStats() {
    const ms = $('msgCount'); if (ms) ms.textContent = this.messages.length;
  },
};

// =============================================================
//  HISTORY
// =============================================================
// ── DeepLake history ─────────────────────────────────────────
let deeplakeHistory = [];   // all rows fetched from /api/history

async function fetchDeepLakeHistory() {
  try {
    const email = AUTH.currentEmail() || '';
    const url   = email ? `/api/history?user_email=${encodeURIComponent(email)}&limit=200` : '/api/history?limit=200';
    const res   = await fetch(url);
    if (!res.ok) return;
    deeplakeHistory = await res.json();
    renderHistory();
  } catch (e) {
    console.warn('Could not fetch DeepLake history:', e);
  }
}

function renderHistory() {
  Memory.loadSessions();
  const list = $('historyList'); if (!list) return;

  // Group DeepLake rows by session_id
  const dlSessions = {};
  deeplakeHistory.forEach(row => {
    const sid = row.session_id || 'default';
    if (!dlSessions[sid]) {
      dlSessions[sid] = { id: sid, rows: [], lastTime: row.timestamp };
    }
    dlSessions[sid].rows.push(row);
    if (row.timestamp > dlSessions[sid].lastTime)
      dlSessions[sid].lastTime = row.timestamp;
  });

  // Merge localStorage sessions + DeepLake sessions
  const localIds = new Set(Memory.sessions.map(s => s.id));
  const dlEntries = Object.values(dlSessions)
    .filter(s => !localIds.has(s.id))
    .map(s => ({
      id:          s.id,
      title:       s.rows[0]?.content?.substring(0, 50) || 'Chat',
      lastUpdated: s.lastTime,
      source:      'deeplake',
      rows:        s.rows,
    }));

  const allSessions = [
    ...Memory.sessions.map(s => ({ ...s, source: 'local' })),
    ...dlEntries,
  ].sort((a, b) => new Date(b.lastUpdated || 0) - new Date(a.lastUpdated || 0));

  if (!allSessions.length) {
    list.innerHTML = '<div class="history-empty">No sessions yet</div>';
    return;
  }

  list.innerHTML = allSessions.map(s => `
    <div class="history-item" onclick="loadSession('${escHtml(s.id)}', '${escHtml(s.source || 'local')}')">
      <div class="history-title">${escHtml((s.title || 'RAG Chat').substring(0, 45))}</div>
      <div class="history-date">
        ${s.source === 'deeplake' ? '<span style="color:#a78bfa;font-size:9px;">☁ DeepLake</span> ' : ''}
        ${new Date(s.lastUpdated || s.createdAt).toLocaleDateString()}
      </div>
    </div>`).join('');
}

function loadSession(id, source) {
  $('welcome')?.remove();
  const box = $('messages'); if (!box) return;
  box.innerHTML = '';

  if (source === 'deeplake') {
    // Load from DeepLake rows
    const dlSession = Object.values(
      deeplakeHistory
        .filter(r => r.session_id === id)
        .reduce((acc, r) => { acc[r.message_id] = r; return acc; }, {})
    );
    dlSession.forEach(row => {
      appendUserBubble(row.content);
      if (row.response_text) appendBotBubble(row.response_text, []);
    });
    Memory.session = { id, title: dlSession[0]?.content?.substring(0, 50) || 'Chat', createdAt: dlSession[0]?.timestamp };
    Memory.messages = dlSession.flatMap(r => [
      { role: 'user',      content: r.content },
      ...(r.response_text ? [{ role: 'assistant', content: r.response_text }] : []),
    ]);
  } else {
    // Load from localStorage
    const s = Memory.loadSession(id);
    if (!s) return;
    s.messages.forEach(m => {
      if (m.role === 'user')           appendUserBubble(m.content);
      else if (m.role === 'assistant') appendBotBubble(m.content, []);
    });
  }
  Memory.updateStats();
}

// =============================================================
//  RENDERING MESSAGES
// =============================================================
let currentMode = 'ask';

function appendUserBubble(text, attachments) {
  const box = $('messages'); if (!box) return;
  // Add divider between exchanges
  if (box.children.length > 1) {
    const hr = document.createElement('hr');
    hr.className = 'exchange-divider';
    box.appendChild(hr);
  }
  const row = document.createElement('div');
  row.className = 'message user';
  let attachHtml = '';
  if (attachments && attachments.length) {
    attachHtml = attachments.map(a =>
      a.preview
        ? `<img src="${a.preview}" class="attachment-thumb" title="${escHtml(a.name)}"/>`
        : `<span class="source-chip">📎 ${escHtml(a.name)}</span>`
    ).join('');
  }
  row.innerHTML = `
    <div>
      ${attachHtml ? `<div style="margin-bottom:6px;text-align:right">${attachHtml}</div>` : ''}
      <div class="msg-text">${escHtml(text)}</div>
      <div class="msg-time">${now()}</div>
    </div>`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function appendBotBubble(text, sources) {
  const box = $('messages'); if (!box) return;
  const row = document.createElement('div');
  row.className = 'message bot';

  const sourcesHtml = sources && sources.length
    ? `<div class="rag-sources">
         <div class="rag-sources-label">Sources</div>
         ${[...new Set(sources)].map(s => `<span class="source-chip">${escHtml(s)}</span>`).join('')}
       </div>`
    : '';

  row.innerHTML = `
    <div class="bot-avatar">✦</div>
    <div class="msg-text">${renderMarkdown(text)}</div>
    ${sourcesHtml}
    <div class="msg-actions">
      <button class="msg-action-btn" onclick="copyToClipboard(this)" title="Copy">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      </button>
    </div>
    <div class="msg-time">${now()}</div>`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
  return row;
}

function copyToClipboard(btn) {
  const msgText = btn.closest('.message.bot').querySelector('.msg-text');
  navigator.clipboard.writeText(msgText.innerText || msgText.textContent || '');
  btn.innerHTML = '✓ Copied';
  setTimeout(() => btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy', 2000);
}

function renderMarkdown(text) {
  return text
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<div class="code-block"><div class="code-lang">${escHtml(lang || 'code')}</div><pre><code>${escHtml(code.trim())}</code></pre></div>`)
    .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^#{3}\s(.+)$/gm, '<h3>$1</h3>')
    .replace(/^#{2}\s(.+)$/gm, '<h2>$1</h2>')
    .replace(/^#{1}\s(.+)$/gm, '<h1>$1</h1>')
    .replace(/^[-*]\s(.+)$/gm, '<li>$1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(?!<[hlp]|<li|<div|<pre)(.+)$/gm, '<p>$1</p>');
}

// Loading bubble
function showLoading() {
  const box = $('messages'); if (!box) return null;
  const row = document.createElement('div');
  row.className = 'message bot loading-row';
  row.innerHTML = `
    <div class="msg-bubble bot-bubble">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
  return row;
}

// =============================================================
//  SEND MESSAGE
// =============================================================
async function sendMessage() {
  const input = $('userMsg'); if (!input) return;
  const text  = input.value.trim();
  if (!text) return;
  if (!AUTH.canSend()) {
    showToast('Daily token limit reached. Upgrade for unlimited access.', '#f59e0b');
    return;
  }

  $('welcome')?.remove();
  input.value = '';
  input.style.height = 'auto';

  const atts = [...pendingAttachments];
  clearAttachments();
  appendUserBubble(text, atts);
  Memory.add('user', text);

  const loader = showLoading();
  const btn    = $('sendBtn');
  if (btn) btn.disabled = true;

  try {
    const model      = $('modelSelect')?.value || 'llama-3.3-70b-versatile';
    const ragEnabled = $('ragToggle')?.checked ?? true;
    const sysPrompt  = Memory.buildSystemPrompt(currentMode);

    const controller = new AbortController();
    const timeoutId  = setTimeout(() => controller.abort(), 120000);
    const res = await fetch('/api/chat', {
      method:  'POST',
      signal:  controller.signal,
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        model,
        system:     sysPrompt,
        messages:   Memory.messages.slice(-CONFIG.MAX_MEMORY_MESSAGES),
        max_tokens: CONFIG.MAX_TOKENS,
        rag:        ragEnabled,
        session_id: Memory.session.id,
        user_email: AUTH.currentEmail() || '',
      }),
    });
    clearTimeout(timeoutId);

    loader?.remove();

    const data = await res.json();
    if (!res.ok) throw new Error(data?.error?.message || 'API error');

    const reply   = data.content?.[0]?.text || '';
    const sources = data.sources || [];
    appendBotBubble(reply, sources);
    Memory.add('assistant', reply);
    AUTH.addUsage(Math.ceil(reply.length / 4));

    // Auto-title session from first user message
    if (Memory.messages.filter(m => m.role === 'user').length === 1) {
      Memory.session.title = text.substring(0, 50) + (text.length > 50 ? '…' : '');
      const st = $('sessionTitle'); if (st) st.textContent = Memory.session.title;
    }
    Memory.saveSession();
    renderHistory();

  } catch (err) {
    loader?.remove();
    appendBotBubble(`⚠️ Error: ${err.message}`, []);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// =============================================================
//  EXPORT
// =============================================================
function exportChat() {
  if (!Memory.messages.length) { showToast('Nothing to export.', '#f59e0b'); return; }
  const md = Memory.messages.map(m =>
    `**${m.role === 'user' ? 'You' : 'ShitalGenAI'}:** ${m.content}`
  ).join('\n\n---\n\n');
  const blob = new Blob([md], { type: 'text/markdown' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `rag-chat-${Date.now()}.md`;
  a.click();
}

// =============================================================
//  DOMContentLoaded INIT
// =============================================================
document.addEventListener('DOMContentLoaded', () => {
  // Auto-login if already in session
  if (AUTH.isLoggedIn()) enterApp();

  // Login email keyboard shortcut
  const emailIn = $('loginEmail');
  if (emailIn) emailIn.addEventListener('keydown', e => {
    if (e.key === 'Enter') submitEmailLogin();
  });

  // Sidebar toggle
  $('toggleSidebar')?.addEventListener('click', () => {
    $('sidebar')?.classList.toggle('collapsed');
  });

  // New chat
  $('newChatBtn')?.addEventListener('click', () => {
    Memory.reset();
    const box = $('messages'); if (box) box.innerHTML = '';
    // Re-add welcome
    const w = document.createElement('div');
    w.className = 'welcome-msg'; w.id = 'welcome';
    w.innerHTML = `
      <div class="welcome-icon">📚</div>
      <h2>RAG Assistant</h2>
      <p>Upload documents in the sidebar, then ask questions.<br/>The AI will answer based on your documents' content.</p>`;
    box?.appendChild(w);
    $('sessionTitle').textContent = 'RAG Assistant';
  });

  // Send button
  $('sendBtn')?.addEventListener('click', sendMessage);

  // Enter key
  $('userMsg')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // Auto-resize textarea
  $('userMsg')?.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 160) + 'px';
  });

  // Mode buttons
  document.querySelectorAll('.mode-btn[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn[data-mode]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
      const ml = $('modeLabel'); if (ml) ml.innerHTML = `Mode: <b>${btn.textContent}</b>`;
    });
  });

  // File input
  $('fileInput')?.addEventListener('change', e => {
    handleFiles(Array.from(e.target.files));
    e.target.value = '';
  });

  // Drag-and-drop
  const drop = $('uploadDrop');
  if (drop) {
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
    drop.addEventListener('dragleave', ()  => drop.classList.remove('drag-over'));
    drop.addEventListener('drop', e => {
      e.preventDefault();
      drop.classList.remove('drag-over');
      handleFiles(Array.from(e.dataTransfer.files));
    });
  }

  // Export
  $('exportBtn')?.addEventListener('click', exportChat);

  // Admin login button (sidebar)
  $('adminLoginBtn')?.addEventListener('click', async () => {
    const pw = prompt('Enter admin password:');
    if (pw === null) return;
    if (await AUTH.loginAdmin(pw)) {
      AUTH.updateBadge();
      showToast('👑 Admin access granted!', '#22c55e');
    } else {
      showToast('Incorrect password.', '#ef4444');
    }
  });
  $('adminLogoutBtn')?.addEventListener('click', () => {
    AUTH.logout(); location.reload();
  });

  // History toggle
  const hToggle   = $('historyToggle');
  const hList     = $('historyList');
  const hChevron  = $('historyChevron');
  if (hToggle && hList) {
    hToggle.addEventListener('click', () => {
      const hidden = hList.style.display === 'none';
      hList.style.display = hidden ? '' : 'none';
      if (hChevron) hChevron.style.transform = hidden ? 'rotate(180deg)' : 'rotate(0deg)';
    });
  }
});

// =============================================================
//  ATTACHMENT HANDLING (files/folders/images in input bar)
// =============================================================
let pendingAttachments = [];  // { name, file, preview? }

function addAttachments(files, isFolder) {
  const fileArr = Array.from(files);

  if (isFolder && fileArr.length > 0) {
    // Show ONE folder chip instead of individual files
    const folderPath = fileArr[0].webkitRelativePath || '';
    const folderName = folderPath ? folderPath.split('/')[0] : 'Folder';
    const key = 'folder:' + folderName;
    if (!pendingAttachments.find(a => a.key === key)) {
      pendingAttachments.push({
        key,
        name: `📁 ${folderName} (${fileArr.length} files)`,
        file: null,
        preview: null,
        isFolder: true,
      });
    }
    renderAttachments();
    // Upload all files to knowledge base silently
    const docsOnly = fileArr.filter(f => !f.type.startsWith('image/'));
    if (docsOnly.length) handleFiles(docsOnly);
    return;
  }

  // Individual files
  fileArr.forEach(file => {
    if (pendingAttachments.find(a => a.name === file.name)) return;
    const att = { key: file.name, name: file.name, file, preview: null };
    pendingAttachments.push(att);
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = e => { att.preview = e.target.result; renderAttachments(); };
      reader.readAsDataURL(file);
    }
  });
  renderAttachments();
  const docsOnly = fileArr.filter(f => !f.type.startsWith('image/'));
  if (docsOnly.length) handleFiles(docsOnly);
}

function renderAttachments() {
  const preview = $('attachmentsPreview'); if (!preview) return;
  if (!pendingAttachments.length) {
    preview.style.display = 'none';
    preview.innerHTML = '';
    return;
  }
  preview.style.display = 'flex';
  preview.innerHTML = pendingAttachments.map((a, i) => `
    <div class="attachment-chip">
      ${a.preview
        ? `<img src="${a.preview}" class="attachment-thumb" alt="${escHtml(a.name)}"/>`
        : `<span style="font-size:16px">${docIcon(a.name)}</span>`}
      <span class="chip-name" title="${escHtml(a.name)}">${escHtml(a.name)}</span>
      <button class="chip-remove" onclick="removeAttachment(${i})" title="Remove">×</button>
    </div>`).join('');
  // Scroll input into view
  preview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function removeAttachment(i) {
  pendingAttachments.splice(i, 1);
  renderAttachments();
}

function clearAttachments() {
  pendingAttachments = [];
  renderAttachments();
}

// Global drag & drop onto the whole page
let dragOverlay = null;
document.addEventListener('dragenter', e => {
  if (!e.dataTransfer.types.includes('Files')) return;
  if (!dragOverlay) {
    dragOverlay = document.createElement('div');
    dragOverlay.className = 'drag-overlay';
    dragOverlay.innerHTML = '<div class="drag-overlay-text">📂 Drop files or folders here</div>';
    document.body.appendChild(dragOverlay);
  }
});
document.addEventListener('dragleave', e => {
  if (!e.relatedTarget || e.relatedTarget === document.documentElement) {
    dragOverlay?.remove(); dragOverlay = null;
  }
});
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  dragOverlay?.remove(); dragOverlay = null;
  const files = Array.from(e.dataTransfer.files);
  if (files.length) addAttachments(files);
});

// Wire up new input bar buttons — called from enterApp() after DOM is ready
function wireAttachmentInputs() {
  const ai = $('attachInput');
  const fi = $('folderAttachInput');
  const ii = $('imageAttachInput');
  if (ai) ai.addEventListener('change', e => { addAttachments(e.target.files); e.target.value = ''; });
  if (fi) fi.addEventListener('change', e => { addAttachments(e.target.files, true); e.target.value = ''; });
  if (ii) ii.addEventListener('change', e => { addAttachments(e.target.files); e.target.value = ''; });
}