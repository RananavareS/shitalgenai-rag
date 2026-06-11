#!/usr/bin/env python3
# ShitalGenAI — RAG + DeepLake Multi-User Permanent Store
# One database stores ALL users data daily permanently
#
# Endpoints:
#   POST   /api/chat           — chat with RAG, saves to DeepLake
#   POST   /api/upload         — upload document
#   GET    /api/documents      — list documents
#   DELETE /api/documents/<id> — remove document
#   GET    /api/history        — fetch history (?user_email= &date= &session_id=)
#   DELETE /api/history        — clear all history
#   GET    /                   — serve index.html

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.request
import json, os, re, math, io, hashlib, time, mimetypes, base64, threading
try:
    import requests as _requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False

# ─── Auto-load .env ────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_URL        = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ADMIN_PASSWORD    = os.environ.get("ADMIN_PASSWORD", "")  # set in .env for production
ACTIVELOOP_TOKEN  = os.environ.get("ACTIVELOOP_TOKEN", "")
DEEPLAKE_ORG      = os.environ.get("DEEPLAKE_ORG", "")
PORT             = int(os.environ.get("PORT", 8080))
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR         = os.path.join(BASE_DIR, "rag_docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# Each model maps to (provider, real_model_id)
# provider: "groq" | "anthropic" | "openai"
MODEL_REGISTRY = {
    # Groq (fast, free tier)
    "llama-3.3-70b-versatile":       ("groq", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant":          ("groq", "llama-3.1-8b-instant"),
    "deepseek-r1-distill-llama-70b": ("groq", "llama-3.3-70b-versatile"),
    "llama3-8b-8192":                ("groq", "llama-3.1-8b-instant"),
    "llama3-70b-8192":               ("groq", "llama-3.3-70b-versatile"),
    "mixtral-8x7b-32768":            ("groq", "llama-3.3-70b-versatile"),
    "gemma2-9b-it":                  ("groq", "llama-3.1-8b-instant"),

    # Anthropic Claude (best for code reasoning)
    "claude-sonnet-4-5":   ("anthropic", "claude-sonnet-4-5-20250929"),
    "claude-opus-4-1":     ("anthropic", "claude-opus-4-1-20250805"),
    "claude-3-5-haiku":    ("anthropic", "claude-3-5-haiku-20241022"),

    # OpenAI GPT
    "gpt-4o":       ("openai", "gpt-4o"),
    "gpt-4o-mini":  ("openai", "gpt-4o-mini"),
    "gpt-4-turbo":  ("openai", "gpt-4-turbo"),

    # Local Ollama models (free, runs on your PC)
    "qwen2.5-coder:7b":      ("ollama", "qwen2.5-coder:7b"),
    "qwen2.5-coder:14b":     ("ollama", "qwen2.5-coder:14b"),
    "deepseek-coder-v2:16b": ("ollama", "deepseek-coder-v2:16b"),
    "codellama:13b":         ("ollama", "codellama:13b"),
}

# Backwards-compat alias used elsewhere in this file
MODEL_MAP = {k: v[1] for k, v in MODEL_REGISTRY.items() if v[0] == "groq"}

# ─── DeepLake — One Permanent Multi-User Database ─────────────────────────────
# Dataset: hub://<org>/shitalgenai_all_users_history
# Schema (one row = one Q&A exchange):
#   date           YYYY-MM-DD        daily filter
#   user_email     text              who asked
#   session_id     text              browser session
#   role           text              always "user"
#   content        text              user's question
#   response_text  text              AI's answer (plain text)
#   model          text              model used
#   timestamp      text              full ISO datetime
#   embedding      float32 vector    128-dim TF-IDF of question
#   embedding_text text              "[QUESTION] ..." readable label
#   message_id     text              unique row ID
# ──────────────────────────────────────────────────────────────────────────────

DATASET_NAME = "shitalgenai_all_users_history"
REQUIRED_TENSORS = {
    "date","user_email","session_id","role","content",
    "response_text","model","timestamp","embedding","embedding_text","message_id"
}

DEEPLAKE_AVAILABLE = False
ds = None


def _create_dataset(deeplake, path):
    global ds
    ds = deeplake.empty(path, overwrite=True)
    ds.create_tensor("message_id",    dtype="str",     htype="text")
    ds.create_tensor("date",          dtype="str",     htype="text")
    ds.create_tensor("session_id",    dtype="str",     htype="text")
    ds.create_tensor("user_email",    dtype="str",     htype="text")
    ds.create_tensor("role",          dtype="str",     htype="text")
    ds.create_tensor("content",       dtype="str",     htype="text")
    ds.create_tensor("response_text", dtype="str",     htype="text")
    ds.create_tensor("model",         dtype="str",     htype="text")
    ds.create_tensor("timestamp",     dtype="str",     htype="text")
    ds.create_tensor("embedding",     dtype="float32", htype="embedding")
    ds.create_tensor("embedding_text",dtype="str",     htype="text")
    ds.flush()
    print(f"  [DeepLake] ✅ Created dataset: {path}")


def init_deeplake():
    global DEEPLAKE_AVAILABLE, ds
    if not ACTIVELOOP_TOKEN or not DEEPLAKE_ORG:
        print("  [DeepLake] Not configured — using local JSON fallback.")
        print("  [DeepLake] Add ACTIVELOOP_TOKEN and DEEPLAKE_ORG to .env")
        return
    try:
        import deeplake
        os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN
        path = f"hub://{DEEPLAKE_ORG}/{DATASET_NAME}"
        try:
            ds = deeplake.load(path)
            existing = set(ds.tensors.keys())
            if not REQUIRED_TENSORS.issubset(existing):
                missing = REQUIRED_TENSORS - existing
                print(f"  [DeepLake] Missing tensors {missing} — recreating...")
                _create_dataset(deeplake, path)
            else:
                print(f"  [DeepLake] ✅ Loaded: {path} ({len(ds)} records)")
        except Exception:
            _create_dataset(deeplake, path)
        DEEPLAKE_AVAILABLE = True
    except ImportError:
        print("  [DeepLake] Not installed — run: pip install deeplake")
    except Exception as e:
        print(f"  [DeepLake] Init error: {e}")


# ─── Embedding ─────────────────────────────────────────────────────────────────
def simple_embed_128(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vec = [0.0] * 128
    for t in tokens:
        vec[int(hashlib.md5(t.encode()).hexdigest(), 16) % 128] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ─── Local JSON fallback ───────────────────────────────────────────────────────
LOCAL_HISTORY_PATH = os.path.join(BASE_DIR, "chat_history.json")

def _load_local():
    if os.path.exists(LOCAL_HISTORY_PATH):
        with open(LOCAL_HISTORY_PATH) as f:
            return json.load(f)
    return []

def _save_local(records):
    with open(LOCAL_HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)

def _append_local(row):
    records = _load_local()
    records.append(row)
    _save_local(records)
    print(f"  [LocalJSON] Saved: [{row.get('date','')}] [{row.get('user_email','anon')}] {row.get('content','')[:40]}")


# ─── Save to DeepLake (background thread) ─────────────────────────────────────
def _save_worker(session_id, user_email, content, reply, model, msg_id, timestamp):
    """Saves ONE row (question + answer) to DeepLake permanently."""
    date = timestamp[:10]   # YYYY-MM-DD
    row = {
        "message_id":    msg_id,
        "date":          date,
        "session_id":    session_id,
        "user_email":    user_email or "anonymous",
        "role":          "user",
        "content":       content,        # user question
        "response_text": reply,          # AI answer
        "model":         model,
        "timestamp":     timestamp,
    }
    if DEEPLAKE_AVAILABLE and ds is not None:
        try:
            import numpy as np
            row["embedding"]       = np.array(simple_embed_128(content), dtype="float32")
            row["embedding_text"]  = f"[QUESTION] {content}"
            ds.append(row)
            ds.flush()
            print(f"  [DeepLake] ✅ [{date}] [{user_email or 'anon'}] Q={content[:40]}")
        except Exception as e:
            print(f"  [DeepLake] Save error: {e}")
            _append_local(row)
    else:
        _append_local(row)


def save_message(session_id, user_email, content, reply, model=""):
    """Fire-and-forget: saves in background so HTTP response is instant."""
    msg_id    = hashlib.md5(f"{session_id}{time.time()}".encode()).hexdigest()[:12]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t = threading.Thread(
        target=_save_worker,
        args=(session_id, user_email, content, reply, model, msg_id, timestamp),
        daemon=True,
    )
    t.start()


# ─── Get history ───────────────────────────────────────────────────────────────
def get_history(session_id=None, user_email=None, date=None, limit=200):
    """Fetch history. Filter by user_email, date (YYYY-MM-DD), or session_id."""
    if DEEPLAKE_AVAILABLE and ds is not None:
        try:
            results = []
            tensors = set(ds.tensors.keys())

            def _val(name, i):
                if name not in tensors: return ""
                v = ds[name][i].numpy()
                if hasattr(v, "flatten"): v = v.flatten()
                return str(v[0]) if len(v) > 0 else ""

            for i in range(len(ds)):
                row = {
                    "message_id":    _val("message_id",    i),
                    "date":          _val("date",          i),
                    "session_id":    _val("session_id",    i),
                    "user_email":    _val("user_email",    i),
                    "content":       _val("content",       i),
                    "response_text": _val("response_text", i),
                    "model":         _val("model",         i),
                    "timestamp":     _val("timestamp",     i),
                }
                if session_id and row["session_id"] != session_id: continue
                if user_email and row["user_email"]  != user_email: continue
                if date       and row["date"]        != date:       continue
                results.append(row)
            return results[-limit:]
        except Exception as e:
            print(f"  [DeepLake] Fetch error: {e}")

    # Fallback: local JSON
    records = _load_local()
    if session_id: records = [r for r in records if r.get("session_id") == session_id]
    if user_email: records = [r for r in records if r.get("user_email") == user_email]
    if date:       records = [r for r in records if r.get("date","").startswith(date)]
    return records[-limit:]


def clear_history():
    if DEEPLAKE_AVAILABLE and ds is not None:
        try:
            import deeplake
            path = f"hub://{DEEPLAKE_ORG}/{DATASET_NAME}"
            deeplake.delete(path, force=True)
            init_deeplake()
            return True
        except Exception as e:
            print(f"  [DeepLake] Clear error: {e}")
    if os.path.exists(LOCAL_HISTORY_PATH):
        os.remove(LOCAL_HISTORY_PATH)
    return True


# ─── In-memory document store ──────────────────────────────────────────────────
CHUNKS: list    = []
DOCUMENTS: list = []

# ── Embeddings: sentence-transformers (semantic) with TF-IDF fallback ──────────
_ST_MODEL = None
def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("  [Embeddings] Loading sentence-transformers model (first time may take a minute)...")
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("  [Embeddings] ✅ Semantic embeddings ready (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"  [Embeddings] sentence-transformers unavailable ({e}) — using TF-IDF fallback")
            _ST_MODEL = False  # mark as unavailable
    return _ST_MODEL

def tokenize(text):   return re.findall(r"[a-z0-9]+", text.lower())
def build_tf(tokens):
    tf = {}
    for t in tokens: tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    return {k: v / total for k, v in tf.items()}
def cosine_sim_dict(a, b):
    common = set(a) & set(b)
    if not common: return 0.0
    dot = sum(a[k]*b[k] for k in common)
    ma  = math.sqrt(sum(v*v for v in a.values()))
    mb  = math.sqrt(sum(v*v for v in b.values()))
    return dot/(ma*mb) if ma and mb else 0.0

def cosine_sim_vec(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def embed(text):
    """Returns either a dense vector (list of floats) or a TF-IDF dict, depending on availability."""
    model = _get_st_model()
    if model:
        return model.encode(text, normalize_embeddings=True).tolist()
    return build_tf(tokenize(text))

def cosine_sim(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return cosine_sim_dict(a, b)
    return cosine_sim_vec(a, b)

def retrieve(query, top_k=4):
    if not CHUNKS: return []
    q = embed(query)
    scored = sorted(
        ((cosine_sim(q, c["embedding"]), c) for c in CHUNKS),
        key=lambda x: x[0],
        reverse=True,
    )
    # Semantic embeddings give meaningful scores around 0.2-0.6+;
    # TF-IDF gives smaller scores. Use a low-but-nonzero threshold either way.
    threshold = 0.15 if _get_st_model() else 0.001
    results = [c for s, c in scored[:top_k] if s > threshold]
    if not results and CHUNKS:
        results = [c for _, c in scored[:2]]
    return results


# ─── Document parsing ──────────────────────────────────────────────────────────
def parse_text_bytes(data, mime, filename):
    if filename.lower().endswith(".pdf"):  return parse_pdf(data)
    if filename.lower().endswith(".docx"): return parse_docx(data)
    try:    return data.decode("utf-8", errors="replace")
    except: return ""

def parse_pdf(data):
    try:
        import PyPDF2
        r = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n\n".join(p.extract_text() or "" for p in r.pages)
    except Exception as e: return f"[PDF error: {e}]"

def parse_docx(data):
    try:
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs if p.text.strip())
    except Exception as e: return f"[DOCX error: {e}]"

def chunk_text(text, size=250, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


# ─── Multi-provider LLM call ───────────────────────────────────────────────────
def _http_post_json(url, payload, headers, timeout=90):
    if USE_REQUESTS:
        resp = _requests.post(url, json=payload, headers=headers, timeout=timeout)
        if not resp.ok:
            import io as _io
            err = urllib.error.HTTPError(url, resp.status_code, resp.text, {}, _io.BytesIO(resp.content))
            raise err
        return resp.json()
    import gzip as _gz
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip": raw = _gz.decompress(raw)
    return json.loads(raw)


def call_groq(messages, model, max_tokens=4096, temperature=0.6):
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages, "temperature": temperature}
    headers = {
        "Content-Type":    "application/json",
        "Authorization":   f"Bearer {GROQ_API_KEY}",
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept":          "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin":          "https://console.groq.com",
        "Referer":         "https://console.groq.com/",
    }
    data = _http_post_json("https://api.groq.com/openai/v1/chat/completions", payload, headers)
    return data["choices"][0]["message"]["content"]


def call_anthropic(messages, model, max_tokens=4096, temperature=0.6):
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    # Anthropic uses a separate "system" field, not a system role in messages
    system_text = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text += ("\n" if system_text else "") + m["content"]
        else:
            chat_messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": chat_messages,
    }
    if system_text:
        payload["system"] = system_text

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    data = _http_post_json("https://api.anthropic.com/v1/messages", payload, headers)
    # Anthropic returns content as a list of blocks
    parts = data.get("content", [])
    return "".join(b.get("text", "") for b in parts if b.get("type") == "text")


def call_openai(messages, model, max_tokens=4096, temperature=0.6):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = _http_post_json("https://api.openai.com/v1/chat/completions", payload, headers)
    return data["choices"][0]["message"]["content"]


def call_ollama(messages, model, max_tokens=4096, temperature=0.4):
    """Call a local Ollama model (free, no API key needed)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    try:
        data = _http_post_json(f"{OLLAMA_URL}/api/chat", payload, headers, timeout=180)
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Is Ollama running? Install from https://ollama.com and run "
            f"'ollama pull {model}'. Error: {e}"
        )
    return data.get("message", {}).get("content", "")


def call_llm(messages, model_id, max_tokens=4096, temperature=0.4):
    """
    Unified entry point. model_id is the friendly name selected in the UI
    (e.g. 'llama-3.3-70b-versatile', 'claude-sonnet-4-5', 'gpt-4o', 'qwen2.5-coder:7b').
    Routes to the correct provider automatically.
    """
    provider, real_model = MODEL_REGISTRY.get(model_id, ("groq", model_id))
    if provider == "anthropic":
        return call_anthropic(messages, real_model, max_tokens, temperature)
    if provider == "openai":
        return call_openai(messages, real_model, max_tokens, temperature)
    if provider == "ollama":
        return call_ollama(messages, real_model, max_tokens, temperature)
    return call_groq(messages, real_model, max_tokens, temperature)


# ─── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/": path = "/index.html"

        if path == "/api/documents":
            self._json(200, DOCUMENTS); return

        if path == "/api/history":
            params = {}
            if "?" in self.path:
                for p in self.path.split("?", 1)[1].split("&"):
                    if "=" in p:
                        k, v = p.split("=", 1)
                        params[k] = v
            history = get_history(
                session_id = params.get("session_id"),
                user_email = params.get("user_email"),
                date       = params.get("date"),
                limit      = int(params.get("limit", 200)),
            )
            self._json(200, history); return

        fp = os.path.join(BASE_DIR, path.lstrip("/"))
        if not os.path.abspath(fp).startswith(BASE_DIR):
            self._text(403, "Forbidden"); return
        ext = os.path.splitext(fp)[1].lower()
        ct  = {".html":"text/html",".css":"text/css",".js":"application/javascript",
               ".png":"image/png",".jpg":"image/jpeg",".svg":"image/svg+xml",
               ".json":"application/json",".ico":"image/x-icon"}.get(ext,"text/plain")
        try:
            body = open(fp,"rb").read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self._cors(); self.end_headers(); self.wfile.write(body)
        except FileNotFoundError: self._text(404, f"Not found: {path}")
        except Exception as e:    self._text(500, str(e))

    def do_DELETE(self):
        if self.path == "/api/history":
            clear_history()
            self._json(200, {"ok": True}); return
        m = re.match(r"^/api/documents/([^/]+)$", self.path)
        if not m: self._json(404, {"error": "Not found"}); return
        doc_id = m.group(1)
        global CHUNKS, DOCUMENTS
        CHUNKS    = [c for c in CHUNKS    if c["doc_id"] != doc_id]
        DOCUMENTS = [d for d in DOCUMENTS if d["id"]     != doc_id]
        self._json(200, {"deleted": doc_id})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        if   self.path == "/api/chat":        self._handle_chat(body)
        elif self.path == "/api/upload":      self._handle_upload(body)
        elif self.path == "/api/admin-login": self._handle_admin_login(body)
        else: self._json(404, {"error": "Not found"})

    def _handle_admin_login(self, body):
        try:
            data = json.loads(body)
            password = data.get("password", "")
            if ADMIN_PASSWORD and password == ADMIN_PASSWORD:
                self._json(200, {"ok": True})
            else:
                self._json(401, {"ok": False, "error": "Incorrect password"})
        except Exception as e:
            self._json(400, {"ok": False, "error": str(e)})

    def _handle_chat(self, body):
        try:
            inc        = json.loads(body)
            model      = inc.get("model", "llama-3.3-70b-versatile")
            messages   = inc.get("messages", [])
            system     = inc.get("system", "")
            rag_on     = inc.get("rag", True)
            session_id = inc.get("session_id", "default")
            user_email = inc.get("user_email", "")

            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

            # Build RAG context
            rag_context = ""
            if rag_on and CHUNKS and last_user_msg:
                hits = retrieve(last_user_msg, top_k=4)
                if hits:
                    # Truncate each chunk's text to keep total prompt size small
                    MAX_CHUNK_CHARS = 800
                    ctx = "\n\n".join(
                        f"[Source: {h['doc_name']}, chunk {h['chunk_index']+1}]\n"
                        f"{h['text'][:MAX_CHUNK_CHARS]}"
                        + ("..." if len(h['text']) > MAX_CHUNK_CHARS else "")
                        for h in hits
                    )
                    rag_context = (
                        "\n\n--- PROJECT CODE CONTEXT (from the user's currently open project) ---\n" + ctx +
                        "\n--- END PROJECT CODE CONTEXT ---\n"
                        "IMPORTANT INSTRUCTIONS:\n"
                        "- The code above is from the user's ACTUAL project. Base your answer on it.\n"
                        "- Reference specific file names, function names, and existing code style shown above.\n"
                        "- When asked to add/change a feature, write code that fits into the existing files shown above "
                        "(matching naming conventions, frameworks, and patterns already used).\n"
                        "- Do NOT give generic textbook answers unrelated to this project.\n"
                        "- If the context truly doesn't cover what's asked, say so explicitly, then proceed with "
                        "your best suggestion that still matches the project's tech stack.\n"
                    )

            full_system = (system + rag_context) if rag_context else system
            # Limit conversation history to last 6 messages to control token usage
            trimmed_messages = messages[-6:]
            groq_msgs = ([{"role": "system", "content": full_system}] if full_system else []) + trimmed_messages
            # Cap max_tokens for the response itself
            max_out_tokens = min(int(inc.get("max_tokens", 2048)), 2048)
            reply = call_llm(groq_msgs, model, max_out_tokens, temperature=0.4)

            # Save one row: question + AI answer permanently
            if last_user_msg:
                save_message(session_id, user_email, last_user_msg, reply, model)

            self._json(200, {
                "content": [{"type": "text", "text": reply}],
                "storage": "deeplake" if DEEPLAKE_AVAILABLE else "local_json",
            })

        except urllib.error.HTTPError as e:
            eb = e.read() if hasattr(e, "read") else b"{}"
            eb = eb or b"{}"
            print(f"[LLM API {e.code}]: {eb.decode(errors='replace')}")
            try:
                parsed = json.loads(eb)
                # Anthropic: {"error":{"type":"...","message":"..."}}
                # OpenAI/Groq: {"error":{"message":"..."}}
                err = parsed.get("error", {}).get("message", str(e))
            except Exception:
                err = eb.decode(errors="replace")
            self._json(e.code, {"error": {"message": f"API {e.code}: {err}"}})
        except RuntimeError as e:
            # e.g. missing API key for selected provider
            print(f"[Config Error]: {e}")
            self._json(400, {"error": {"message": str(e)}})
        except Exception as e:
            import traceback
            print(f"[Server Error]:\n{traceback.format_exc()}")
            self._json(500, {"error": {"message": str(e)}})

    def _handle_upload(self, body):
        try:
            ct = self.headers.get("Content-Type", "")
            if "application/json" in ct:
                p        = json.loads(body)
                filename = p["filename"]
                raw      = base64.b64decode(p["data"])
            elif "multipart/form-data" in ct:
                filename, raw = self._parse_multipart(body, ct)
            else:
                self._json(400, {"error": "Unsupported content-type"}); return
            mime = mimetypes.guess_type(filename)[0] or ""
            text = parse_text_bytes(raw, mime, filename)
            if not text.strip(): self._json(400, {"error": "No text found"}); return
            doc_id = hashlib.md5((filename+str(time.time())).encode()).hexdigest()[:12]
            cks    = chunk_text(text)
            for i, ck in enumerate(cks):
                CHUNKS.append({"id":f"{doc_id}_{i}","doc_id":doc_id,"doc_name":filename,
                                "chunk_index":i,"text":ck,"embedding":embed(ck)})
            meta = {"id":doc_id,"name":filename,"type":mime or "text/plain",
                    "chunk_count":len(cks),
                    "uploaded_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
            DOCUMENTS.append(meta)
            self._json(200, {"ok": True, "doc": meta})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _parse_multipart(self, body, ct):
        boundary = re.search(r"boundary=([^\s;]+)", ct)
        if not boundary: raise ValueError("No boundary")
        for part in body.split(("--"+boundary.group(1)).encode()):
            if b"filename=" not in part: continue
            m = re.search(rb'filename="([^"]+)"', part)
            if not m: continue
            fn  = m.group(1).decode("utf-8", errors="replace")
            sep = part.find(b"\r\n\r\n")
            fd  = part[sep+4:].rstrip(b"\r\n--") if sep!=-1 else part[part.find(b"\n\n")+2:].rstrip(b"\r\n--")
            return fn, fd
        raise ValueError("No file part")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors(); self.end_headers(); self.wfile.write(body)

    def _text(self, code, msg):
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type","text/plain")
        self._cors(); self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    init_deeplake()
    k = f"✅ Loaded ({GROQ_API_KEY[:8]}...)" if GROQ_API_KEY else "❌ NOT SET"
    d = f"✅ Connected ({DATASET_NAME})"     if DEEPLAKE_AVAILABLE else "⚠️  Fallback (local JSON)"
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║   ShitalGenAI — RAG + DeepLake Multi-User Edition        ║
  ║   http://localhost:{PORT}                                   ║
  ╠══════════════════════════════════════════════════════════╣
  ║   GROQ_API_KEY : {k:<39}║
  ║   DeepLake     : {d:<39}║
  ╚══════════════════════════════════════════════════════════╝
    """)
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()