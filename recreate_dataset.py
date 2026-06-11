import os, deeplake

# Load .env
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.environ.get("ACTIVELOOP_TOKEN", "")
ORG   = os.environ.get("DEEPLAKE_ORG", "")
os.environ["ACTIVELOOP_TOKEN"] = TOKEN

DATASET_NAME = "shitalgenai_all_users_history"
path = f"hub://{ORG}/{DATASET_NAME}"

# Delete old datasets
for old in ["shitalgenai_chat_history", "shitalgenai_all_users_history"]:
    try:
        deeplake.delete(f"hub://{ORG}/{old}", force=True)
        print(f"✅ Deleted: hub://{ORG}/{old}")
    except: pass

print(f"\nCreating: {path}")
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

print(f"""
✅ Dataset ready: {path}

Columns:
  date           — YYYY-MM-DD (filter by day)
  user_email     — who asked
  content        — user question (plain text)
  response_text  — AI answer (plain text)
  model          — model used
  timestamp      — full datetime
  session_id     — browser session
  embedding      — 128-dim vector
  embedding_text — [QUESTION] readable label
  message_id     — unique row ID

View: https://app.activeloop.ai/{ORG}/{DATASET_NAME}

Filter URLs (while server is running):
  All history:      http://localhost:8080/api/history
  By user:          http://localhost:8080/api/history?user_email=shital@gmail.com
  By date:          http://localhost:8080/api/history?date=2026-06-10
  By user+date:     http://localhost:8080/api/history?user_email=shital@gmail.com&date=2026-06-10

Now run: python server.py
""")