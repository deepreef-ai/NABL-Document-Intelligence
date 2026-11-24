# Local setup

Getting NABL Document Intelligence running on a fresh machine, start to finish.

For what the system does — architecture, the NABL schema, the OCR fallback — see
[README.md](README.md).

---

## 1. Prerequisites

| | Version | Notes |
|---|---|---|
| Python | 3.12 | 3.11+ should work |
| Node.js | 20.x | ships with npm 10 |
| AWS credentials | — | required — see below |

No Docker, Postgres or Qdrant needed. The app uses SQLite and local disk, both created on
first start.

**AWS credentials are not optional.** Every document is extracted by Amazon Nova on Bedrock,
which authenticates through the ambient AWS credential chain — environment variables,
`~/.aws/credentials`, or an SSO profile. There is no API key to paste. Check yours work before
going further:

```bash
aws sts get-caller-identity
```

You need `bedrock:InvokeModel` in `us-east-1`. Without this the app runs, uploads succeed, and
every document then fails to extract.

---

## 2. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Leave it running. Keep the port as 8000 — the frontend and the CORS allowlist both expect it.

**Wait about two minutes before uploading anything.** The server responds straight away, but it
builds the Docling document converter on a background thread (~110s) and the first upload
blocks until that finishes. Check with:

```bash
curl http://localhost:8000/health
# {"status":"ok","docling_warm":true}
```

`docling_warm: false` is not an error — it just means the build is still going.

---

## 3. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

There is no `.env` to copy here — `frontend/.env` already exists and holds the one setting that
matters, `VITE_API_BASE=http://localhost:8000`.

**It must run on port 5173.** The backend only accepts requests from that origin. If 5173 is
taken, Vite picks another port and the app will load but every request will fail with a CORS
error — free up 5173 rather than letting it move.

---

## 4. Check it works

1. <http://localhost:5173> opens on **Upload supporting documents**. There is no login and no
   setup wizard — an application is created for you and its id goes into the URL.
2. Drag in a PDF or image. The row shows `Processing`, then `Extracted` with a page and field
   count.
3. Click **Review** — the scanned page appears on the left, auto-filled fields on the right, and
   the extracted JSON underneath.
4. Confirm the fields, then **Approve & submit**.

**Extraction is slow, and that is normal:**


---

## Ports

| Service | URL |
|---|---|
| API | <http://localhost:8000> |
| API docs | <http://localhost:8000/docs> |
| Frontend | <http://localhost:5173> |

---

## Starting over

To reset to an empty app, stop the backend and delete both together:

```bash
rm backend/nabl.db
rm -rf backend/storage/*
```


