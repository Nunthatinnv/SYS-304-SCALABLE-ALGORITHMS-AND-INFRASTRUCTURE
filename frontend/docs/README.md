# Frontend — House Price Predictor UI

A single-page, dependency-free interface for the prediction API.

## What it does

1. On load, calls `GET /health` to show an API status pill.
2. Calls `GET /schema` and **builds the form from that response** — labels,
   input types, defaults, numeric bounds and the `Neighborhood` dropdown all
   come from the backend.
3. On submit, `POST`s the collected values as JSON to `/predict` and renders the
   returned price.
4. On failure, shows a readable message. FastAPI `422` bodies are unpacked into
   `field: reason` lines instead of a raw JSON dump.

## Files

| File | Purpose |
|---|---|
| `index.html` | Markup and element hooks |
| `app.js` | Fetch logic, form building, submit, error handling |
| `style.css` | Styling (CSS custom properties, responsive grid) |
| `config.js` | Sets `window.API_URL`; rewritten at container start |
| `nginx.conf` | Static serving plus a `/healthz` route |
| `docker-entrypoint.sh` | Regenerates `config.js` from `$API_URL`, then starts nginx |
| `Dockerfile` | `nginx:alpine`, no build step |

## Why no framework

The milestone asks for "a simple web interface". Plain HTML/JS means:

- no `node_modules`, no bundler, no lockfile drift,
- CI needs only Python, so the pipeline stays fast,
- the Docker image is a few megabytes of nginx plus four static files.

## Why the form is generated, not hard-coded

If the field list lived in both `app.js` and `schemas.py`, the two would drift.
`GET /schema` makes the backend the single source of truth: adding a field to
`USER_FIELDS` there makes it appear in the UI with no frontend change. The test
suite asserts this by checking that no Kaggle column name is hard-coded in
`app.js`.

## Configuration

The browser — not the frontend container — calls the API, so `API_URL` must be
an address reachable from the user's machine. That is why `docker-compose.yml`
sets `API_URL: http://localhost:8000` rather than the compose service name
`http://backend:8000`.

`docker-entrypoint.sh` writes `config.js` at container start, so the API address
can change without rebuilding the image:

```bash
API_URL=http://192.168.1.10:8000 docker compose up -d frontend
```

Opening `index.html` straight from disk also works, as long as the backend runs
on `http://localhost:8000` and CORS allows the origin.

## Tests

```bash
pytest frontend/tests
```

`test_ui.py` asserts on the delivered source, since there is no build step:

- every element id `app.js` looks up exists in `index.html`,
- `config.js` loads before `app.js`,
- the client calls `/health`, `/schema` and `/predict`,
- `/predict` is sent as JSON via `POST`,
- no Kaggle column name is hard-coded in `app.js`,
- the entrypoint rewrites `config.js` from `API_URL`,
- `nginx.conf` serves the `/healthz` route the healthcheck uses.

## Manual check

```bash
./deploy.sh
open http://localhost:8080
```

Submit the form while watching `./deploy.sh --logs` to see the backend handle
each request.
