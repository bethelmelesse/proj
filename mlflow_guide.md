# MLflow Guide

Local MLflow setup for this project.

- Tracking URI: `http://localhost:5000`
- Backend store: `sqlite:///mlflow.db` (in project root)
- Artifact store: `mlartifacts/` (created on first artifact)

## Start the server

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Open the UI at <http://localhost:5000>.

## Stop the server

```bash
pkill -f "mlflow server"
```

Or find and kill explicitly:

```bash
pgrep -af "mlflow server"
kill <pid>
```

## Clear the server (full reset)

Wipes all experiments, runs, metrics, params, artifacts.

```bash
pkill -f "mlflow server"
rm -f mlflow.db
rm -rf mlartifacts mlruns
uv run mlflow server --host 127.0.0.1 --port 5000
```

## Garbage-collect soft-deleted entries

Keeps active experiments, permanently drops anything in the trash:

```bash
.venv/bin/mlflow gc --backend-store-uri sqlite:///mlflow.db
```

## Delete one experiment

While the server is running:

```bash
curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/delete \
  -H "Content-Type: application/json" \
  -d '{"experiment_id":"1"}'
```

## Restore a soft-deleted experiment

The UI's "Delete" button soft-deletes — the experiment still shows up in
`get_experiment_by_name(...)` with `lifecycle_stage='deleted'`, which makes
`mlflow.set_experiment(name)` fail. Restore it:

```bash
curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/restore \
  -H "Content-Type: application/json" \
  -d '{"experiment_id":"1"}'
```

Find the experiment id:

```bash
curl -sS http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results":50,"view_type":"ALL"}'
```

(`view_type: ALL` includes soft-deleted ones; default hides them.)

## Quick smoke test

Confirms the server accepts runs:

```bash
.venv/bin/python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('smoke')
with mlflow.start_run(run_name='smoke'):
    mlflow.log_param('ok', 1)
    mlflow.log_metric('val', 0.42)
"
```
