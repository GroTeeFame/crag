import multiprocessing
import os

# IMPORTANT: With the current in-memory task state, run a single worker.
# When Redis-backed jobs are enabled, you can raise workers > 1.

bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
workers = int(os.getenv("GUNICORN_WORKERS", "1"))  # set >1 after Redis/RQ integration
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "16"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "600"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

