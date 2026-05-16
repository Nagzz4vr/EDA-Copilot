import os

HITL_DB_PATH = os.getenv("HITL_DB_PATH", "hitl_store.db")

HITL_POLL_INTERVAL = float(os.getenv("HITL_POLL_INTERVAL", "3.0"))

HITL_TIMEOUT_SECONDS = int(os.getenv("HITL_TIMEOUT_SECONDS", "1800"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")

LEDGER_DIR = os.getenv("LEDGER_DIR", "token_ledger/")
