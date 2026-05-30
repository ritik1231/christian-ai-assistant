import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# Silence chromadb 0.5.x telemetry capture() bug (prints to stderr harmlessly)
try:
    import chromadb.telemetry.product.posthog as _ph
    _ph.Posthog.capture = lambda *a, **kw: None
except Exception:
    pass
