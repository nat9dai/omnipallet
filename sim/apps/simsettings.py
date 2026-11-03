"""Sync render settings: frames lag exactly one render step (async: 1 to 4), so stamp them t - rendering_dt."""
import carb

SYNC = {"/app/asyncRendering": False, "/app/asyncRenderingLowLatency": False,
        "/omni/replicator/asyncRendering": False, "/app/hydraEngine/waitIdle": True}


def make_rendering_synchronous():
    s = carb.settings.get_settings()
    for k, v in SYNC.items():
        s.set(k, v)
