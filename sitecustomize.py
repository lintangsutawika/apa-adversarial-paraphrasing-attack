import os


def _patch_verl_fsdp_sync_module_states() -> None:
    if os.environ.get("APA_DISABLE_FSDP_SYNC_MODULE_STATES") != "1":
        return

    try:
        import verl.workers.fsdp_workers as fsdp_workers
    except Exception:
        return

    if getattr(fsdp_workers, "_apa_sync_patch_applied", False):
        return

    original_build_model_optimizer = (
        fsdp_workers.ActorRolloutRefWorker._build_model_optimizer
    )

    def _patched_build_model_optimizer(self, *args, **kwargs):
        original_fsdp = fsdp_workers.FSDP

        def _patched_fsdp(*fsdp_args, **fsdp_kwargs):
            if fsdp_kwargs.get("sync_module_states"):
                if os.environ.get("RANK", "0") == "0":
                    print("APA patch: disabling FSDP sync_module_states during init")
                fsdp_kwargs["sync_module_states"] = False
            return original_fsdp(*fsdp_args, **fsdp_kwargs)

        fsdp_workers.FSDP = _patched_fsdp
        try:
            return original_build_model_optimizer(self, *args, **kwargs)
        finally:
            fsdp_workers.FSDP = original_fsdp

    fsdp_workers.ActorRolloutRefWorker._build_model_optimizer = (
        _patched_build_model_optimizer
    )
    fsdp_workers._apa_sync_patch_applied = True


_patch_verl_fsdp_sync_module_states()
