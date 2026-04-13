import os


def _patch_transformers_v5_compat() -> None:
    try:
        import transformers
    except Exception:
        return

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return

    replacement = getattr(transformers, "AutoModelForImageTextToText", None)
    if replacement is None:
        return

    transformers.AutoModelForVision2Seq = replacement


def _patch_vllm_qwen35_text_processing() -> None:
    if os.environ.get("APA_QWEN35_TEXT_ONLY") != "1":
        return

    try:
        import vllm.model_executor.models.qwen3_5 as qwen35_module
    except Exception:
        return

    if getattr(qwen35_module, "_apa_qwen35_text_patch_applied", False):
        return

    def _convert_hf_config(config, target_cls):
        if isinstance(config, target_cls):
            return config
        if hasattr(config, "to_dict"):
            return target_cls(**config.to_dict())
        return target_cls(**dict(config))

    def _text_or_vl_config(self):
        config = self.ctx.model_config.hf_config
        if getattr(config, "model_type", None) == "qwen3_5_text":
            return _convert_hf_config(config, qwen35_module.Qwen3_5TextConfig)
        return _convert_hf_config(config, qwen35_module.Qwen3_5Config)

    def _text_or_vl_moe_config(self):
        config = self.ctx.model_config.hf_config
        if getattr(config, "model_type", None) == "qwen3_5_moe_text":
            return _convert_hf_config(config, qwen35_module.Qwen3_5MoeTextConfig)
        return _convert_hf_config(config, qwen35_module.Qwen3_5MoeConfig)

    def _text_or_vl_data_parser(self):
        hf_config = self.get_hf_config()
        if not hasattr(hf_config, "vision_config"):
            from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser

            return Qwen2VLMultiModalDataParser(
                1,
                video_needs_metadata=False,
                expected_hidden_size=self._get_expected_hidden_size(),
            )
        return qwen35_module.Qwen3VLProcessingInfo.get_data_parser(self)

    qwen35_module.Qwen3_5ProcessingInfo.get_hf_config = _text_or_vl_config
    qwen35_module.Qwen3_5MoeProcessingInfo.get_hf_config = _text_or_vl_moe_config
    qwen35_module.Qwen3_5ProcessingInfo.get_data_parser = _text_or_vl_data_parser
    qwen35_module._apa_qwen35_text_patch_applied = True


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
        printed = {"done": False}

        def _patched_fsdp(*fsdp_args, **fsdp_kwargs):
            if fsdp_kwargs.get("sync_module_states"):
                if os.environ.get("RANK", "0") == "0":
                    print("APA patch: disabling FSDP sync_module_states during init")
                fsdp_kwargs["sync_module_states"] = False

            if os.environ.get("APA_EXCLUDE_VOCAB_WRAP") == "1":
                auto_wrap_policy = fsdp_kwargs.get("auto_wrap_policy")
                if auto_wrap_policy is not None:
                    import torch.nn as nn

                    if not printed["done"] and os.environ.get("RANK", "0") == "0":
                        print(
                            "APA patch: excluding giant vocab modules from FSDP auto-wrap"
                        )
                        printed["done"] = True

                    def _patched_auto_wrap_policy(module, recurse, nonwrapped_numel):
                        if isinstance(module, nn.Embedding):
                            return False
                        if (
                            isinstance(module, nn.Linear)
                            and getattr(module, "weight", None) is not None
                            and max(module.weight.shape) >= 100000
                        ):
                            return False
                        return auto_wrap_policy(module, recurse, nonwrapped_numel)

                    fsdp_kwargs["auto_wrap_policy"] = _patched_auto_wrap_policy

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


def _patch_verl_fsdp_wrap_policy() -> None:
    if os.environ.get("APA_EXCLUDE_VOCAB_WRAP") != "1":
        return

    try:
        import torch.nn as nn
        import verl.workers.fsdp_workers as fsdp_workers
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except Exception:
        return

    if getattr(fsdp_workers, "_apa_wrap_patch_applied", False):
        return

    original_get_fsdp_wrap_policy = fsdp_workers.get_fsdp_wrap_policy
    printed = {"done": False}

    def _patched_get_fsdp_wrap_policy(module, config=None, is_lora=False):
        if config is None:
            return original_get_fsdp_wrap_policy(module, config=config, is_lora=is_lora)

        if is_lora:
            return original_get_fsdp_wrap_policy(module, config=config, is_lora=is_lora)

        def _get_attr(attr_name, default_value=None):
            if hasattr(config, "get"):
                return config.get(attr_name, default_value)
            return getattr(config, attr_name, default_value)

        min_num_params = _get_attr("min_num_params", 0)
        if min_num_params <= 0:
            return original_get_fsdp_wrap_policy(module, config=config, is_lora=is_lora)

        if not printed["done"] and os.environ.get("RANK", "0") == "0":
            print(
                "APA patch: excluding giant vocab modules from size-based FSDP auto-wrap"
            )
            printed["done"] = True

        def _policy_fn(m, recurse, nonwrapped_numel):
            if isinstance(m, nn.Embedding):
                return False
            if (
                isinstance(m, nn.Linear)
                and getattr(m, "weight", None) is not None
                and max(m.weight.shape) >= 100000
            ):
                return False
            return size_based_auto_wrap_policy(
                module=m,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                min_num_params=min_num_params,
            )

        return _policy_fn

    fsdp_workers.get_fsdp_wrap_policy = _patched_get_fsdp_wrap_policy
    fsdp_workers._apa_wrap_patch_applied = True


_patch_transformers_v5_compat()
_patch_vllm_qwen35_text_processing()
_patch_verl_fsdp_sync_module_states()
_patch_verl_fsdp_wrap_policy()
