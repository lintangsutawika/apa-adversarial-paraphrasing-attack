def setup() -> None:
    import importlib
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        replacement = getattr(transformers, "AutoModelForImageTextToText", None)
        if replacement is None:
            raise ImportError(
                "Transformers v5 compatibility patch could not find AutoModelForImageTextToText."
            )
        transformers.AutoModelForVision2Seq = replacement

    importlib.import_module("sitecustomize")
