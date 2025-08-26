"""All-in-One Image Restoration package.

Subpackages:
- aoiir.datasets: dataset classes and Lightning DataModules
- aoiir.models: model wrappers (CompVis VQ-f4), PromptIR modules
- aoiir.engines: Lightning training modules (e.g., encoder alignment)
- aoiir.pipelines: inference pipelines (e.g., Stable Diffusion restoration)
"""

__all__ = [
    "datasets",
    "models",
    "engines",
    "pipelines",
]





