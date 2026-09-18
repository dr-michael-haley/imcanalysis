"""Thin vis-stage adapter. Scientific rendering remains in the figures package."""
from pathlib import Path


def run_figure_jobs(adata, jobs, output_root):
    from .specs import Figure
    from .sources import Dataset
    names = [job.name for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError('figure_jobs must use distinct output names.')
    results = {}
    for job in jobs:
        recipe = Figure.load(job.recipe)
        binding = job.model_dump(exclude={'name', 'recipe', 'rois', 'formats', 'on_error'})
        data = Dataset(adata=adata, **binding)
        results[job.name] = recipe.export_rois(data, Path(output_root) / job.name,
            rois=job.rois, formats=job.formats, on_error=job.on_error)
    return results
