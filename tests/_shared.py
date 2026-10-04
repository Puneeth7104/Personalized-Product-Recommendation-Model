"""Shared, cached small pipeline run so the test-suite stays fast."""

from functools import lru_cache

from recsys.pipeline import PipelineConfig, run_pipeline


@lru_cache(maxsize=1)
def small_result():
    return run_pipeline(PipelineConfig(n_users=600, n_items=250, seed=7))
