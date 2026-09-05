"""Explicit one-factor C29 + previously validated C26 composition; default off."""
PLAN_CONTRACT = {
    "name": "c29_plus_exact_critic_fa2_checkpoint_v1",
    "parent": "c29-continuous-b16-generator-rms-epilogue",
    "added_factor": "critic_fa2_exact_gpu_checkpoint_outputs_v1",
    "generator_path_unchanged": True,
    "application_calls_unchanged": True,
    "numeric_policy_unchanged": True,
}
