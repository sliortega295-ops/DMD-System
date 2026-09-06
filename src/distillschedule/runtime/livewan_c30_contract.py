"""Explicit one-factor C29 + previously validated C26 composition; default off."""
PLAN_CONTRACT = {
    "name": "c29_plus_exact_critic_fa2_checkpoint_v1",
    "parent": "c29-continuous-b16-generator-rms-epilogue",
    "added_factor": "critic_fa2_exact_gpu_checkpoint_outputs_v1",
    "generator_path_unchanged": True,
    "application_calls_unchanged": True,
    "numeric_policy_unchanged": True,
}

# A separate registered composition, not a silent mutation of C30's G path.
C31_PLAN_CONTRACT = {
    "name": "c30_plus_generator_cast_square_v1",
    "parent": "c30-continuous-b16-generator-rms-critic-checkpoint",
    "added_factor": "generator_nograd_rms_cast_square_v1",
    "critic_path_unchanged": True,
    "application_calls_unchanged": True,
    "numeric_policy_unchanged": True,
}

C32_PLAN_CONTRACT = {
    "name": "c31_plus_generator_fp32_modulation_v1",
    "parent": "c31-continuous-b16-generator-rms-cast-square",
    "added_factor": "generator_nograd_fp32_modulation_v1",
    "critic_path_unchanged": True,
    "application_calls_unchanged": True,
    "numeric_policy_unchanged": True,
}

C33_PLAN_CONTRACT = {
    "name": "c32_plus_generator_fp32_residual_gate_v1",
    "parent": "c32-continuous-b16-generator-modulation",
    "added_factor": "generator_nograd_fp32_residual_gate_v1",
    "critic_path_unchanged": True,
    "application_calls_unchanged": True,
    "numeric_policy_unchanged": True,
}
