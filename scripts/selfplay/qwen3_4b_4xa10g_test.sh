#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

OUTPUT_SEED_PATH=${OUTPUT_SEED_PATH:-data/a10g_test_seed.jsonl}
OUTPUT_ERROR_SEED_PATH=${OUTPUT_ERROR_SEED_PATH:-data/3b_coder_error_seed_io.jsonl}
OUTPUT_CODE_F_SEED_PATH=${OUTPUT_CODE_F_SEED_PATH:-data/3b_coder_code_f_seed_io.jsonl}

# Clean up potential old checkpoint directory
CHECKPOINT_DIR_TO_CLEAN="checkpoints/code_io/azr/azr_4xa10g_test/test_answer/Qwen2.5-Coder-3B/answer_conditional"
if [ -d "$CHECKPOINT_DIR_TO_CLEAN" ]; then
  echo "[coder3b_4xa10g_test.sh] Removing existing checkpoint directory: $CHECKPOINT_DIR_TO_CLEAN"
  rm -rf "$CHECKPOINT_DIR_TO_CLEAN"
else
  echo "[coder3b_4xa10g_test.sh] Checkpoint directory not found, no need to remove: $CHECKPOINT_DIR_TO_CLEAN"
fi

python -m absolute_zero_reasoner.main_azr_ppo \
    data.shuffle=True \
    actor_rollout_ref.ref.include_ref=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/code_reason/test_answer.parquet \
    data.val_files=data/code_reason/test_answer.parquet \
    `# TEST: Smaller batch sizes for faster testing` \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    `# A10G OPTIMIZATION 2: Moderate sequence lengths for A10G (pushing 24GB limits)` \
    data.max_prompt_length=3072 \
    data.max_validation_prompt_length=3072 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    `# A10G OPTIMIZATION 3: Reduced batch sizes for longer sequences` \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    `# A10G OPTIMIZATION 4: Disable sequence parallelism (using tensor parallelism instead)` \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    `# A10G OPTIMIZATION 5: Enable gradient checkpointing for memory efficiency` \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.pretrained_tokenizer=True \
    `# A10G OPTIMIZATION 6: Aggressive offloading for longer sequences` \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    `# A10G OPTIMIZATION 7: Reduced batch sizes for longer sequences` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    `# A10G OPTIMIZATION 8: Use 4-way tensor parallelism across A10Gs` \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    `# A10G OPTIMIZATION 9: vLLM batch tokens optimized for 3072-token sequences` \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    `# A10G OPTIMIZATION 10: Higher GPU memory utilization for longer sequences` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.enforce_eager=True \
    `# A10G OPTIMIZATION 11: Free cache between runs` \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    `# BUGFIX: Ensure vLLM uses the same sequence lengths` \
    actor_rollout_ref.rollout.prompt_length=3072 \
    actor_rollout_ref.rollout.response_length=3072 \
    `# A10G OPTIMIZATION 12: Enable ref model offloading` \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='azr' \
    `# TEST: Different experiment name for testing` \
    trainer.experiment_name='azr_4xa10g_test' \
    `# A10G OPTIMIZATION 13: 4 GPU setup` \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    `# TEST: More frequent saves for testing` \
    trainer.save_freq=5 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.del_local_ckpt_after_load=True \
    `# TEST: More frequent validation for testing` \
    trainer.test_freq=5 \
    +trainer.val_before_train=False \
    reward_fn.extraction_type=answer_conditional \
    reward_fn.math_metric=math_verify \
    trainer.val_generations_to_log_to_wandb=0 \
    azr.data_selection_strategy.update_iteration=1 \
    azr.seed_dataset=data/3b_coder_seed_io.jsonl \
    azr.output_seed_path=data/3b_coder_seed_io.jsonl \
    azr.error_seed_dataset=data/3b_coder_error_seed_io.jsonl \
    azr.output_error_seed_path=data/3b_coder_error_seed_io.jsonl \
    azr.code_f_seed_dataset=data/3b_coder_code_f_seed_io.jsonl \
    azr.output_code_f_seed_path=data/3b_coder_code_f_seed_io.jsonl \
    azr.pretrain_pred_steps=-1 \
    azr.executor=qwq \
    azr.ast_check=True \
    `# TEST: Fewer reward samples for faster testing` \
    azr.reward.n_samples=3 \
    azr.problem_types=['code_i','code_o','code_f'] \
    `# BUGFIX 1: Remove overly restrictive banned keywords` \
    azr.data_selection_strategy.banned_keywords_for_errors_and_exceptions=[] \
    trainer.debug=False \
    azr.reward.generation_reward_config.complexity_reward.coef=0.0 \
    azr.reward.generation_reward_config.complexity_reward.max=0.0 \
    azr.reward.generation_reward_config.complexity_reward.enabled=False \
    azr.reward.generation_reward_config.mean_edit_distance_reward.coef=0.0 \
    azr.reward.generation_reward_config.mean_edit_distance_reward.max=0.0 \
    azr.reward.generation_reward_config.mean_edit_distance_reward.enabled=False \
    azr.reward.generation_reward_config.halstead_reward.coef=0.0 \
    azr.reward.generation_reward_config.halstead_reward.max=0.0 \
    azr.reward.generation_reward_config.halstead_reward.enabled=False \
    azr.reward.generation_reward_config.answer_diversity_reward.coef=0.0 \
    azr.reward.generation_reward_config.answer_diversity_reward.max=0.0 \
    azr.reward.generation_reward_config.answer_diversity_reward.enabled=False \
    azr.reward.generation_reward_config.answer_diversity_reward.hierarchical=False \
    azr.pred_data_mix_strategy=uniform_total \
    azr.data_selection_strategy.seed_batch_factor=3 \
    `# BUGFIX 2: Use less restrictive program filter (allows all valid programs)` \
    azr.data_selection_strategy.valid_program_filter=all \
    `# BUGFIX 3: Ensure content_max_length allows longer prompts` \
    azr.data_selection_strategy.content_max_length=12288 \
    azr.data_selection_strategy.max_programs=12288 \
    azr.data_selection_strategy.batched_estimate=False \
    azr.reward.generation_reward_config.intrinsic_combine_method=sum \
    azr.gen_data_probabilities_strategy=uniform \
    trainer.resume_mode=auto \
    azr.data_selection_strategy.composite_start_step=-1 \
    azr.data_selection_strategy.composite_chance=0.0 \
    azr.reward.generation_reward_config.remove_comments=False \
    azr.reward.generation_reward_config.remove_after_return=False \
    azr.reward.generation_reward_config.use_original_code_as_ref=True \
    azr.reward.generation_reward_config.remove_print=False \
    azr.data_selection_strategy.composite_function_n_min=0 \
    azr.data_selection_strategy.composite_function_n_max=0 \
    azr.reward.code_f_reward_type=binary \
    trainer.wandb_run_id=null \
    `# TEST: Only 2 epochs for quick validation - MUST BE LAST` \
    trainer.total_epochs=2 \
    trainer.total_training_steps=-1 $@ 