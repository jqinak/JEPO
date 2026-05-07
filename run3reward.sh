cd /project/peilab/qjl/2026/JEPO
source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
conda activate wmrl
module load cuda12.2

mkdir -p /project/peilab/qjl/2026/JEPO/logs
LOG_FILE="/project/peilab/qjl/2026/JEPO/logs/jepo_terminal_400_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=0 python -u -m main_jepo_qwenpi \
  trainer.total_training_steps=400 \
  reward.jepo.reward_type=terminal \
  trainer.output_dir=/project/peilab/qjl/2026/JEPO/checkpoints/jepo_terminal_400 \
  trainer.wandb.project=qwenPI_jepo_terminal_400 2>&1 | tee "${LOG_FILE}"

# LOG_FILE="/project/peilab/qjl/2026/JEPO/logs/jepo_sparse_400_$(date +%Y%m%d_%H%M%S).log"
# CUDA_VISIBLE_DEVICES=1 python -u -m main_jepo_qwenpi \
#   data.libero_subset=libero_goal \
#   trainer.total_training_steps=400 \
#   reward.jepo.reward_type=sparse_milestone \
#   trainer.output_dir=/project/peilab/qjl/2026/JEPO/checkpoints/jepo_sparse_400 2>&1 | tee "${LOG_FILE}"

# LOG_FILE="/project/peilab/qjl/2026/JEPO/logs/jepo_dense_400_$(date +%Y%m%d_%H%M%S).log"
# CUDA_VISIBLE_DEVICES=2 python -u -m main_jepo_qwenpi \
#   data.libero_subset=libero_goal \
#   trainer.total_training_steps=400 \
#   reward.jepo.reward_type=dense_milestone \
#   trainer.output_dir=/project/peilab/qjl/2026/JEPO/checkpoints/jepo_dense_400 2>&1 | tee "${LOG_FILE}"