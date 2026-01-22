#!/bin/bash

# compare_experiments.sh
# Automated script to compare single-task vs multi-task grokking

echo "=============================================="
echo "  Grokking: Task Diversity Comparison"
echo "=============================================="
echo ""

# Configuration
SEED=42
LR=1e-3
WD=1e-3
STEPS=250000
LOG_INTERVAL=50

# Create results directory
mkdir -p results
cd results || exit

echo "Starting experiments..."
echo "Configuration: lr=$LR, weight_decay=$WD, steps=$STEPS, seed=$SEED"
echo ""

# Experiment 1: Baseline (Division only)
echo "[1/5] Running baseline: Division only..."
python ../run.py \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path exp1_division_only.png \
  > exp1_division_only.log 2>&1
echo "✓ Complete. Results saved to exp1_division_only.png"
echo ""

# Experiment 2: Multi-task 50% Division + 50% Addition
echo "[2/5] Running multi-task: 50% Division + 50% Addition..."
python ../run.py \
  --multi_task \
  --task_division 0.5 \
  --task_addition 0.5 \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path exp2_multitask_50_50.png \
  > exp2_multitask_50_50.log 2>&1
echo "✓ Complete. Results saved to exp2_multitask_50_50.png"
echo ""

# Experiment 3: Multi-task 70% Division + 30% Addition
echo "[3/5] Running multi-task: 70% Division + 30% Addition..."
python ../run.py \
  --multi_task \
  --task_division 0.7 \
  --task_addition 0.3 \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path exp3_multitask_70_30.png \
  > exp3_multitask_70_30.log 2>&1
echo "✓ Complete. Results saved to exp3_multitask_70_30.png"
echo ""

# Experiment 4: Multi-task 90% Division + 10% Addition
echo "[4/5] Running multi-task: 90% Division + 10% Addition..."
python ../run.py \
  --multi_task \
  --task_division 0.9 \
  --task_addition 0.1 \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path exp4_multitask_90_10.png \
  > exp4_multitask_90_10.log 2>&1
echo "✓ Complete. Results saved to exp4_multitask_90_10.png"
echo ""

# Experiment 5: Multi-task with all three operations
echo "[5/5] Running multi-task: 33% Division + 33% Addition + 34% Subtraction..."
python ../run.py \
  --multi_task \
  --task_division 0.33 \
  --task_addition 0.33 \
  --task_subtraction 0.34 \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path exp5_multitask_three_way.png \
  > exp5_multitask_three_way.log 2>&1
echo "✓ Complete. Results saved to exp5_multitask_three_way.png"
echo ""

# Extract and compare grokking delays
echo "=============================================="
echo "  RESULTS SUMMARY"
echo "=============================================="
echo ""

for log_file in exp*.log; do
  exp_name=$(basename "$log_file" .log)
  
  echo "--- $exp_name ---"
  
  # Try to extract grokking info
  if grep -q "Grokking delay:" "$log_file"; then
    grep "Grokking delay:" "$log_file" | tail -1
  else
    echo "Grokking not detected"
  fi
  
  # Extract final accuracies
  grep "Final Train Accuracy:" "$log_file" | tail -1
  grep "Final Val Accuracy:" "$log_file" | tail -1
  echo ""
done

echo "=============================================="
echo "All experiments complete!"
echo "Results saved in: $(pwd)"
echo "=============================================="