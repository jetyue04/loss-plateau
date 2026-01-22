#!/bin/bash

# quick_test.sh
# Fast test to validate multi-task implementation (shorter training runs)

echo "=============================================="
echo "  Quick Multi-Task Validation Test"
echo "=============================================="
echo ""
echo "This runs shortened experiments to verify"
echo "the implementation works correctly."
echo ""

# Configuration for quick testing
SEED=42
LR=1e-3
WD=1e-3
STEPS=25000  # Much shorter for quick testing
LOG_INTERVAL=100

mkdir -p quick_test_results
cd quick_test_results || exit

echo "Running quick tests (${STEPS} steps each)..."
echo ""

# Test 1: Division only
echo "[1/3] Testing single-task (division only)..."
python ../run.py \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path test_division.png \
  > test_division.log 2>&1
echo "✓ Complete"

# Test 2: Multi-task (division + addition)
echo "[2/3] Testing multi-task (division + addition)..."
python ../run.py \
  --multi_task \
  --task_division 0.5 \
  --task_addition 0.5 \
  --lr $LR \
  --weight_decay $WD \
  --num_steps $STEPS \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --save_path test_multitask_2ops.png \
  > test_multitask_2ops.log 2>&1
echo "✓ Complete"

# Test 3: Multi-task (all three operations)
echo "[3/3] Testing multi-task (division + addition + subtraction)..."
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
  --save_path test_multitask_3ops.png \
  > test_multitask_3ops.log 2>&1
echo "✓ Complete"

echo ""
echo "=============================================="
echo "Quick tests complete!"
echo ""
echo "Check the following files in quick_test_results/:"
echo "  - test_division.png"
echo "  - test_multitask_2ops.png"
echo "  - test_multitask_3ops.png"
echo ""
echo "Note: These are short runs ($STEPS steps) and"
echo "may not show full grokking. Run full experiments"
echo "with compare_experiments.sh for complete results."
echo "=============================================="