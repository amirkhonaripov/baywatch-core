#!/usr/bin/env bash
set -u
M=rfdetr-large-ft
python -W ignore -m lanes.flow.run_detect --models $M --datasets detrac_test_s5,ndispark_train,ndispark_validation,ndispark_test,cocoval_person --batch 8 2>&1 | grep -E "^saved|Error" | cut -c1-200
python -m lanes.flow.eval_detect --dataset detrac_test_s5 --models rfdetr-large,$M --exp ft_detrac --n-boot 300 2>&1 | tail -6
python -m lanes.flow.eval_detect --dataset cocoval_person --models rfdetr-large,$M --exp ft_cocoval_person --target person --n-boot 200 2>&1 | tail -5
python -m lanes.flow.eval_ndispark_gap --models rfdetr-large,$M --exp ft_ndispark --n-boot 500 2>&1 | tail -5
python -m lanes.flow.eval_count --models rfdetr-large,$M --exp ft_ndispark_count 2>&1 | tail -6
python -m lanes.flow.eval_track --model $M --exp track_$M --n-boot 100 2>&1 | tail -3
echo FT_EVAL_DONE
