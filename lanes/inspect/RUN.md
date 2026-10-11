# Lane D (inspect): how every number is regenerated

All commands run from the Mac repo root; `scripts/sync.sh` first. `B=scripts/box.sh`. Raw data and runs live on the HDD workspace,
prepared/staged images on the NVMe (`/home/amirkhon/seen_fast/inspect`, 12 GB budget).

```bash
# 0. data (idempotent; Kaggle via the authenticated CLI, Roboflow key from the box .env, never printed)
$B -- python -P -m lanes.inspect.fetch carparts cardamagepanel jijo cardd syndcar tyrenet cococardamage vehide hitl tyre_warcoder tyre_sameer
$B -- bash -c 'set -a; source /mnt/bigdrive/amirkhon/seen/.env; set +a; python -P -m lanes.inspect.fetch_roboflow dot dot_finder dot_code sidewall'

# 1. prepare (one IO-heavy job at a time): stage -> hash/dedup on the NVMe copy -> splits
$B -- python -P -m lanes.inspect.prep_damage        # writes FAST/inspect/damage (variants A and B, eval sets)
$B -- python -P -m lanes.inspect.prep_parts         # needs prep_damage first (SYNDCAR is staged there once)
$B -- python -P -m lanes.inspect.prep_dot

# 2. train on GPU0 (always --lock); time-budgeted
$B --gpu 0 --lock --bg dmg_A -- python -P -m lanes.inspect.train_damage --exp dmg_A --data $FAST/damage/data_A.yaml --hours H ...
$B --gpu 0 --lock --bg dmg_B -- python -P -m lanes.inspect.train_damage --exp dmg_B --data $FAST/damage/data_B.yaml --hours H ...
$B --gpu 0 --lock --bg parts -- python -P -m lanes.inspect.train_damage --exp parts --data $FAST/parts/data.yaml --hours H ...
$B --gpu 0 --lock --bg dot_chars  -- python -P -m lanes.inspect.train_damage --exp dot_chars  --weights yolo11s.pt --imgsz 640 --aug '{"fliplr":0.0,...}' ...
$B --gpu 0 --lock --bg dot_region -- python -P -m lanes.inspect.train_damage --exp dot_region --weights yolo11s.pt --imgsz 640 ...

# 3. evaluate on GPU1 -> runs/inspect/<exp>/component_results_*.json
$B --gpu 1 -- python -P -m lanes.inspect.eval_damage --exp dmg_B --weights <best.pt>
$B --gpu 1 -- python -P -m lanes.inspect.eval_parts parts  --exp parts --weights <best.pt>
$B --gpu 1 -- python -P -m lanes.inspect.eval_parts assign --exp assign_B --parts-weights <best.pt> --damage-preds runs/inspect/dmg_B/preds_syndcar.json
$B --gpu 1 -- python -P -m lanes.inspect.tyre_cls --exp tyre_dino_lr
$B --gpu 1 -- python -P -m lanes.inspect.dot chars  --exp dot_chars  --weights <best.pt>
$B --gpu 1 -- python -P -m lanes.inspect.dot region --exp dot_region --weights <best.pt>
# tread depth (cut-able): fetch_tyrecv then lanes.inspect.tread

# 4. export + parity + latency (GPU1), failure gallery, collect
$B --gpu 1 -- python -P -m lanes.inspect.export yolo --name damage_seg --weights <best.pt> --imgsz 1024 --task seg --parity-dir ...
$B --gpu 1 -- python -P -m lanes.inspect.export latency --names damage_seg,parts_seg,tyre_cond,dot_region,dot_chars
$B --gpu 1 -- python -P -m lanes.inspect.failures damage --preds runs/inspect/dmg_B/preds_syndcar.json --set syndcar --exp dmg_B --out docs/...
$B -- python -P -m lanes.inspect.collect_results && scripts/sync.sh pull runs/inspect/component_results.json docs/lanes/inspect_results.json
```

Tests (CPU): `pytest lanes/inspect/test_smoke.py -q`.
