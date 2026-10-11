# Lane C (ops) splits

Lane G had no `ops_*` manifest when this lane trained, so splits are by **source** (never by frame). All training lists
are also passed through `eval.manifest.filter_train` against every published manifest (incl. `jury`), which removes
held-out items by path, id or pHash near-duplicate; the removed list is written to `runs/ops/<exp>/dropped_by_manifests.json`
and the training list to `runs/ops/<exp>/train_files.txt` for G's audit.

| Data | Unit of split | Test | Train / val |
|---|---|---|---|
| Kinetics-700 clips | YouTube clip (official K700 split) | K700 **validation** split (all selected classes) | K700 train split; val = 15 % of train sources (seeded hash) for weight decay and Platt scaling |
| CrossTask + COIN car-task videos | YouTube video id: test iff sha1("seen-ops-v1\|yt:<id>") bucket < 0.25 (same hash as `eval.manifest._bucket`, per-id so it is stable as videos are added) | test videos | remaining videos (15 % of them = val) |
| Any YouTube id that is test in one source is test everywhere | YouTube id | — | — |
| Roboflow car-hood-open | Roboflow split | RF `test` | RF `train` + `valid` |
| Roboflow car-open-hood | Roboflow split | RF `test` | RF `train` + `valid` |
| Roboflow car-condition-detection (other uploader) | whole project | all images (cross-project hood test) | not used |
| Roboflow washing-car, carparts-seg, Commons lift photos | — | — | train only (Commons: 7 jury items removed by filter_train) |
| Public workshop CCTV/timelapse clips (6) | whole clip | **all** hand-labelled bay ROIs (real-domain test) | never used for training, thresholds or calibration |
| Hand-check handheld sample (96 frames) | — | drawn only from test videos/clips | not used for training |

Bootstrap units for CIs: K700 clip / CrossTask-COIN video / CCTV bay sequence (clip x ROI). Frames inside one unit are
strongly correlated, so per-frame Wilson intervals would be too narrow and are reported only for simple proportions.
