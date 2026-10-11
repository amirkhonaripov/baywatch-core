# Lane E (sim) — data sources, assets and models

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| Kenney Car Kit | https://kenney.nl/assets/car-kit | CC0 1.0 | synthetic renders (cars, wheels): train + test | n / n (toy models, no plates) |
| Kenney Mini Characters | https://kenney.nl/assets/mini-characters | CC0 1.0 | synthetic renders (proxy mechanics): train + test | stylised cartoon faces, not real people / n |
| Poly Haven HDRIs (auto_service, autoshop_01, garage, skylit_garage, machine_shop_02, empty_workshop) | https://polyhaven.com/hdris | CC0 1.0 | lighting and background of renders; skylit_garage held out for test | n / tiny distant background cars, plates not readable at 1k |
| Poly Haven textures (concrete_floor_02, concrete_floor_painted, concrete_floor_worn_001, painted_plaster_wall) | https://polyhaven.com/textures | CC0 1.0 | floor and wall materials | n / n |
| Blender 5.0.1 | https://www.blender.org | GPL-3.0 (outputs are ours) | rendering | – |
| Scenario engine (own code, `sim/scenario.py`) | this repo | project licence | all synthetic timelines, work orders, noise test | n / n (random synthetic plate strings in Uzbek format) |

The duration priors in `sim/scenario.py` are assumptions, loosely anchored on public figures quoted in
`research/service_ops_kpis.md`: Toyota TSM's "about 50 % of servicing time is the car stagnating", ~1 h express
service, J.D. Power's 22 min average handover, and ~20 % workshop idle. No dataset is used to fit them.

Pretrained models: none trained by this lane. `sim/eval_synthetic.py` runs whatever `seen.pipeline.default_pipeline()`
deploys at the time; licences are listed by the lanes that own those models (currently RF-DETR COCO and fast-alpr,
from `seen/baseline.py`).
