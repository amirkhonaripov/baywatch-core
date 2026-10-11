# Lane E (sim) — splits

All data in this lane is synthetic. Splits are by **source** (generator seed, camera, HDRI, car model), never by frame.

## Scenario engine (perception-free noise test)
| Use | Generator | Seed | n visits |
|---|---|---|---|
| HMM transition / duration priors (`sim/priors/hmm_rates.json`) | default `ScenarioConfig` | 1000 | 600 |
| Decoder tuning (eps, weight, prune scale) | default | 1500 | 40 |
| Test (all reported noise-test numbers) | default | 2000 | 120 |
| Prior-mismatch test | `MISMATCH` config (durations ×1.6, pauses ×1.9, absences ×2, uniform job mix) | 2001 | 120 |

## Blender renders (`renders/bay_v1`, generator seed 3000)
| Split | Episodes | Cameras | HDRIs | Car models |
|---|---|---|---|---|
| train | ep00–ep05 (6 episodes, 18 visits) | cam_A, cam_B | auto_service, autoshop_01, garage, machine_shop_02, empty_workshop | sedan, sedan-sports, suv, van, truck, hatchback-sports |
| test | ep06–ep09 (4 episodes, 11 visits, every frame) + ep10–ep12 (3 episodes, 10 visits, every 4th frame = 1 per 16 simulated s, Cycles 16 spp; generated with `--n-test 7`, same seed) = 21 visits | cam_C | skylit_garage | suv-luxury, taxi |

No camera, HDRI or car model in test appears in train. Characters (Kenney mini characters) and floor textures are
shared between splits, and so are the scenario job types and priors. Lane C may train on `train` episodes. Report
synthetic end-to-end numbers on `test` only.

Lane G: no eval manifest is needed for these renders; they are not drawn from any public dataset.
