# Stage-by-stage data and synthesis plan for service-flow analytics (Uzbek car dealers)

Research date: 2026-10-10. Follows /Users/diznect/Downloads/hackathon_final/research/data_models.md (read first; this file does not repeat its Uzbek plate format facts, model licences or PKLot/PhysicalAI-SmartSpaces details, it only adds to them).

Hackathon constraints used throughout: (a) every data source must be listed with its licence; (b) real faces or real licence plates of people who did not consent are forbidden in our data (blurring or replacing may make a source acceptable, noted per row); (c) synthetic data is allowed if the generation method is declared; (d) no real footage can be filmed now, the organisers provide no data.

## How to read the evidence tags

- **[F]** = read on the page, API, raw LICENSE/README file or rendered page (a browser pane was used for Roboflow, Kaggle, Pexels, Pixabay).
- **[S]** = taken from a search-result summary only. Treat as a lead.
- **[I]** = my inference or recommendation.
- "Not found" / "not read" = I looked and could not find it, or I opened the page and the field did not render. It does not mean it does not exist.
- Roboflow Universe "License: CC BY 4.0" is a label set by the uploader, not by the owner of the images. Most Universe projects say "A description for this project has not been published yet", so image provenance is unknown. Treat every Universe set as class B below unless proven otherwise.

## Acceptance classes used in the tables

- **A** = licence is clear and permissive and no real faces/plates are expected. Use directly, list with attribution.
- **B** = licence label is permissive but provenance is unknown, or real plates/faces may be visible. Usable only after a screening pass and plate/face blur or replacement. List with the caveat.
- **C** = reference or qualitative demo only (licence or terms of service do not allow ML training or bulk copying, or the data is too small or off-domain). Do not put in the training or test counts.
- **D** = avoid (non-commercial / research-only terms, signed agreement needed, real plates central to the data, or terms conflict with the hackathon rules).

---

## 0. Headline findings

1. **There is still no public labelled data for the middle of the flow (service-bay states).** Not on Roboflow Universe, Kaggle, Hugging Face, Open Images, LVIS, Objects365, ImageNet-1k, Wikimedia Commons or the Internet Archive. The closest labelled real images cover only single cues: bonnet open/closed (1,987 and 1,090 images), doors/bonnet open (411 + 408 images), car being washed (2,439 images), wheel/rim (1,286 images) and a 405-image "Carlifting" set of unknown content. No set has a class for "car on lift", "mechanic under car", "wheel removed" or "empty bay". [F]
2. **Stage 4 therefore needs synthetic data, and the cheapest credible route is a scripted Blender scene** built from CC BY Sketchfab car-lift/garage models (at least 8 car-lift models, all "CC Attribution"), CC0 Poly Haven garage/workshop HDRIs (about 15, e.g. `auto_service`, `autoshop_01`, `garage`, `skylit_garage`) and CC BY car models. Poly Haven itself has no usable car model (only `covered_car`, `old_tyre`, `rusted_wheel_rim_*`). [F]
3. **Stock video is thin and its terms are hostile to ML.** Pexels has roughly 380 unique car-mechanic/garage/tyre/wash clips reachable through 8 searches (about 5 show a lift), Pixabay roughly 20-30. Both Terms of Service forbid scraping/bots/automatic collection "for machine learning purposes" and bulk copying (the short licence summaries are silent on ML). Treat as class C. [F]
4. **Several well-known night/rain/damage datasets are restricted:** BDD100K (UC Regents licence: research/non-profit only, commercial only for BDD/BAIR members), DAWN (CC BY-NC 3.0, "commercial use strictly prohibited"), ExDark (README rewritten 2026-09-30: academic non-commercial, written confirmation needed for for-profit-sponsored projects), PVDN (CC BY-NC-ND 4.0 on Kaggle), CarDD (signed form; "testing commercial systems" counts as commercial use; no redistribution). Hugging Face mirrors of CarDD/ExDark/UA-DETRAC carry permissive-looking tags (apache-2.0, bsd-3-clause, cc-by-4.0) that do not override the original terms, and a Kaggle copy of CarDD in YOLO format also exists (its licence tag was not read). [F]
5. **Many Roboflow "night" sets use exactly the BDD100K class names** (car, truck, bus, train, rider, traffic light, traffic sign, motor, bike, person), so they are probably BDD100K derivatives carrying the BDD licence under a CC BY label. [I from class names read on the pages]
6. **Clean, fixed-camera, indoor/driveway real images exist only in small amounts**, e.g. "Parking Study (Garage)" (2,548 images, Vehicle + Person, CC BY 4.0, a fixed low-resolution garage/driveway view in the preview). Not verified for plates/faces. [F]
7. **Real plates are visible in at least one sampled preview** ("washing car", 2,439 images: a night image with a readable Arabic-script plate). So Universe sets need plate replacement before use. [F, one thumbnail only]
8. **Plate synthesis is feasible with MIT-licensed templates** (kstar1996/Car_Plate_DataGenerator composes plates onto car images; qjadud1994 Korean, zabir-nabil Bangla, simonyos Thai generators), and the Mercosur paper's trick (detect the real plate, replace it with a synthetic one at the same position and slope) doubles as an anonymiser for public car photos. Wikimedia Commons has 86 files in "License plates of Uzbekistan" (CC0/public domain/CC BY/CC BY-SA) for geometry reference. [F/S]
9. **Generative video is licence-safe for training augmentation only with Wan 2.x (Apache-2.0).** HunyuanVideo's licence forbids using outputs to improve any other AI model and excludes the EU/UK/South Korea; CogVideoX-5B needs a commercial registration. Generated clips carry no ground truth and cannot serve as test data. [F]
10. **The honest test claim** is "validated on synthetic timelines with known timestamps plus a few hundred real still images per cue from public sets; not yet validated on real service-bay CCTV". Section 7 gives the protocol.

---

## Stage 1. Entrance / arrival (vehicle at gate, plate reading, night and rain)

### 1.1 Real images of vehicles in night / rain / gate scenes

| Source | Link | Licence | Size | Annotations | Faces/plates? | Class / use | Tag |
|---|---|---|---|---|---|---|---|
| Roboflow "car-night" (car-ryzuj) | https://universe.roboflow.com/car-ryzuj/car-night-b9wra | CC BY 4.0 (uploader label) | 868 images | car, truck, bus, bike, motor, rider, person | Unknown (not inspected). Class names equal BDD100K's, so probably BDD-derived | B (D if BDD-derived) | [F] page, [I] provenance |
| Roboflow "car night" (autonomousvehicles-bokch) | https://universe.roboflow.com/autonomousvehicles-bokch/car-night-wihfx | CC BY 4.0 | 3,179 images | car, bus, bike, cycle, auto rickshaw, cycle rickshaw, mini truck, objects, scooty | Unknown; Indian-style road scenes | B | [F] |
| Roboflow "carnight" | https://universe.roboflow.com/carnight/car-night-a59cn | Licence not read | 6,049 images | car, truck, bus, traffic light, train, rider, traffic sign (BDD class names) | Unknown | B/D | [F] |
| Roboflow "Car cctv night" | https://universe.roboflow.com/ary-eswara-zz0dc/car-cctv-night | CC BY 4.0 | 173 images | cars | Unknown | B, tiny | [F] |
| Roboflow "night-car-track" | https://universe.roboflow.com/zhangs-workspace/night-car-track | CC BY 4.0 | 463 images | cars | Unknown | B | [F] |
| Roboflow "rainy" | https://universe.roboflow.com/car-5nkvd/rainy-4pkiz | CC BY 4.0 | 1,553 images | car, truck, pedestrian, biker, traffic lights | Unknown | B | [F] |
| Roboflow "Rainy vehicle detection" | https://universe.roboflow.com/nit-jalandhar-euvaa/rainy-vehicle-detection | Licence not read | 830 images | classes "0"-"5" (unnamed) | Unknown | B | [F] |
| Roboflow "Vehicle Gate Counter" | https://universe.roboflow.com/hssafty/vehicle-gate-counter | CC BY 4.0 | 252 images | car, truck, bus, motorcycle, big/mid/small bus and truck, cng | Unknown | B; useful gate-view sample | [F] |
| Roboflow "Vehicle On Gate" / "vehicle-gate-detection" | https://universe.roboflow.com/devagan/vehicle-on-gate ; https://universe.roboflow.com/feyza-axujh/vehicle-gate-detection | CC BY 4.0 | 9,293 / 462 images | toll-gate vehicle classes (GOL-I...; golongan_1-5) | Unknown; class scheme is a toll-gate axle class, not our need | B/off-domain | [F] |
| Roboflow "Vehicle Entrance MAnagement" | https://universe.roboflow.com/demo-nbeln/vehicle-entrance-management | CC BY 4.0 | 300 images | one class "licence" (plate boxes) | Real plates | D for our data | [F] |
| Roboflow "Parking Study (Garage)" | https://universe.roboflow.com/traffic-data-j2xav/parking-study-garage | CC BY 4.0 | 2,548 images, updated about a month ago | Vehicle, Person | Preview shows a fixed, low-res garage/driveway view; faces/plates not checked | B; best fixed-camera real sample found | [F] |
| BDD100K | https://doc.bdd100k.com/ ; licence text quoted in https://huggingface.co/datasets/dgural/bdd100k | UC Regents licence: "educational, research, and not-for-profit purposes" free; commercial use only for BDD and BAIR Commons members | 100k videos (not re-counted) | boxes, lanes, segmentation | Real roads; plate/face blurring not verified | D (a commercial dealer pilot is not covered) | [F] licence text, size [S] |
| DAWN (adverse weather) | https://data.mendeley.com/datasets/766ygrbt8y/3 | CC BY-NC 3.0; page: "commercial use/distribution of this database is strictly prohibited" | 1,000 images (fog, snow, rain, sandstorm) | vehicle boxes | Real traffic images; paper says images came from search engines/Flickr [S] | D | [F] licence+page, [S] sources |
| ExDark | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | README (updated 2026-09-30): non-commercial academic research; "written confirmation from the Licensor" needed where a for-profit entity sponsors the project; underlying photos not licensed by ExDark | 7,363 low-light images, 12 classes | image class + boxes | Real photos, third-party | D. The HF mirror `dronefreak/ExDark` is tagged bsd-3-clause but calls itself unofficial; the BSD licence in the repo covers code | [F] |
| PVDN (Provident Vehicle Detection at Night) | https://www.kaggle.com/datasets/saralajew/provident-vehicle-detection-at-night-pvdn | CC BY-NC-ND 4.0 (Kaggle page) | 38 GB; 54,659 or 59,746 annotated grayscale images depending on version [S] | headlights/vehicle labels, rural night | Rural night roads | D (NC-ND) | [F] licence, [S] counts |
| Zenseact Open Dataset (ZOD) | https://huggingface.co/datasets/Zenseact/ZOD | CC BY-SA 4.0 (HF card + PyPI); devkit MIT | 100k curated camera frames (paper); HF size tag 10K-100K | 2D/3D boxes, segmentation | **Faces and plates anonymised** (blur or Deep Natural Anonymization, by brighterAI) | A/B for "vehicle detection on real European roads incl. varied weather/light"; windshield camera, not a fixed gate view; night/rain share not verified; share-alike | [F] licence+anonymisation, [S] condition coverage |
| UA-DETRAC | https://arxiv.org/abs/1511.04136 | Licence not found. HF mirror `abhineet123/ua_detrac` says cc-by-4.0 but is an unofficial reformat | 100 sequences, >140k frames [S] | boxes, weather, occlusion | Real Chinese traffic, plates visible | D until a licence is found | [S] |
| Open Images V7 (val split checked) | https://storage.googleapis.com/openimages/web/factsfigures_v7.html | Images "CC BY 2.0" per Flickr, annotations CC BY 4.0 (previous report) | In the 41,620-image val split, human-verified "Car" is positive in 1,074 images and "Vehicle registration plate" in 504 | boxes (Car, Tire, Wheel, Vehicle registration plate, Tool, Wrench, Truck are boxable) | Real plates/faces appear | B only with plate replacement and face blur; use as a source of ordinary car images | [F] counts from the label CSV |
| COCO / LVIS / Objects365 / ImageNet-1k | various | COCO images Flickr mixed CC [S]; others not read | - | LVIS 1,203 classes include car, police cruiser, forklift, steering wheel, pickup truck; Objects365 has Car, Pickup Truck, Sports Car, Truck, Heavy Truck, Fire Truck; ImageNet-1k has "car wheel", "car mirror", "garbage truck" | Real | No garage/lift/service class in any list. Only useful as pretraining already inside RF-DETR weights | [F] class lists |

**Verdict [I]:** for stage 1 there is no clean, fixed-camera night/rain vehicle set with a licence we can rely on. Use (i) ZOD for a licence-clean real test of "vehicle detection under varied light/weather" (declare the windshield-camera domain), (ii) screened Roboflow gate/night sets as class B extras, and (iii) our own weather/night synthesis on top of scenes we render or composite (see 1.3).

### 1.2 Plate reading: synthetic Uzbek plates and compositing

| Source | Link | Licence | What it gives | Use | Tag |
|---|---|---|---|---|---|
| Wikimedia Commons "License plates of Uzbekistan" | https://commons.wikimedia.org/wiki/Category:License_plates_of_Uzbekistan | Per file. Of the 86 files: CC0 21, public domain 22, CC BY 4.0 22, CC BY-SA 3.0 13, CC BY-SA 4.0 8 | Plate type sheets ("Тип IA ... XIIIА" public-domain GIFs 180-300 px wide), 1996-era plate renderings (CC0; this is the older layout, do not copy for the current design), "Uzbekistan 2008 passenger license plate.png" (CC BY-SA 4.0), "Uzb autonum.png" (public domain), "UZ EV license plate.jpg" (CC BY 4.0, 455x106), Persian-wiki "Pelak ...-UZ.png" renderings (CC BY-SA 3.0; image content not viewed), and about 20 photographs of real plates (Bukhara/Samarkand/Karakalpakstan series, CC BY 4.0) | Reference for geometry, colours, flag block and type variants. **Keep the real-plate photographs out of any dataset** (they are real plates of real vehicles); use only as visual reference | [F] |
| kstar1996/Car_Plate_DataGenerator | https://github.com/kstar1996/Car_Plate_DataGenerator | MIT (GitHub API), 2 stars, last push 2022-01-10 | Plain/brightness/perspective/rain/sun/shadow plate scripts and a car-image generator that pastes generated plates onto your own car images | Closest template for compositing; Korean layout must be replaced | [F] licence, [S] features |
| qjadud1994/Korean-license-plate-Generator | https://github.com/qjadud1994/Korean-license-plate-Generator | MIT | Synthetic Korean plate generator, 64 stars | Template for OCR crop generation | [F] licence |
| zabir-nabil/bangla-license-plate-generator ; simonyos/thai-plate-synth | GitHub | MIT ; MIT | Plate generator ; plate generator + two-stage YOLO OCR pipeline | Templates | [F] licence |
| apereiracv/cr-plates-generator | https://github.com/apereiracv/cr-plates-generator | **GPL-3.0** | Costa Rica plates in several formats with perspective and character boxes | Read for ideas only; copyleft | [F] |
| Synthetic Ukrainian plate dataset (Zenodo) | https://zenodo.org/records/13342103 | CC BY 4.0 | 10,000 diffusion-generated plate images in YOLO format, 236 MB zip (paper: arXiv 2501.03374 "License plate images generation with diffusion models") | Style/reference for realism; wrong country | [F] record, [S] paper |
| Mercosur synthetic plates | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7736924/ | Not read | Detect legacy plates in real photos and overwrite them with synthetic plates at the same position and slope | The technique to adopt: it removes real plates from public car photos | [S] |
| fast-plate-ocr / fast-alpr / open-image-models | see data_models.md section 2.5 | MIT | OCR and plate-detector baselines | Fine-tune OCR on synthetic Uzbek plates | [F] (previous report) |

**Plate pipeline recommended [I], about 1 day:**
1. Render plate images from the format rules in data_models.md section 2.1 (private `DD L DDD LL`, legal `DD DDD LLL`, EV green region field, trailer two-line; invented strings only; region codes from the section 2.3 ranges) at 520:110 aspect with the flag/UZ block. Font: try FE-Schrift-style candidates and verify each font's licence (not resolved in the previous report).
2. Photometric realism: perspective warp, motion blur, JPEG, retroreflective glare, dirt, rain streaks, night/IR grayscale with blooming (Albumentations is MIT but the GitHub repo is flagged archived, last push 2025-06-25 [F]; the licence of any successor package was not checked, so write these few transforms with OpenCV).
3. **Replace, do not blur, real plates** in any public car photo: detect with the open-image-models plate detector, warp the synthetic plate in by homography. Annotation = plate box + text; the same step anonymises the source image.
4. OCR training set: 50-100k crops as planned. Detector set: composited car images and Blender renders where the plate is a textured quad.
5. Label every output "synthetic plate, invented number" in the data list and README (the method must be declared).

### 1.3 Night / rain synthesis options

| Option | Licence | What it does | Effort [I] | Tag |
|---|---|---|---|---|
| OpenCV/NumPy transforms (rain streaks, fog, gamma, noise, headlight bloom) | Apache-2.0 | Quick domain augmentation on stills and frames | 0.25 day | [I] |
| Albumentations (RandomRain, RandomFog, RandomShadow, RandomSunFlare) | MIT; GitHub repo flagged archived, last push 2025-06-25 | Same, ready-made | 0.1 day; check which package is maintained | [F] licence/archived |
| Poly Haven night HDRIs (61 tagged "night") and rain-ready materials in Blender | CC0 | Lighting for rendered scenes | included in the Blender build | [F] count |
| CARLA weather presets | Code MIT, assets CC BY (docs) | `WeatherParameters` (rain, fog, sun angle) apply to the RGB camera only; roads/driveways, no service bay | 1-2 days incl. setup; building custom maps needs the Unreal editor | [F] API text, [S] asset licence |
| NVIDIA Cosmos-Transfer1-7B | NVIDIA Open Model License, gated (auto), repo Apache-2.0 | Re-render a simulated clip with new weather/lighting | Heavy GPU; not tested; probably 1+ day | [F] licence tags, [I] effort |

### 1.4 Stage 1 recommendation

Best path: (1) synthetic Uzbek plates + plate-replacement on a small pool of CC BY/PD car photos (Commons, Open Images with detector-based replacement); (2) ZOD for a licence-clean real vehicle-detection test; (3) Blender-rendered driveway/gate scenes with HDRI night/rain; (4) screened Roboflow gate/night sets as extras. Say plainly in the deck that Uzbek plates in the test are synthetic.

---

## Stage 2. Incoming inspection (exterior damage, undercarriage, tyres)

### 2.1 Exterior damage

| Source | Link | Licence | Size | Annotations | Faces/plates? | Class | Tag |
|---|---|---|---|---|---|---|---|
| CarDD (USTC / CAS PIC Lab) | https://cardd-ustc.github.io/ | Licensing form: research use after prior consent; "any commercial use ... including testing commercial systems" needs authorisation; no redistribution to third parties | 4,000 images, 9,000+ instances, 6 categories (dent, scratch, crack, glass shatter, lamp broken, tire flat) | boxes + segmentation | Real car photos (plates not checked) | D. HF `shawnmichael/CarDD` (apache-2.0 tag), `harpreetsahota/CarDD` and a Kaggle YOLO-format copy are third-party redistributions | [F] licence PDF + page |
| VehiDE | https://doaj.org/article/6e0eb600059440b390431a42dfe8d138 | Not found; HF copies `hoangnguyenvan/VehiDE-dataset` and `thejashr/vehiDE_processed` have no licence tag | 13,945 images, 8 damage classes, 32,000+ instances [S] | classification/detection/segmentation | Real Vietnamese photos | D until licence found | [S] |
| Roboflow "car-damage" (skillfactory) | https://universe.roboflow.com/skillfactory/car-damage-c1f0i | CC BY 4.0 | 8,857 images, 31 classes | scratch, dent, damaged door/bumper/lamp/hood... | Unknown | B | [F] |
| Roboflow "Car Damage Dataset" (truck-damage) | https://universe.roboflow.com/truck-damage/car-damage-dataset-iqcfm | CC BY 4.0 | 7,133 images | 7 classes (numeric) | Unknown | B | [F] |
| Roboflow "Car Damages Kaggle" | https://universe.roboflow.com/ai-proyect/car-damages-kaggle | CC BY 4.0 | 814 images | scratch, dent, corrosion, broken part, cracked, flaking, missing part, paint chip | Unknown | B | [F] |
| Roboflow "Scratch/Dent" (FIU) | https://universe.roboflow.com/florida-international-university/scratch-dent | CC BY 4.0 | 207 images | scratch, dent | Unknown | B | [F] |
| Kaggle "Coco Car Damage Detection Dataset" | https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset | CC0 (uploader label) | 152 files, 15 MB, 58K downloads | COCO JSON | Unknown | B | [F] |
| Kaggle "Car Parts and Car Damages" (Humans in the Loop) | listed at https://www.kaggle.com/datasets?search=car+damage&tags=13207-Computer+Vision (page not opened) | Not read | 3 GB | not read | Unknown | B | [F] listing |

Note: the team already rejected photo damage inspection as a product, so stage 2 only needs a cue ("damage present at arrival: yes/no with location") and an evidence photo. 2-3 Roboflow sets above are enough; do not pull CarDD or VehiDE.

### 2.2 Undercarriage / under-vehicle (leaks, rust, damage)

| Source | Link | Licence | Size | Annotations | Class | Tag |
|---|---|---|---|---|---|---|
| Roboflow "undercarriage-rust-detection" | https://universe.roboflow.com/anirudh-qjb2e/undercarriage-rust-detection-f9grj | CC BY 4.0 | 2,055 images, updated about a month ago | rust, rust_normal, rust_severe | B; best undercarriage set found | [F] |
| Roboflow "Underbody car defects" | https://universe.roboflow.com/karthik-z0q6k/underbody-car-defects | CC BY 4.0 | 519 images | rust, bolt, rust-bolts | B | [F] |
| Roboflow "undercarriageCropping" / "Undercarriage-cropping" | https://universe.roboflow.com/car-segmentation-k5akd/undercarriagecropping ; https://universe.roboflow.com/copart-i960g/undercarriage-cropping | CC BY 4.0 | 588 / 417 images | one class (undercarriage / "0"), used to crop the underside | B | [F] |
| Roboflow "underbody-hard-parts-seg" | https://universe.roboflow.com/psbhatt-tekion-com/underbody-hard-parts-seg-sk6dq | Licence not read | 53 images | 13 classes (tyre, brake caliper/disc/pads/rotor, coil spring, exhaust tail pipe, stabilizer bar, strut assembly...) | B, tiny | [F] |
| Roboflow "Underbody Inspection System" (vw-nfqmp) | https://universe.roboflow.com/vw-nfqmp/underbody-inspection-system | CC BY 4.0 | 1,185 images | bracket/washer classes | Off-domain (factory) | [F] |
| Roboflow oil/coolant leak sets | https://universe.roboflow.com/hamza-bashir-nzrcf/oil-and-coolant-leak-drzbm ; https://universe.roboflow.com/oil-detection/oil-leak-detection ; https://universe.roboflow.com/tryout/oil-leak | CC BY 4.0 ; CC BY 4.0 ; not read | 49 ; 26 ; 68 images | one class each | Tiny; may be industrial rather than vehicle | [F] |
| Academic undercarriage work | https://techconnectworld.com/World2022/a.php?i=268 | Not a released dataset | 1,100 labelled images (8 tags about image defects, not rust) [S] | - | Not usable | [S] |
| UVeye / Ravin / patents (US 12177598, 11605151) | patents | Proprietary | - | - | Not usable | [S] |

Under-vehicle leak data: **no public vehicle leak dataset found** (only three tiny single-class sets). [F/not found]

### 2.3 Tyres: tread, sidewall, DOT code

| Source | Link | Licence | Size | Annotations | Class | Tag |
|---|---|---|---|---|---|---|
| Kaggle "Tyre Quality Classification" | https://www.kaggle.com/datasets/warcoder/tyre-quality-classification | CC BY 4.0 (page: "Attribution 4.0 International") | 1,856 files, 3 GB, 4,362 downloads | good vs defective | A/B (images of tyres; sources not documented) | [F] |
| Kaggle "Tyre Condition Classification Dataset" | listed at https://www.kaggle.com/datasets?search=tyre&tags=13207-Computer+Vision (page not opened) | Not read | 548 files, 3 GB | classification | B | [F] listing |
| Roboflow "OCR" (TyreAI) | https://universe.roboflow.com/tyreai/ocr-yphpv | CC BY 4.0 | 4,054 images | 39 character classes (0-9, A-Z, hyphen, slash, boundary) for sidewall text | A/B; best DOT-text set | [F] |
| Roboflow "Tire DOT finder" / "Dot Code" / "dot code" / "Sams Tires Analyzer" | https://universe.roboflow.com/dots-tracking/tire-dot-finder ; /subhams-workspace-l47ut/dot-code-6shg0 ; /panida-utamaang/dot-code ; /edgeoos/sams-tires-analyzer | CC BY 4.0 (all four) | 83 ; 97 ; 294 ; 180 images | one or three classes (DOT region / code rect) | B, tiny | [F] |
| Roboflow "tyre sidewall text label" | https://universe.roboflow.com/kumbhar-saket/tyre-sidewall-text-label | CC BY 4.0 | 129 images, updated 18 days ago | 31 classes (dot, construction, brand name, inflation pressure, size...) | B | [F] |
| Roboflow "Tyre sidewall text detection v2" | https://universe.roboflow.com/tyj-euwyt/tyre-sidewall-text-detection-v2 | Licence not read | 494 images | DOT, Brand, Model, Size | B | [F] |
| Roboflow "Tyre defect detection" | https://universe.roboflow.com/tyre-defect/tyre-defect-detection | Licence not read | 1,755 images | bulge, cracks, flat spots, non-defective | B | [F] |
| Roboflow "wheel-tire-detection" | https://universe.roboflow.com/bimmerwheels/wheel-tire-detection-ceyns | CC BY 4.0 | 1,286 images | car-tire, center, rim, Wheel-Rim, wheels | B; also helps "wheel removed" (rim visible) | [F] |
| Roboflow "tire_image_detection" | https://universe.roboflow.com/vinimagedetection/tire_image_detection | Licence not read | 973 images | tire_profile, tire_tread, tracks | B | [F] |
| Roboflow "Tire Tread" (Bams) ; "tyre-tread-detection" (MultiChart) | https://universe.roboflow.com/bams-workspace/tire-tread-82ob2 ; https://universe.roboflow.com/multichart/tyre-tread-detection-ulhub | CC BY 4.0 per search summary; the pages returned 403 to my fetch | not read | bad/bald/normal ; tread/tread-shallow | B; labels are qualitative, not mm of depth | [S] |
| DOT code format | https://developer.michelin.com/en/products/tires/dot-image-recognition | - | - | plant code, size code, manufacturer code, 4-digit week/year (e.g. "T7D3 1BH 3218" = week 32 of 2018) | Use as a validator regex | [S] |
| Michelin DOT API ; Anyline | links in source list | Paid services | - | - | Not data | [S] |

Open-licence tread-depth data in millimetres: **not found**. Tyre close-ups carry low face/plate risk. [I]

### 2.4 Synthesis for stage 2 [I]

- Damage: composite synthetic scratches/dents (alpha-blended decals with normal-map shading in Blender, or copy-paste of damage crops from the CC BY Roboflow sets onto panels) to make a test pool with known locations; about 0.5 day.
- DOT code: render DOT strings (valid plant/size/week/year patterns) onto tyre-sidewall textures with perspective and arc deformation (text on an annulus) using PIL/OpenCV; about 0.5 day. TyreAI's 4,054 character images give real-appearance negatives.
- Underbody: no photo-real synthesis worth the effort in 2 days; use the rust sets for a classifier head and say the leak cue is out of scope.

### 2.5 Stage 2 recommendation

Damage: Roboflow `car-damage` (skillfactory) + Kaggle Coco Car Damage as class B. Tyres: Kaggle Tyre Quality (A/B) + TyreAI OCR. Undercarriage: Roboflow rust set. Say clearly that CarDD/VehiDE were excluded for licence reasons.

---

## Stage 3. Reception / waiting area (occupancy, queue)

| Source | Link | Licence | Size | Annotations | Faces? | Class | Tag |
|---|---|---|---|---|---|---|---|
| NVIDIA PhysicalAI-SmartSpaces | https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces | CC-BY-4.0 | 280+ h from about 1,800 cameras: warehouses, hospitals, retail; tracks humans, forklifts, pallet trucks, AMRs | MOT-style tracks + calibration | Synthetic people, no real faces | A for testing person-in-zone and dwell-time code; no cars | [F] card |
| Roboflow "Parking Study (Garage)" | https://universe.roboflow.com/traffic-data-j2xav/parking-study-garage | CC BY 4.0 | 2,548 images | Vehicle, Person | Not checked | B | [F] |
| COCO person (already inside RF-DETR/D-FINE COCO weights) | - | per previous report | - | - | Real faces in COCO | No new data needed; the weights are the only exposure | [F] previous report |
| MOTSynth (GTA V pedestrians) | https://arxiv.org/abs/2108.09518 | Licence not found; game-engine content terms unclear | 1.3M frames [S] | boxes, tracks, masks | Synthetic | D until a licence is found | [S] |
| Pexels/Pixabay waiting-room clips | see Stage 4 | Pexels/Pixabay terms | - | none | Identifiable people | C | [F] |
| OpenCV YuNet face detector (blur at ingest) | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT (model directory LICENSE); repo Apache-2.0 | - | - | Used to blur faces before storage | A | [F] |

**Verdict [I]:** there is no public reception/waiting-area video worth using. Options, in order of cost: (1) **skip a trained model**: count "persons in the lounge polygon" with the pretrained person class, blur faces at ingest, and test only the zone logic on PhysicalAI-SmartSpaces retail scenes plus the synthetic timeline; (2) add generic synthetic humans to the Blender scene (Sketchfab CC BY humanoids; check each model's licence), counted but never identified. Do not claim queue accuracy on real waiting rooms.

---

## Stage 4. Service bay states (car on lift, wheel removed, bonnet open, mechanic under car, empty bay)

### 4.1 Labelled real images found

| Source | Link | Licence | Size | Annotations | Faces/plates? | Class | Tag |
|---|---|---|---|---|---|---|---|
| Roboflow "car-hood-open" | https://universe.roboflow.com/carhoodopen/car-hood-open | CC BY 4.0 | 1,987 images, updated 2 months ago | hood_closed, hood_open | Preview shows a BYD-type SUV outdoors; plates/faces not systematically checked | B | [F] |
| Roboflow "Car Open Hood" | https://universe.roboflow.com/vehicles-ztmmv/car-open-hood | CC BY 4.0 | 1,090 images | NotOpenHood, OpenHood (classification) | Not checked | B | [F] |
| Roboflow "car condition-detection" / "car condition" | https://universe.roboflow.com/deep-learning-ocos/car-condition-detection ; https://universe.roboflow.com/deep-learning-ocos/car-condition-ue963 | CC BY 4.0 | 411 / 408 images | all_closed, each door open, hood_open (per-part open/closed) | Not checked | B; useful for "doors/bonnet open" | [F] |
| Roboflow "washing car" | https://universe.roboflow.com/um-84bee/washing-car | CC BY 4.0 | 2,439 images, 5 versions | normal car, open doors, open hood, washing car | **A night image with a readable plate in the preview** | B (needs plate replacement) | [F] |
| Roboflow "wheel-tire-detection" | https://universe.roboflow.com/bimmerwheels/wheel-tire-detection-ceyns | CC BY 4.0 | 1,286 images | car-tire, center, rim, Wheel-Rim, wheels | Not checked | B | [F] |
| Roboflow "Carlifting" (SOBIRIN) | https://universe.roboflow.com/sobirin-bnpsx/carlifting-ppaom | CC BY 4.0 | 405 images | Tyre, Backlinear, Frontcar, Frontlinear, Midlinear, PlateNo, Stopper | Has a PlateNo class, so real plates | B/D; content unknown, probably a lane/stopper set not a lift set | [F] |
| Roboflow "Garage" (Egge Public) | https://universe.roboflow.com/egge-public/garage | CC BY 4.0 | 335 images | person, garbage bin, honda civic, honda crv, tool bucket (home-garage tracking) | Likely a private home (people, plates) | B | [F] |
| Roboflow "Dunexa Garage" | https://universe.roboflow.com/safety-warehouse/dunexa-garage | CC BY 4.0 | 133 images | Car, People, Cars-People-Forklifts | Not checked | B | [F] |
| Roboflow "garage-cars" | https://universe.roboflow.com/mole-detection-vdnqx/garage-cars-agxxa | CC BY 4.0 | 70 images | plate, cars, white | Real plates | D | [F] |
| Roboflow "Garage Car" ; "Workshop" ; "cars in garage" | https://universe.roboflow.com/garage-xwypn/garage-car-14cgx ; /worksho-wvdwu/workshop-zocjw ; /marc-lemoine-si0mt/cars-in-garage | CC BY 4.0 | 301 (COCO's 80 classes) ; 1,043 (mobil, truk) ; 572 (truck, camaro) | vehicle boxes only | Not checked | Off-domain | [F] |
| Roboflow "carwash-repair" (PPL) | https://universe.roboflow.com/ppl-ombde/carwash-repair | CC BY 4.0 | 1,679 images | Car, License Plate; "gathered from multiple sources on the internet" | Real plates by design | D | [F] |
| Roboflow "Lift" family (ImageMaker7/8/9 etc.) | https://universe.roboflow.com/search?q=car+lift | CC BY 4.0 (label) | 225-700 images each | lift / lifts | **Elevators, not car lifts** (all 50 results were lifts/elevators/forklifts/ski lifts) | Off-domain | [F] |
| Roboflow "final_project" (class `car_lift`) | https://universe.roboflow.com/university-e0xh4/final_project-hazx9 | CC BY 4.0 | 189 images | car, bird, airplane, cone, container, person, car_lift, fork_lift, lift, palette | Looks like an airport/apron project, not a service lift [I] | Off-domain | [F] classes |
| Kaggle (5 searches in the UI) | https://www.kaggle.com/datasets?search=car+lift | - | "car lift": 9 irrelevant CSV results; "car repair garage images": 0; "mechanic workshop": 2 (Scania APS CSV, "Automobile workshop dataset" CSV); "tyre dot tire sidewall": 0 | - | - | Nothing for bay states | [F] |
| Hugging Face datasets (API search for car lift, vehicle lift, garage, auto repair, mechanic, workshop, tow truck) | https://huggingface.co/api/datasets?search=garage | - | Only noise (`lzyanhui/carla_garage_v1` has no card/README; `apcl/autorepair` is not images) | - | - | Nothing | [F] |
| Open Images V7 | https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv | Images CC BY 2.0 [previous report] | The 19,957-class image-level vocabulary has "Automobile repair shop", "Auto mechanic", "Garage", "Bus garage", "Car wash", "Car dealership", "Auto part", "Automotive tire", "Alloy wheel"; the 601 boxable classes have none of these (only Car, Tire, Wheel, Tool, Wrench, Truck, Vehicle registration plate...) | **Val split: 0 human-verified positives** for repair shop, mechanic, garage, car wash; machine-generated labels with confidence >= 0.7: repair shop 8, auto mechanic 4, garage 9, car dealership 14 images (of 41,620). Scaling by roughly 40x gives a few hundred unverified train candidates [I] | Real people/plates | C/D: too few, unverified | [F] counts |
| LVIS / Objects365 / ImageNet-1k | see Stage 1 | - | No lift/garage/service class | - | - | None | [F] |

### 4.2 Stock video and public-domain footage

**Pexels** (counted through the site's search pages in a browser pane, 8 queries x up to 6 pages; each query tops out at about 140 results; media files were not downloaded):

| Query | Results reachable | Slug-relevant (mechanic, garage, repair, tyre, wheel, wash, lift, engine...) |
|---|---|---|
| car lift | 142 | 80 |
| car service | 140 | 115 |
| tire change | 131 | 96 |
| oil change | 142 | 109 |
| car wash | 144 | 66 |
| car mechanic | 141 | 114 |
| car workshop | 142 | 111 |
| car garage | 143 | 96 |
| **Unique across all** | 623 unique clips | **about 383 by strict keyword, about 514 by loose keyword** |

- The site's headline numbers ("Car Lift 135.4K videos", "Auto Repair Shop 24.9K") are fuzzy semantic matches and are not usable counts. [F]
- Clips that actually show a car on a lift: about 5 ("luxury-car-on-lift-in-modern-garage" 36543594, "luxury-sports-car-on-vehicle-lift-in-garage" 36543596, "mechanics-using-a-surface-mounted-lift-to-work-on-a-car" 14514413, "the-moving-wheels-of-a-car-on-a-lifter" 8986875). About 86 slugs mention tyre/wheel, 26 "under", 11 "hood/bonnet", 107 "wash" (many are hand washing or other). [F counts, I interpretation]
- Content is mostly close-up handheld shots of one mechanic; ID clustering (8986xxx-8987xxx, 14514xxx, 4482xxx) suggests a few prolific uploaders. Not wide fixed-camera bay views. [I]
- **Licence:** the Pexels licence page (Wayback copy of the 2026 page, snapshot date not shown) says all photos and videos are free to use, no attribution needed, modification allowed; not allowed: showing identifiable people "in a bad light", selling unaltered copies, implying endorsement, redistributing on other stock platforms, using as trade-mark. It does not mention ML. [F]
- **Terms of Service** (Wayback copy): "Data mining, extraction, scraping and the use of programs or robots for automatic data collection ... is strictly prohibited for all unauthorised purposes, including without limitation for machine learning purposes." and "Bulk, large-scale or systematic copying of Content is strictly prohibited unless explicit permission has been granted". Uploaders warrant that they hold model/property releases, but the Terms also tell users to check whether consent is needed for identifiable people. Pexels bans generative-AI uploads. [F]
- Class **C**. Manual download of a handful of clips for a visual demo is allowed by the licence, but do not build the training set from them, do not bulk-download, do not commit the clips to the public repo (store IDs/URLs), and ask the organisers whether stock clips with consenting models count as "our data". Note: my counting pages were fetched by a script in a browser pane (about 100 search pages, no media), which the Terms discourage; do not repeat it at scale.

**Pixabay** (first result page of 10 queries, about 100 listed per query):

- Headline counts shown by the site: "Car Repair 2,400+", "Auto Repair Shop 807+", "Garage Mechanic 411+", "Tire Change 376+", "Oil Change 858+", "Car Wash 2,424+", "Vehicle Lift 1,293+". These are fuzzy; "car lift" and "vehicle lift" returned ski lifts, cable cars and forklifts. [F]
- Genuinely relevant clips in the first 40 results of each query: about 20-30 (a "station-service-car-auto-garage-1774xx" series of about 10, two "garage-workshop-mechanic-service", about 6 car-wash clips, "wheels-auto-automotive-tire-service-36656", "garage-vehicle-truck-repair-200276"). [F counts, I judgment]
- Licence summary (Wayback): free use, no attribution needed, can modify; not allowed: standalone sale/distribution, trade-mark use, immoral/illegal use "especially Content which features recognisable people", misleading use. The full Terms contain the same data-mining/scraping/"machine learning purposes" prohibition as Pexels and the same bulk-copying prohibition. Pixabay allows AI-generated uploads (with labelling), so some clips are AI-generated. [F]
- Class **C**.

**Wikimedia Commons** (API, polite use):

| Category | Files | Notes |
|---|---|---|
| Automobile repair shops | 166 files + 16 subcategories | Licences: CC BY-SA 4.0 54, CC BY-SA 3.0 34 (+2 it), public domain 29, CC BY 3.0 16, CC BY 2.0 12, CC0 10, CC BY 2.5 5. Titles include "Old car raised on car lift for diagnostics in workshop", "Car mechanic worker repairing suspension of lifted auto...", "Technician's hand is controlling the lifting button..." |
| Automobile repair | 231 files + 12 subcategories | CC BY-SA 4.0 75, CC BY-SA 3.0 50, public domain 38, CC BY 2.0 23, CC0 10, CC BY 4.0 8... Titles include "A vehicle is lifted on a garage lift", "QuickJack Lift" |
| Automobile maintenance | 150 files + 15 subcategories | public domain 46, CC BY-SA 3.0 45, CC BY-SA 4.0 29, CC BY 2.0 17... includes "Car poised above a maintenance pit in a garage" |
| Garages | 222 files + 14 subcategories | not tallied |
| Vehicle inspection | 142 files | 108 public domain (many US Navy security-check photos with people) |
| Tire shops | 49 files | CC BY-SA 4.0 24, CC BY-SA 3.0 9 ... |
| Car washes | 49 files (43 images, 6 other) + 7 subcategories | CC BY-SA 4.0 16, CC0 8, CC BY-SA 3.0 7, CC BY 4.0 5... |
| Videos of automobiles | 151 files + 29 subcategories | search result for service topics is small, see below |

- Relevant **videos** found by search (licence by file): "Auto mechanic.webm" / "Auto mechanic 2.webm" / "Auto mechanic 4.webm" (CC BY-SA 4.0, 1080p, one vertical), "Minor Car Repair Series Vol. I" (CC BY 3.0, 320x192), "Tire Changer-1.ogv" and "Tire Balancer.ogv" (CC BY-SA 3.0, 640x480), "Tyre Fix Full/Short Video.webm" (CC BY-SA 4.0), "Tire leak soapy water find.webm" (CC0, vertical), car-wash clips: "Timelapse-wasstraat van Slingelandtstraat.webm" (CC BY-SA 4.0), "2019-10-12 car wash Cosy-Wasch ... Berlin.webm" (CC BY 4.0), "Limingantie 5 Oulu 01/02.webm" (CC0), "Waschanlage.ogv" (CC BY 3.0). Fewer than 20 relevant clips in total. [F]
- Class **B/C**: per-file licences are clear, attribution is required, and **CC BY-SA** (about half of the repair-shop images) makes any dataset built from them a share-alike candidate; prefer the CC0/public-domain/CC BY subset (about 72 of 166 in "Automobile repair shops": public domain 29, CC BY 3.0 16, CC BY 2.0 12, CC0 10, CC BY 2.5 5). Screen every image for faces and plates. Good for the qualitative jury-robustness pool and for a held-out "real photo" set of lifts/pits/hoods, not for training at scale.

**Internet Archive:** keyword searches return mostly unlicensed or unrelated uploads (e.g. for "car wash" 415 title hits, nearly all personal uploads without a licence field). The Prelinger collection has about 14 title hits for garage/service station/tyre/motor/lubrication, e.g. "Home Movies: Mobil Service Station Family with Children", "Specialized Lubrication", "America's Safest Tire"; many Prelinger items carry a public-domain licence URL, but they are historical films with real people. Class **C**; not worth the time. [F counts from the advancedsearch API]

### 4.3 Stage 4 synthetic and hybrid options

See section 6 for engines. The stage-4 recipe [I]:
- **Backplates:** Poly Haven CC0 HDRIs rendered as perspective views or as the world lighting in Blender: `auto_service` (rustic workshop with a car in the doorway), `autoshop_01` (large garage with parked cars), `garage`, `skylit_garage`, `abandoned_garage`, `empty_workshop`, `small_workshop`, `machine_shop_01-03`, `university_workshop`, `parking_garage` (65 workshop/garage/parking/hangar-type HDRIs matched my keyword search). The two I opened contain tiny background cars; check for readable plates at full resolution. [F listing, F thumbnails]
- **Lifts:** Sketchfab CC Attribution: "Hydraulic car lift" (23.8k faces), "Car lift" (21.1k, 36.9k, 22.7k faces), "hydraulic_car_lift", "Carlift" (3.3k), "Low-poly parking lift" (5.7k), "Parking Lift", "Renault Megane On Car Lift" (29.3k), "Red Car Lift" (945k).
- **Workshops:** "Car Workshop" (2.4k faces), "Car Workshop Garage" (9.2k), "Car workshop items" (26.5k), "CAR WORKSHOP OFFICE" (273k), "Mexican Auto Repair Shop - Photogrammetry Scan" (160k), "Old Garage Building" (1.4k). All "CC Attribution". [F]
- **States as code:** `lift_height`, `wheel_visible` (hide the wheel mesh), `bonnet_angle`, `person_under` (proxy human), `car_present`, `doors_open`; each state change is a keyframe, so ground truth = the timeline JSON.

### 4.4 Stage 4 recommendation

No real labelled data exists for bay states. Use the Roboflow hood/doors/wheel sets (class B, screened) for the cues that have real images, Commons CC0/PD/CC BY images (about 50-100 after screening) as a real held-out photo set for lift/pit/hood, and a scripted synthetic bay for everything else. Do not present stock clips as a test set.

---

## Stage 5. Wash / ready zone / handover

| Source | Link | Licence | Size | Annotations | Faces/plates? | Class | Tag |
|---|---|---|---|---|---|---|---|
| Roboflow "washing car" | https://universe.roboflow.com/um-84bee/washing-car | CC BY 4.0 | 2,439 images | normal car / open doors / open hood / washing car | Real plate visible in a preview | B | [F] |
| Roboflow "car washing polygonal detection" | https://universe.roboflow.com/bhumins-workspace/car-washing-polygonal-detection | CC BY 4.0 | 751 images | 19 classes: wheel, hood, trunk, bumpers, glass, doors, lights (per-panel polygons) | Not checked | B | [F] |
| Roboflow "Wash Bay Presence" | https://universe.roboflow.com/juanita-yvpi2/wash-bay-presence-i2mx9 | CC BY 4.0 | 36 images | License Plate, Brush, Cars | Real plates | D (tiny) | [F] |
| Roboflow "car wash" (elsoudy) | https://universe.roboflow.com/elsoudy/car-wash-s136f | CC BY 4.0 | 636 images | two classes named "0", "1" | Not checked | B/unclear | [F] |
| Roboflow "car-wash" (emstone) | https://universe.roboflow.com/emstone/car-wash | CC BY 4.0 | 247 images | **smoke, flame** (fire detection; not a wash dataset) | - | Off-domain; the previous report listed it as a wash set | [F] |
| Wikimedia Commons "Car washes" + car-wash videos | https://commons.wikimedia.org/wiki/Category:Car_washes | Per file (see Stage 4) | 49 files; 4-5 relevant videos | none | Screen | B/C | [F] |
| Pexels/Pixabay car-wash clips | see Stage 4 | Pexels/Pixabay terms | Pexels: 66 strict slug matches for the "car wash" query and about 107 slugs containing wash/foam/soap/clean across all 8 queries (many are hand-washing or other); Pixabay about 6 | none | People visible | C | [F] counts, [I] judgment |
| Kaggle (car wash search) | https://www.kaggle.com/datasets?search=car+wash | - | 16 results, all business-listing CSVs | - | - | Nothing | [F] |
| PKLot (UFPR) | https://web.inf.ufpr.br/vri/databases/parking-lot-database/ | CC BY 4.0 | 12,417 images, 695,900 spaces | occupied/vacant per space | Outdoor; plates unlikely readable | A for "slot occupied/empty" pretraining (previous report) | [F] |
| CNRPark+EXT | https://iris.cnr.it/handle/20.500.14243/409744 | **ODbL v1.0** per a third-party dataset card (the previous report found no licence; the official cnrpark.it page was not reachable) | about 150,000 patches, 164 spaces, 9 cameras | per-space labels | Outdoor Italian lot | B until the official licence is confirmed | [S] |
| ACPDS | https://arxiv.org/abs/2107.12207 | MIT (authors) | 293 images | per-space occupancy | Outdoor | A, small | [F]/[S] |
| Indoor parking | - | - | **No open indoor-parking occupancy image set found**; Edge Impulse "cars-in-parking-garage" (290 items, BSD-3) per previous report | - | - | Weak | [S] |
| Moscow Parking Occupancy (HF) | https://huggingface.co/datasets/matrosovdani/moscow-parking-occupancy | CC BY-4.0 | 4.6M half-hourly occupancy snapshots, 210 lots | tabular time series, no images | None | Only if we want realistic arrival/occupancy curves to drive the simulation | [F] |

**Ready zone / handover:** no data exists (no public set shows a customer meeting a car at handover). Handover = car in ready polygon + person within a distance + car leaves, which is a rule on detections; test it on the synthetic timeline only. [I]

**Stage 5 recommendation:** wash state from the Roboflow "washing car" and "car washing polygonal" sets after plate replacement, with 4-5 Commons car-wash videos (CC0/CC BY/CC BY-SA) as a qualitative real test; slot-occupancy logic from PKLot/ACPDS; the rest synthetic.

---

## Stage 6. Simulation / synthetic generation for the whole flow

### 6.1 Engines and asset sources

| Option | Licence | Fit for a service centre with lifts and moving cars | Hardware | Effort for a 1-2 day build [I] | Tag |
|---|---|---|---|---|---|
| **Blender 4.x + Python (bpy)**, Eevee or Cycles | Blender GPL; "what you create with Blender is your sole property" (blender.org licence page) | Best fit: scripted keyframes, object-index pass for auto boxes, any camera model | Any GPU; CPU works | **1.5-2 days for one person** (details below) | [F] licence |
| BlenderProc | GPL-3.0 (GitHub API) | Procedural Blender pipeline: loads obj/fbx/blend, Haven assets, writes COCO/BOP annotations and segmentation | Same | Saves about 0.5 day of labelling code if used; GPL applies to the tool, not to renders [I] | [F] |
| Kubric (Google) | Apache-2.0 | PyBullet + Blender; physics scenes with COCO-style labels; geared to object scenes more than rooms | Same | Similar to BlenderProc | [F] licence |
| NVIDIA Isaac Sim + Omniverse Replicator | Isaac Sim code Apache-2.0 (isaac-sim/IsaacSim LICENSE); "Omniverse Kit SDK ... as well as 3D models and textures" under NVIDIA's separate Additional Software and Materials licence; selling simulation outputs does not need a licence per the FAQ [S] | Environments are warehouses, offices, hospital, simple rooms; **no service bay or car library** [S]; import Sketchfab models | **GeForce RTX 4080 (16 GB) minimum**; GPUs without RT cores (A100, H100) not supported [F] | 2-3 days for a team new to it; not recommended unless someone already knows it | [F] hardware+licence, [S] FAQ |
| CARLA (UE4/UE5) | Code MIT (GitHub API); assets CC BY per docs; Epic's automotive materials follow Epic's terms | Roads, towns, weather presets; **no garages/service stations** in what I found; custom maps need the Unreal editor | GPU 8+ GB class [I] | 1-2 days for a driveway/gate scene only | [F] licence, [S] assets |
| Unity Perception | Licence "Other" (Unity companion licence); last push 2024-11-08 | Labelling package, stale | Any | Not recommended | [F] |
| Unreal Engine / Fab free assets | Epic content licence has a Section 17 on AI (NoAI-tagged content must not be used in generative-AI datasets); the Fab EULA text was not readable | Many car assets, but the training-data terms are unclear | Heavy | Avoid unless the Fab EULA is read | [S] |
| Poly Haven | CC0; FAQ: usable "including training AI models" | 997 HDRIs (301 indoor, 61 night), 521 models (tools, toolboxes, tyre, rim, covered_car), 867 textures incl. `garage_floor`, `hangar_concrete_floor`. No cars. | - | - | [F] |
| Sketchfab (downloadable, CC Attribution / CC0) | Per model; authors may change the licence later and can set a "NoAI" flag against generative-AI use [S] | 269 unique models from my 7 queries; 68 relevant; cars, lifts, workshops (listed in 4.3) | - | Attribution file required; check each model's licence page on the day | [F] listing, [S] NoAI |
| Objaverse | Dataset ODC-By; per-object CC: 721K CC BY, 25K CC BY-NC, 52K CC BY-NC-SA, 16K CC BY-SA, 3.5K CC0 [S]; provenance dispute (scraped from Sketchfab before NoAI) | Thousands of cars, filter to CC BY/CC0 | - | Adds filtering work; Sketchfab direct is easier | [S] |
| Existing synthetic parking/garage datasets | - | Underground car park Blender generator (github.com/Snyard/parkinglot-generator, licence not read) [S]; CARLA parking datasets (ParkingScenes, SunnyParking, CarlaOcc) licences not read; PhysicalAI-SmartSpaces (CC-BY-4.0, no cars) | - | Not a drop-in; no service bay | [S] |

### 6.2 Blender build plan (recommended)

| Block | Work | Time |
|---|---|---|
| Assets | Import one garage shell (Sketchfab "Car Workshop Garage" or a box room with `garage_floor`/`concrete` textures), 2 lifts, 4-6 CC BY cars of different colours, a toolbox/cart set from Poly Haven; set scale; attribution file | 0.25 day |
| Lighting/camera | Poly Haven garage HDRI + a few area lights; 2-3 fixed CCTV cameras with wide-angle/fisheye distortion | 0.25 day |
| Timeline script | `bpy` script that reads a scenario JSON (arrive, park, raise lift, hide a wheel, open bonnet, person under car, lower, wash, ready, leave) and keyframes it; random durations and gaps | 0.5 day |
| Labels | Object-index/cryptomatte pass for 2D boxes per frame; state labels and event times from the scenario JSON; per-frame zone occupancy | 0.25 day |
| Randomisation + plates | Car colour/model, HDRI rotation, light intensity, camera jitter, plate textured quads with synthetic Uzbek plates, noise/blur/JPEG, night/rain variants | 0.5 day |
| Render + QA | Eevee about 1-3 s/frame at 1080p on a consumer RTX [I, not measured]: 10k frames in roughly 3-8 h unattended; visual QA of 100 frames; leakage-safe split by scene/car/HDRI | 0.25 day plus overnight render |

Human proxies are the weakest part: a mechanic under the car and a customer need a rigged humanoid; use a CC BY humanoid from Sketchfab or a plain capsule proxy and say it is a proxy. [I]

### 6.3 Cheaper hybrid: 2D compositing (0.5-1 day)

Poly Haven HDRI perspective crops as garage backplates + car cut-outs (masks from SAM 2, Apache-2.0) from CC0/PD photos pasted on a lift-shaped platform + synthetic plates. Free labels, minimal realism, no 3D. Good for detector robustness; weak for bay-state timelines because there is no real geometry. [I]

### 6.4 Generative video models

| Model | Licence | Notes | Credible use | Tag |
|---|---|---|---|---|
| **Wan 2.2** (T2V-A14B, I2V-A14B, TI2V-5B) | Apache-2.0 (HF API tags + GitHub repo) | TI2V-5B: text+image to video, 720P at 24 fps, "can run on a single consumer-grade GPU such as the 4090"; the model card's example needs at least 24 GB VRAM with offloading, speed table is an image (numbers not read) | Backdrops and appearance augmentation, image-to-video from our own renders to add realism | [F] |
| HunyuanVideo / HunyuanVideo-I2V / 1.5 | Tencent Hunyuan Community License: Territory excludes the EU, UK and South Korea; **"You must not use the Tencent Hunyuan Works or any Output ... to improve any other AI model"** | Uzbekistan is inside the territory, but the output clause forbids training our detector on its clips | Do not use | [F] LICENSE text |
| CogVideoX-5B / 1.5-5B | CogVideoX licence: free for academic research; commercial use needs registration and a "basic commercial licence", with a 1M visits/month cap | CogVideoX-2B is Apache-2.0 (HF tag) | Avoid 5B; 2B is low quality | [F] |
| LTX-Video / LTX-2 | "other" / LTX-2 community licence agreement (text not read) | - | Not evaluated | [F] tag |
| Mochi-1 | Apache-2.0 (HF tag) | Large model | Not evaluated | [F] tag |
| NVIDIA Cosmos (Predict2, Transfer1) | NVIDIA Open Model License, gated | Re-render/continue simulated clips; `PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic` is a CC-BY-4.0 pointer to the Cosmos-Drive-Dreams dataset | Possible weather/lighting variation; heavy | [F] |

**Is it credible as test data? No.** A generated clip has no ground truth, the physics and counts are unreliable (lifts, wheels, plates and text are typically hallucinated), the author cannot say what the model "meant", and a judge can reasonably call it a faked demo. **As training augmentation it is acceptable if the method is declared** (model name, version, prompts, licence) and the clips are only used with weak labels or as unlabelled backgrounds. I found no paper that trains a detector on text-to-video frames and tests on real data (the sim-to-real literature is about 3D rendering and image diffusion). [S/I]

**Effort for a generative-video batch [I]:** 0.5-1 day to set up Wan 2.2 TI2V-5B, 100-200 five-second clips (prompts per state), review and discard bad ones; GPU time per clip not read from the card, measure on the day. Output: unlabelled augmentation clips; labels, if any, come from our own model plus manual check.

### 6.5 Recommended synthesis approach

1. Blender timeline (6.2) as the main generator: it gives exact labels and event timestamps for the whole flow, which is what the test needs.
2. Synthetic Uzbek plates on rendered cars and on plate-replaced real car photos (1.2).
3. Optional: a Wan 2.2 I2V pass on a sample of Blender frames to test whether the detector survives a more photo-real look; report that as an out-of-distribution probe, not as accuracy.
4. Real images (class A/B sets, screened) only for the cues that have them: hood/doors, wheel/rim, wash, tyres, damage.

---

## Stage 7. Test-set strategy without real service-bay footage

### 7.1 Principles

- Report **separate numbers per evidence tier**; never pool synthetic and real numbers.
- Split by source and scene, not by frame: whole HDRIs, car models, camera poses and Roboflow projects are held out of training; time-contiguous blocks only.
- State sample sizes and 95% Wilson intervals. For a proportion near 0.9, n = 50 gives about +/-8 points, n = 100 about +/-6, n = 300 about +/-3.4. [I, standard formula]
- Keep a **failure-case gallery** and say what the system outputs when unsure ("unknown", not a guess).
- Declare every synthetic method in the data list (generator, assets with licences, random seeds, scenario JSON in the repo).

### 7.2 Protocol per stage

| Stage | Tier 1: held-out real stills from public sets | Tier 2: synthetic timeline with known timestamps | Tier 3: consented capture (later) | Metrics |
|---|---|---|---|---|
| 1 Entrance | ZOD frames (vehicle detection under varied light), plate-replaced CC BY car photos for plate OCR (labelled as synthetic plates on real cars), screened Roboflow gate/night sets | Rendered driveway scenes at day/night/rain with synthetic plates, known arrival times | Staff/volunteer cars at a real gate with written consent, real plates of consenting owners | Vehicle recall at fixed IoU, plate exact-match and character accuracy per condition, arrival-time error |
| 2 Inspection | Roboflow damage sets and Kaggle CC0 set held out by project; Kaggle Tyre Quality held-out split; TyreAI OCR test split | Synthetic scratches/DOT strings with known positions | Walk-around of a consenting owner's car | Damage present/absent F1, tyre good/defective accuracy, DOT string exact match |
| 3 Reception | PhysicalAI-SmartSpaces retail scenes (synthetic people) | Blender scene with proxy people entering a lounge polygon | Not planned (face rule) | Person-in-zone count error, dwell-time error; no identity |
| 4 Bay states | 50-100 screened Commons images per cue (lift, pit, hood, wheel off) and the Roboflow hood/doors/wheel sets split by project | Scripted bay, 200+ scenarios with start/end times | 2 days of filming in a dealer bay under a data-processing agreement | Per-state accuracy and confusion matrix, state-change precision/recall within +/-2 s, dwell-time MAE |
| 5 Wash/ready/handover | Roboflow wash sets after plate replacement, 4-5 Commons wash videos, PKLot/ACPDS for slot occupancy | Same timeline, wash/ready/handover events | Same capture | Event precision/recall, end-to-end visit-time error |
| End-to-end | - | 50-100 whole synthetic visits with a ground-truth event log | Pilot | Visit-level: event F1, total time error, missed-event rate |

### 7.3 Claims ladder (what the deck may say) [I]

1. "Tested on N synthetic service visits (method X, assets Y) with exact ground truth: numbers." True and checkable.
2. "Tested on M real still images per cue from public sets (list), with licence and screening notes: numbers." True; the domain gap to dealer CCTV is stated.
3. "Not yet tested on real service-bay CCTV; plan: a 2-day consented capture in the pilot with the acceptance threshold X." Honest.
4. Never: "accuracy on dealer footage" or a single blended accuracy number.

### 7.4 Jury input handling

The jury supplies the input at the demo. Keep an "unsupported scene / low confidence" path, log the model's confidence per state, and run the same code on the jury file as on the test set (no hard-coded answers).

---

## 8. Not found / open questions

- Any public image or video set labelled with car-on-lift, mechanic-under-car, wheel-removed or empty-bay states. Not found after Roboflow (name and `class:` searches), Kaggle, HF, Open Images, LVIS, Objects365, ImageNet-1k, Commons, Internet Archive. Roboflow's `class:` search hit rate limits (HTTP 429) during parts of my session; the follow-up runs that succeeded returned nothing for `class:car_on_lift`, `vehicle_on_lift`, `car_lifted`, `wheel_nut`, `open_hood`, `mechanic_working`, `inspection_pit`; a few earlier batches (e.g. `class:oil_leak`, `class:tyre_change`) were partly throttled and are not fully conclusive.
- Licence of UA-DETRAC, VehiDE, MOTSynth, Objects365, CNRPark+EXT (official), PVDN's source repo, the Fab EULA AI clause, LTX-2 and the Isaac Sim "Additional Software and Materials" licence text.
- Whether the organisers accept public CC BY sets with possible incidental plates/faces after blur/replacement, and whether stock clips with consenting models count as "our data".
- Plate font and exact current plate geometry (unchanged from data_models.md section 9).
- GPU time per clip for Wan 2.2 on our hardware; Blender Eevee time per frame on our GPU.
- Provenance of Roboflow sets (all descriptions empty); BDD100K plate/face blurring status.

## 9. Method notes

- Roboflow Universe pages, Kaggle search pages, Pexels and Pixabay search pages were read in a browser pane (listing text and counts only). Pexels/Pixabay/Roboflow pages were fetched by script for counting; no media or datasets were downloaded. The only large files touched were the Open Images validation label CSVs (about 28 MB and 31 MB, streamed and counted, not stored) and one CarDD licence PDF.
- Licence texts were read from raw LICENSE/README files or HF/GitHub APIs wherever possible. Pexels and Pixabay licence/terms pages were read through the Internet Archive copies (the live pages were behind a Cloudflare check); the snapshot date was not shown on the pages.
- Counts for Pexels/Pixabay are keyword matches on clip slugs, not human-reviewed.

---

## 10. Source list

Roboflow Universe (all pages read in a browser; licence labels are uploader-set)
- https://universe.roboflow.com/carhoodopen/car-hood-open ; /vehicles-ztmmv/car-open-hood ; /deep-learning-ocos/car-condition-detection ; /deep-learning-ocos/car-condition-ue963 ; /um-84bee/washing-car ; /bhumins-workspace/car-washing-polygonal-detection ; /juanita-yvpi2/wash-bay-presence-i2mx9 ; /elsoudy/car-wash-s136f ; /emstone/car-wash ; /bimmerwheels/wheel-tire-detection-ceyns ; /sobirin-bnpsx/carlifting-ppaom ; /egge-public/garage ; /safety-warehouse/dunexa-garage ; /mole-detection-vdnqx/garage-cars-agxxa ; /garage-xwypn/garage-car-14cgx ; /worksho-wvdwu/workshop-zocjw ; /marc-lemoine-si0mt/cars-in-garage ; /ppl-ombde/carwash-repair ; /university-e0xh4/final_project-hazx9 ; /traffic-data-j2xav/parking-study-garage
- Entrance/night/rain: /car-ryzuj/car-night-b9wra ; /autonomousvehicles-bokch/car-night-wihfx ; /carnight/car-night-a59cn ; /ary-eswara-zz0dc/car-cctv-night ; /zhangs-workspace/night-car-track ; /car-5nkvd/rainy-4pkiz ; /nit-jalandhar-euvaa/rainy-vehicle-detection ; /hssafty/vehicle-gate-counter ; /devagan/vehicle-on-gate ; /feyza-axujh/vehicle-gate-detection ; /demo-nbeln/vehicle-entrance-management
- Damage: /skillfactory/car-damage-c1f0i ; /truck-damage/car-damage-dataset-iqcfm ; /ai-proyect/car-damages-kaggle ; /florida-international-university/scratch-dent
- Underbody: /anirudh-qjb2e/undercarriage-rust-detection-f9grj ; /karthik-z0q6k/underbody-car-defects ; /car-segmentation-k5akd/undercarriagecropping ; /copart-i960g/undercarriage-cropping ; /psbhatt-tekion-com/underbody-hard-parts-seg-sk6dq ; /vw-nfqmp/underbody-inspection-system ; /hamza-bashir-nzrcf/oil-and-coolant-leak-drzbm ; /oil-detection/oil-leak-detection ; /tryout/oil-leak
- Tyres: /tyreai/ocr-yphpv ; /dots-tracking/tire-dot-finder ; /subhams-workspace-l47ut/dot-code-6shg0 ; /panida-utamaang/dot-code ; /edgeoos/sams-tires-analyzer ; /kumbhar-saket/tyre-sidewall-text-label ; /tyj-euwyt/tyre-sidewall-text-detection-v2 ; /tyre-defect/tyre-defect-detection ; /vinimagedetection/tire_image_detection ; /bams-workspace/tire-tread-82ob2 ; /multichart/tyre-tread-detection-ulhub
- (prefix every path above with https://universe.roboflow.com)

Kaggle
- https://www.kaggle.com/datasets/warcoder/tyre-quality-classification (CC BY 4.0)
- https://www.kaggle.com/datasets/saralajew/provident-vehicle-detection-at-night-pvdn (CC BY-NC-ND 4.0)
- https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset (CC0)
- Search pages used: https://www.kaggle.com/datasets?search=car+lift ; ?search=car+wash ; ?search=mechanic+workshop ; ?search=tyre&tags=13207-Computer+Vision ; ?search=car+damage&tags=13207-Computer+Vision ; ?search=vehicle+night

Datasets and licences
- CarDD: https://cardd-ustc.github.io/ ; licence PDF https://cardd-ustc.github.io/docs/CarDD_license.pdf ; paper https://arxiv.org/abs/2211.00945
- VehiDE: https://doaj.org/article/6e0eb600059440b390431a42dfe8d138 ; HF copies https://huggingface.co/datasets/hoangnguyenvan/VehiDE-dataset
- BDD100K licence text: https://huggingface.co/datasets/dgural/bdd100k ; repo https://github.com/bdd100k/bdd100k
- DAWN: https://data.mendeley.com/datasets/766ygrbt8y/3
- ExDark: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset ; HF mirror https://huggingface.co/datasets/dronefreak/ExDark
- ZOD: https://huggingface.co/datasets/Zenseact/ZOD ; https://arxiv.org/abs/2305.02008
- UA-DETRAC: https://arxiv.org/abs/1511.04136 ; HF mirror https://huggingface.co/datasets/abhineet123/ua_detrac
- Open Images V7 class lists and label CSVs: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv ; .../oidv7-class-descriptions-boxable.csv ; .../oidv7-val-annotations-human-imagelabels.csv ; .../oidv7-val-annotations-machine-imagelabels.csv
- LVIS class names (Ultralytics config): https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/lvis.yaml ; Objects365: .../Objects365.yaml ; ImageNet-1k class list: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
- NVIDIA PhysicalAI-SmartSpaces: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces ; Cosmos synthetic AV pointer: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic
- PKLot: https://web.inf.ufpr.br/vri/databases/parking-lot-database/ ; ACPDS: https://arxiv.org/abs/2107.12207 ; CNRPark+EXT: https://iris.cnr.it/handle/20.500.14243/409744 ; Moscow parking occupancy: https://huggingface.co/datasets/matrosovdani/moscow-parking-occupancy
- MOTSynth: https://arxiv.org/abs/2108.09518
- Michelin DOT format: https://developer.michelin.com/en/products/tires/dot-image-recognition

Stock and archives
- Pexels licence (Wayback): https://web.archive.org/web/2026/https://www.pexels.com/license/ ; terms: https://web.archive.org/web/2026/https://www.pexels.com/terms-of-service/ ; example searches https://www.pexels.com/search/videos/car%20lift/ ; https://www.pexels.com/search/videos/auto%20repair%20shop/
- Pixabay licence summary (Wayback): https://web.archive.org/web/2026/https://pixabay.com/service/license-summary/ ; terms: https://web.archive.org/web/2026/https://pixabay.com/service/terms/ ; example search https://pixabay.com/videos/search/car%20repair/
- Wikimedia Commons categories: https://commons.wikimedia.org/wiki/Category:Automobile_repair_shops ; Category:Automobile_repair ; Category:Automobile_maintenance ; Category:Vehicle_inspection ; Category:Tire_shops ; Category:Car_washes ; Category:Garages ; Category:Videos_of_automobiles ; https://commons.wikimedia.org/wiki/Category:License_plates_of_Uzbekistan
- Internet Archive advancedsearch API (Prelinger etc.): https://archive.org/advancedsearch.php

Plates and weather
- https://github.com/kstar1996/Car_Plate_DataGenerator ; https://github.com/qjadud1994/Korean-license-plate-Generator ; https://github.com/zabir-nabil/bangla-license-plate-generator ; https://github.com/simonyos/thai-plate-synth ; https://github.com/apereiracv/cr-plates-generator
- Synthetic Ukrainian plates: https://zenodo.org/records/13342103 ; diffusion paper https://arxiv.org/abs/2501.03374 ; Mercosur synthetic plates https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7736924/
- Albumentations: https://github.com/albumentations-team/albumentations ; CARLA: https://github.com/carla-simulator/carla ; weather API https://carla.readthedocs.io/en/latest/python_api/ ; Cosmos-Transfer1: https://github.com/nvidia-cosmos/cosmos-transfer1 ; https://huggingface.co/nvidia/Cosmos-Transfer1-7B
- OpenCV YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Simulation and generation
- Blender licence: https://www.blender.org/about/license/ ; BlenderProc: https://github.com/DLR-RM/BlenderProc ; Kubric: https://github.com/google-research/kubric
- Isaac Sim: https://github.com/isaac-sim/IsaacSim (LICENSE) ; requirements https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html ; licence FAQ https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-faq.html ; Replicator https://docs.omniverse.nvidia.com/py/replicator/latest/index.html
- Unity Perception: https://github.com/Unity-Technologies/com.unity.perception
- Poly Haven: https://polyhaven.com/faq ; API https://api.polyhaven.com/assets?t=hdris ; HDRIs https://polyhaven.com/a/auto_service ; https://polyhaven.com/a/autoshop_01
- Sketchfab API search (downloadable CC): https://api.sketchfab.com/v3/search?type=models&q=car+lift&downloadable=true&license=by ; NoAI note https://sketchfab.com/blogs/community/restricting-generative-ai-use-of-free-models
- Objaverse licence summary: https://pypi.org/project/objaverse/0.0.4
- Epic content licence change log (AI Section 17): https://www.unrealengine.com/en-US/eula-change-log/content
- Wan 2.2: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B ; https://github.com/Wan-Video/Wan2.2 ; HunyuanVideo licence: https://huggingface.co/tencent/HunyuanVideo/raw/main/LICENSE ; CogVideoX licence: https://huggingface.co/THUDM/CogVideoX-5b/raw/main/LICENSE ; LTX-2: https://huggingface.co/Lightricks/LTX-2 ; Mochi-1: https://huggingface.co/genmo/mochi-1-preview
- Synthetic-to-real gap background: https://arxiv.org/abs/2211.16066 ; https://arxiv.org/abs/2510.12208
