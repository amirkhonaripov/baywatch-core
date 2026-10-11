# Service-flow analytics for a car dealer group (Uzbekistan): datasets, Uzbek plates, models, prior work, GPU decode

Research date: 2026-10-10. Scope: public datasets and licences, Uzbek licence plates, open-source models, prior work, GPU video decoding.
Hackathon constraints used throughout: (a) we must list every data source with its licence; (b) real faces and real plates of non-consenting people are forbidden in our data; (c) 1-2 days of build time, one GPU workstation, on-site GPU box in the pilot.

## How to read the evidence tags

- **[F]** = read on the page itself (web page, raw LICENSE file in the repo, Hugging Face API metadata, PyPI JSON, or a page rendered in a browser).
- **[S]** = taken from a search-result summary only; I did not open the primary page. Treat as a lead, verify before relying on it.
- **[I]** = my inference or recommendation.
- "Not found" means I looked and could not find it. It does not mean it does not exist.

## 0. Headline findings (read this first)

1. **There is no public labelled dataset of car service bays / garage CCTV** (bay states, cars on lifts, wheel off, people working). The closest items are small Roboflow sets (car-hood-open, 1,987 images) and generic parking-occupancy sets (PKLot). This is the biggest data gap; we must stage and label our own footage. [F/S, see section 1]
2. **Every public Uzbek plate dataset I found is made of real photos of real vehicles** (Roboflow Uzbek_Plates 3,012 images; Mendeley Central Asia 500 images; the platesmania-derived data behind anpr-uz). They are CC BY 4.0 on paper, but they conflict with the hackathon rule on real plates of non-consenting people. The only clean route is **synthetic plates generated from the format rules** plus staged footage with consent. [F/I, section 2]
3. **Licence traps found while reading primary files** (search summaries had several of these wrong):
   - **DEIMv2 is NOT Apache-2.0.** Its LICENSE.md is a custom non-commercial licence ("No rights are granted for Commercial Use"). [F: raw LICENSE.md]
   - **Ultralytics YOLO11/YOLO26 is AGPL-3.0**, and Ultralytics states an Enterprise licence is needed for internal business tools, embedded/edge deployments, SaaS and fine-tuned models in proprietary settings. [F]
   - **BoxMOT (the popular tracker package) is AGPL-3.0**; the original ByteTrack, BoT-SORT and OC-SORT repos are MIT. [F]
   - **YOLO-World is GPL-3.0**, official YOLOv9 is GPL-3.0, YOLOv10 is AGPL-3.0, nomeroff-net is GPL-3.0. [F]
   - **RF-DETR: only Nano/Small/Medium/Large are Apache-2.0**; Atto/Femto/Pico/XL/2XL are under Roboflow's PML 1.0. [F]
   - **Qwen2.5-VL-3B-Instruct is "qwen-research" licence (non-commercial)**; Qwen3-VL-2B/4B/8B and Qwen2.5-VL-7B are Apache-2.0. [F: HF API]
   - **D-FINE Objects365-pretrained checkpoints "should not be assumed to be commercially cleared"** (D-FINE repo says so itself). [F]
   - **anpr-uz (the only public Uzbek plate OCR model) is CC BY-NC 4.0** and trained on platesmania.com photos with no stated permission. [F]
4. **Uzbek plate facts that matter for the build**: private `01 A 123 BC`, legal entity `01 123 ABC`, 520 x 110 mm, white retroreflective with black characters, Latin letters only, EV plates have a green region-code field, a German-style (FE-Schrift-like) font is claimed but **no official specification was found**. [F/S, section 2]
5. **Recommended core stack is permissive** (Apache-2.0 / MIT / BSD): RF-DETR S/M (Apache-2.0), ByteTrack via `trackers` or `supervision`, fast-alpr family (MIT) with the OCR fine-tuned on synthetic Uzbek plates, Qwen3-VL-2B/4B (Apache-2.0) as label bootstrapper / jury-photo explainer, FFmpeg-NVDEC or PyNvVideoCodec for decode, ONNX Runtime / TensorRT for inference. [section 6]

---

## 1. Datasets for service bays, parking occupancy, tracking/re-ID, and repair activity

### 1.1 Service bays / garages / workshops / car lifts / mechanics (images or video)

| Dataset | What it is | Size / annotations | Licence | Notes | Tag |
|---|---|---|---|---|---|
| Roboflow Universe "car-hood-open" (carhoodopen) | Car with bonnet open vs closed | 1,987 images, 4 versions, 2 classes (`hood_closed`, `hood_open`), object detection | CC BY 4.0 | Useful for the "bonnet open" state. Provenance of images unknown; check for readable plates/faces before use. Updated ~2 months before today. | [F] (rendered page) |
| Roboflow Universe "Car Open Hood" (vehicles) | Classification open/not open hood | 1.09k images | Licence not read | Search listing only | [S] |
| Roboflow Universe "Carlifting" (SOBIRIN) | Vehicle-lane / lifting-style images with classes Tyre, Backlinear, Frontcar, Frontlinear, Midlinear, PlateNo, Stopper | 405 images, 1 version, no model | CC BY 4.0 | No description. Has a `PlateNo` class so images may show real plates. Not a bay-state dataset. | [F] |
| Roboflow Universe "Garage Car" (Garage workspace) | Name suggests garage cars | 301 images, 1 model | Not read | Classes and licence not seen | [S] |
| MIT SUN database "auto mechanics" | Scene images (indoor 45 images / outdoor 17) | 10 + 8 annotated images, 109 + 121 objects | Not checked | Tiny. Research use only guess [I] | [S] |
| VinDynamics "brake rotor replacement" / "alternator disassembly" video datasets | Commercial first-person garage repair video | Listing claims 100.000 hours on both products (looks like a template value), MP4/MOV/AVI + CSV | Commercial, price "Contact" | Voice, plates, identifiers stripped by vendor. Egocentric, not CCTV. Not usable in a 1-2 day hackathon. | [S] |
| Wikimedia Commons "Category:Automobile repair shops" | CC-licensed photos of repair shops | 166 files + 16 sub-categories (counted via the Commons API) | Per-file (CC0 / CC BY / CC BY-SA mostly) | Small, mixed quality, per-file attribution needed; screen for faces/plates manually. Good for a demo-test pool, not for training at scale. | [F] (API count), [I] (use) |
| Wikimedia Commons "Category:Car washes" | CC-licensed car-wash photos | 50 files + 7 sub-categories | Per-file | Same caveats | [F] |
| Unsplash / SuperStock "auto service" stock photos | ~5k photos on Unsplash search; SuperStock story with 145 assets | Unlabelled | Unsplash and SuperStock terms (not read) | Check terms on ML training before use | [S] |

**Verdict [I]:** nothing public matches "fixed ceiling CCTV of a bay with state labels". We need staged footage plus our own labels.

### 1.2 Parking occupancy, indoor parking, car wash, drive-through, vehicle-in-zone

| Dataset | Size / annotations | Licence | Real people/plates? | Use for us [I] | Tag |
|---|---|---|---|---|---|
| **PKLot** (UFPR) | 12,417 images 1280x720, 2 lots (3 camera angles), XML with space coordinates + occupied/vacant, ~695,900 segmented spaces, 4.6 GB | **CC BY 4.0**: "This database is licensed under a Creative Commons Attribution 4.0 International License"; cite Almeida et al. 2015 | Parked outdoor cars from an elevated camera; plates mostly unreadable at that scale [I] | Pretraining/transfer for "slot occupied/empty" (bay empty vs car present). Weak match for a service bay. | [F] |
| PKLot mirrors on Hugging Face (dronefreak/PKLot YOLO format, Voxel51/PKLot) | YOLO-format redistribution | cc-by-4.0 / "cc" tags; dronefreak card says "unofficial redistribution ... under the original CC BY 4.0" | Same | Convenient YOLO format | [F] (card + API) |
| **CNRPark+EXT** | ~150,000 labelled patches, 164 spaces, 9 cameras (Nov 2015 - Feb 2016) | **Not found.** The GitHub README (fabiocarrara/deep-parking) lists zip downloads from cnrpark.it (CNR-EXT patches 449.5 MB) and states no licence | Outdoor Italian lot | Skip unless licence clarified | [F] |
| ACPDS (Action-Camera Parking Dataset) | 293 images at ~10 m height, GoPro Hero 6 (count from search); baseline 98% accuracy on unseen lots | **MIT** (authors share dataset, code, models under MIT) | Outdoor parking | Small; low value | [F] (abstract page), [S] (293) |
| NDISPark (Zenodo) | ~250 images, 7 cameras, COCO annotations, "open" | Not read | Parking | Skip | [S] |
| SNU-SPS | ~3,500 images, indoor+outdoor, 4 labels | Not read | Parking | Possible indoor-parking cue; download availability unconfirmed | [S] |
| Roboflow "washing car" (um-84bee) | Classes normal car / open doors / open hood / washing car | CC BY 4.0 (listing) | Unknown | Closest to "wash" state; size not read | [S] |
| Roboflow "car-wash" (emstone) | Object detection | CC BY 4.0 (listing) | Unknown | Very low usage, no description | [S] |
| Edge Impulse "cars-in-parking-garage" | 290 items, label `vehicle` | 3-Clause BSD | Garage | Tiny, vehicle only | [S] |
| Drive-through / drive-thru vehicle datasets | **Not found** (only a patent family and an arXiv paper that collected 25 vehicles at 10 dealerships for 3D reconstruction, data release unknown) | - | - | - | [S] |

### 1.3 Multi-camera vehicle tracking and re-identification

| Dataset | Content | Licence / access | Verdict | Tag |
|---|---|---|---|---|
| **VeRi-776** | >50,000 images, 776 vehicles, 20 cameras, 1 km2, bounding boxes, types, colours, brands, plates, spatio-temporal data | No formal licence. Obtain by emailing the maintainer; page says information is requested "only to make sure the dataset is used for non-commercial purposes" and the data will not be given to third parties | Non-commercial, by request, includes plates. **Avoid.** | [F] |
| VeRi-Wild | 174 CCTV cameras, one month; plate numbers hidden in released images | Licence not found in the pages I read | Unclear; skip | [S] |
| VehicleID | PKU | A catalogue labels it non-commercial (unverified) | Skip | [S] |
| CityFlow / AI City Challenge vehicle tracks | Outdoor city intersections | **Terms not found** (no aicitychallenge.org page retrieved). | Wrong domain anyway (outdoor) | [S] |
| **NVIDIA PhysicalAI-SmartSpaces** (AI City 2024-2026 MTMC data) | **Synthetic** (Omniverse + Cosmos Transfer), >280 h from ~1,800 indoor cameras (warehouse, hospital, retail), 1080p H.264 30 fps, MOT-style ground truth, calibration; objects are humans, forklifts, pallet trucks, AMRs (no cars) | **CC-BY-4.0** (HF API + card) | No real people. Useful to test multi-camera person tracking logic and zone dwell code, not for cars. | [F] |

**Re-ID for our use case [I]:** a service visit is a small closed world (tens of cars, plate read at the entrance). We can associate cars across cameras by plate + zone transitions + time, and use a generic appearance embedding (DINOv2-small, Apache-2.0; SigLIP2, Apache-2.0) only to disambiguate. See 3.4 for model licences.

### 1.4 Repair / maintenance action datasets (egocentric and third person)

| Dataset | Relevant content | Licence / access | Verdict | Tag |
|---|---|---|---|---|
| **Ego4D** | Hundreds of scenarios; a secondary paper groups "car mechanic" and "getting car fixed" scenarios under "Mechanic" | **Signed licence agreement required, approval before access.** I read the University of Bristol licence text (PDF): permitted purposes include training ML models "to detect, classify, recognise ... at least one object, event, place, or activity", usable for "commercial or noncommercial product development"; licensee keeps rights in models; but "all or any portion of the data ... may not appear in or be visible in any program, dataset, or product" and no redistribution | Licence is actually permissive for model training, but needs a signed agreement + wait for approval, footage is first-person with real faces. **Not feasible in 1-2 days and conflicts with the "no real faces" rule.** Two vendor blogs contradict each other on its commercial status; the primary text above is authoritative. | [F] (licence PDF), [S] (scenario names) |
| Ego-Exo4D | Bike repair (102 takes of 4 repairs), no car repair found | Gated licence system | Skip | [S] |
| Assembly101 | 4,321 videos, people assembling/disassembling 101 toy vehicles | CC-BY-NC-4.0 (Korean catalogue + HF sample repo) | Non-commercial, toy vehicles. Skip. | [S] |
| HOI4D | Egocentric hand-object | Licence not found | Skip | [S] |
| COIN | 11,827 YouTube videos, 180 tasks, 12 domains incl. "vehicles"; I could not confirm a car-tyre task on the project page | Signed agreement from advisor/organisation; YouTube links + JSON annotations on GitHub | Skip | [F] (page), [S] (tyre task) |
| HowTo100M, CrossTask, INRIA Instructional Videos (changing a car tyre, jumping a car) | YouTube instructional | Videos not distributed; licences not confirmed | Skip | [S] |
| Kinetics-700 | Label map contains "changing wheel (not on bike)", "checking tires", "changing gear in car", "vacuuming car", "pushing car" | Annotations/URLs reported as CC BY 4.0 (secondary); the videos stay on YouTube | Not usable for CCTV state recognition; YouTube terms issue below | [F] (label map from mmaction2), [S] (licence) |
| HA4M (manufacturing assembly, 217 videos, 41 subjects), Toyota-SH, MMAct | Industrial/daily activity multi-view | Not read | Wrong domain | [S] |
| MECCANO (industrial-like toy assembly) | Egocentric | Not read | Wrong domain | [S] |

**Verdict [I]:** no usable repair-action dataset. For process stages we do not need action recognition anyway; zone occupancy + bay state + dwell time carry the process.

### 1.5 Can usable training data be scraped legally from YouTube (CC BY garage timelapses)?

- **YouTube Terms of Service forbid it as written.** The Permissions and Restrictions section bars "access, reproduce, download, distribute..." content unless the Service expressly authorises it or YouTube/rights holders give prior written permission, and bars accessing the service "using any automated means (such as robots, botnets or scrapers)" except public search engines obeying robots.txt or with written approval. The section I read does not mention Creative Commons and does not address AI training. [F: youtube.com/t/terms]
- **A CC BY tag does not mention downloading.** YouTube's CC help page says other creators can reuse CC BY work under the licence terms, attribution is required, and CC BY can only be applied to content the uploader has the right to license (a Content ID claim blocks it). The page does not mention downloading CC BY videos. [F: support.google.com/youtube/answer/2797468]
- **No court ruling found** on whether a CC BY tag overrides the platform's anti-scraping terms; there is active litigation over bulk YouTube scraping for AI training (allegations only). [S]
- **Practical position [I]:** (a) the CC licence covers copyright in the video, but the ToS still restrict how we get the file; (b) CC tags can be wrong; (c) garage timelapses typically contain mechanics' faces and customer plates, which the hackathon forbids regardless of copyright; (d) Uzbek law adds consent requirements for personal data (see section 8). **Do not scrape YouTube for training data.** If a creator is contacted and gives written permission, that is a different case. Use YouTube only to watch examples for scene design.

---

## 2. Uzbek licence plates

### 2.1 Plate formats and series (what I could verify)

Characters are **Latin only**; white retroreflective plate, black characters and border; region code on the left, Uzbek flag with "UZ" on the right for the standard types. [F: Wikipedia; goldenpages.uz]

| Plate type (goldenpages type no.) | Layout | Colours | Source | Tag |
|---|---|---|---|---|
| **Private persons (III)** | `DD L DDD LL` e.g. `01 A 123 BC` (region, 1 series letter, 3 digits, 2 letters) | white, black | goldenpages: individuals "one letter, then three numbers, and again two letters"; Wikipedia standard format; plate-recognition-uz regex `^(\d{2})([A-HJ-Z])(\d{3})([A-HJ-Z]{2})$` (matched 148 of 175 sampled platesmania plates) | [F] |
| **Legal entities (I)** | `DD DDD LLL` e.g. `01 123 ABC` (region, 3 digits, 3 letters) | white, black | goldenpages: "three numbers and three letters one after the other"; Wikipedia "alternative" format (does not map it to legal entities); regex `^(\d{2})(\d{3})([A-HJ-Z]{3})$` (19 of 175) | [F] |
| **Electric vehicles** (III-E individuals, I-E legal) | Same character patterns, with the **region-code field on a green background**. Rule: electric motor >= 4 kW and >= 40 km/h; issuance began Nov 2021; extended to hybrids in March 2023 | white with green region box | goldenpages type table; spot.uz 2021 | [F] (green region box, 4 kW/40 km/h), [S] (hybrids 2023) |
| EV plate pattern conflict | The anpr-uz model card and plate-recognition-uz describe EV plates as `DD M DDDDDD` (e.g. `01 M 018461`), fully green. gov.uz says series **M** = staff of foreign firms/investors, and Wikipedia calls "M" plates foreign companies, green background/white characters. | - | **Unresolved. Do not trust the `M DDDDDD` EV format until we sample real plates.** | [F] conflict |
| Ambulances (I-2 "STY"), electric/hybrid ambulances (I-E2) | Same layout; I-E2 has green region field | white | anhor.uz, goldenpages; introduced by a July 2024 resolution | [F] |
| State bodies | Types I-1 "DAV" (public administration), II (higher state bodies, ministries). Letter series: PAA, PSS (presidential), PPP (Oliy Majlis), APC (ministries), VSB.../PSF (interior), TSS/TST (state security), ZDF (prosecutor), DXA (justice), FVV (emergencies), **TTA-TTG (taxis)** | white | gov.uz advice page; goldenpages | [F] |
| Foreign citizens / stateless (X) | "H" series, one letter + 6 digits per the plate-recognition-uz format `DD H DDDDDD` | yellow, black | goldenpages; Wikipedia; plate-recognition-uz | [F] |
| Diplomatic | Prefix CMD (heads of mission), D (diplomats), T (technical staff) + 3-digit mission code (001 Turkey ... 064 UAE); UN series | CMD/D/T: green with white characters; UN: blue with white | Wikipedia; goldenpages types IV, V, VI | [F] |
| Motorcycles (XI) and electric motorcycles (XI-E) | Square, 3 digits on top, 2 letters below, region code at upper right; XI-1 mopeds and XI-E1 scooters two-line; XI-2 "HH", XI-3 "HHH" yellow for foreign-owned | white (green region field for electric) | Wikipedia; goldenpages | [F] |
| Trailers | Square two-line: 4 digits on top, 2 letters below; the region code comes last, e.g. `4169 AA 70` | white | Wikipedia; plate-recognition-uz (8 of 175 sampled) | [F] |
| Temporary / transit for exported vehicles | "T" in the left part of the plate | not found | Wikipedia only, and that page carries a "no citations" banner | [F]/[I] weak |
| Military, police | **Not found** | - | - | - |

Operational notes: the 2024 resolution moved issuance to random selection from 1 Aug 2024; since 1 Jan 2026 sale through the commodity exchange ended and plates are issued through the AvtoRaqam app; special combinations are bought in the app. The articles do not say the physical plate design changed. [F: kun.uz 2026-03-24; S: 2024 articles]

Letter alphabet: the plate-recognition-uz regexes exclude the letter `I` (`[A-HJ-Z]`) which is an empirical rule from a platesmania sample, not an official statement. [F] about the regex, [I] about the rule. Use it only as a soft validator.

### 2.2 Dimensions and font

- **Size: 520 mm x 110 mm** (Wikipedia, which carries a "lacks sources" banner; I found no official standard or resolution text that confirms it). Kyrgyzstan's Wikipedia article lists the same size, so the two are easy to confuse. [F]/[S]
- **Font: not officially confirmed.** goldenpages.uz says: "a special German font that makes falsification difficult", with a slotted zero to distinguish it from O. [F] The English Wikipedia FE-Schrift article lists Uzbekistan among countries using FE-Schrift or a derivative. [F] The German Federal Highway Institute (BASt) never published digital versions; free third-party digitisations exist (e.g. "Euro Plate" from andyhoppe.com), licence unverified. [F]
- **[I] Working assumption:** the characters look like FE-Schrift or a close derivative. Test by rendering candidate fonts next to a handful of consented real photos; check P/R, B/8, 0/O, F/E, 6/G glyph shapes. Do not assume the font is DIN 1451 (the predecessor, proportional, not machine-readable-by-design).
- I did not find the Cabinet of Ministers text that would settle size and font. Leads: Resolution No. 131 of 27.06.2007 (issuance rules) and No. 429 of 17.07.2024 (new plate types), both on lex.uz. The lex.uz document I opened (No. 196 of 2007) only amends registration rules and has no dimensions. [F]

### 2.3 Region codes (current system) [F: goldenpages.uz]

| Range | Region | Previous code |
|---|---|---|
| 01-09 | Tashkent city | 10 / 30 |
| 10-19 | Tashkent region | 11 |
| 20-24 | Syrdarya | 12 |
| 25-29 | Jizzakh | 13 |
| 30-39 | Samarkand | 14 |
| 40-49 | Fergana | 15 |
| 50-59 | Namangan | 16 |
| 60-69 | Andijan | 17 |
| 70-74 | Kashkadarya | 18 |
| 75-79 | Surkhandarya | 19 |
| 80-84 | Bukhara | 20 |
| 85-89 | Navoi | 21 |
| 90-94 | Khorezm | 22 |
| 95-99 | Karakalpakstan | 23 |

Wikipedia lists only the base code per region (01, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95), consistent with the ranges. plate-recognition-uz assumes region codes 01-95. [F] Dealer group in Tashkent probably sees 01-09 and 10-19 most often [I].

### 2.4 Public Uzbek plate datasets

| Dataset | Content | Size | Licence | Concern | Tag |
|---|---|---|---|---|---|
| Roboflow Universe **Uzbek_Plates** (Suxrobs Workspace) | Object detection, one class `plate`, 1 hosted model | **3,012 images** in the project overview (a version listing in search showed 8,434, likely augmented); updated Mar 2026; no description | **CC BY 4.0** | Real photos of real vehicles; origin of images not documented | [F] (rendered page) |
| Roboflow "Uzbek car license number OCR" | Character-level detection classes (digits, A...) | 40 images, 2 models | CC BY 4.0 (search summary) | Tiny; page returned 403 to my fetch | [S] |
| Mendeley Data "Central Asia vehicle number plate with Annotations" (Chazhabaev, Sadyk, Bazarkulova; v2, 2 Apr 2024) | 5 country classes: KZ, KG, **UZ**, TJ, TM; JSON annotations; JPG 964x964; 80:20 split | **500 images total**, per-country counts not stated | **CC BY 4.0** | Real plates. One search summary describes Uzbek plates there as "three letters and five digits", which contradicts the formats above; verify before using | [F] |
| anpr-uz (HF, IzzatulloOne) | OCR model, TPS-ResNet-BiLSTM-CTC, 32x100 grayscale, single-line plates only | trained on photos and labels scraped from platesmania.com/uz, hand-cleaned; ~90.7% private+legal, 2.8% foreign, 5.8% trailers | **CC BY-NC 4.0** (model); service code MIT | NC weights, scraped data with no stated permission; two-line trailer plates read 0 of 4. Self-reported 72/80 (90%) exact match on 80 white-plate photos; coloured plates 30/39 after format rules | [F] |
| Roboflow "Carlifting" | has a `PlateNo` class, country unknown | 405 images | CC BY 4.0 | see 1.1 | [F] |
| Kaggle, GitHub raw datasets for Uzbek plates | **Not found** (plate-recognition-uz README says no ready Uzbek YOLO model exists on HF/Roboflow and the only Uzbek Roboflow project had 40 images) | - | - | - | [F] |
| Nomeroff-net / AUTO.RIA dataset | OCR sets for UA, EU, RU, KZ, GE, BY, "SU (exUSSR)", KG, AM; **Uzbekistan not listed**; dataset licence not stated | - | code GPL-3.0 | - | [F] |

**Rule check [I]:** Uzbek_Plates, the Mendeley set and anything scraped from platesmania contain real plates of non-consenting people. Unless the organisers explicitly accept public CC BY datasets, we should not put them in our data list. Ask the organisers; until answered, plan without them.

### 2.5 Open-source ALPR projects (licence, Uzbek/CIS support)

| Project | Licence | What it gives | Uzbek/CIS | Tag |
|---|---|---|---|---|
| **fast-alpr** (ankandrew) | MIT (raw LICENSE); v0.4.0, 15 Mar 2026; 825 stars | ONNX pipeline: plate detector + OCR, backends CPU/CUDA/OpenVINO/DirectML/QNN | Not mentioned on the page | [F] |
| **open-image-models** | MIT; v0.6.0, 27 Jul 2026 | Plate detectors `yolo-v9-{t,s}-*-license-plate-end2end` (mAP50 0.858-0.966 depending on size, mAP50-95 0.606-0.772; e.g. `yolo-v9-t-640`: P 0.966, R 0.896, mAP50 0.958) and RF-DETR COCO detectors (nano/small/medium/large) | Training data of the plate detectors **not stated**; YOLOv9 lineage, so weights provenance is unverified | [F] |
| **fast-plate-ocr** | MIT; v1.1.0, 14 Mar 2026 | CCT-S/XS "global" OCR models: `cct-xs-v2-global-model` 0.47 ms b=1 (2,144 plates/s), `cct-s-v2` 0.68 ms (1,480 plates/s), measured on an RTX 3090 (TensorRT/CUDA/CPU execution providers benchmarked); legacy `global-plates-mobile-vit-v2-model` = "plates from +65 countries, 85k+ plates", 93.3% accuracy; Keras 3 training (TF/JAX/PyTorch), export to ONNX/TFLite/CoreML; **fine-tuning tutorial notebook** | Country list not published; Uzbekistan coverage unknown | [F] |
| Nomeroff-Net | **GPL-3.0** (raw LICENSE); v4.0.0; YOLOv8 + custom RNN OCR; says classifier is "configured mainly for ... Ukrainian numbers" | CIS countries but not UZ | [F] |
| plate-recognition-uz (IzzatulloOne) | Code MIT; depends on Ultralytics YOLO11 (AGPL, the README warns about it); weights carry the authors' own terms; includes Uzbek format rules (`plate_rules.py`), voting across frames, FastAPI | Uzbek only | [F] |
| PaddleOCR | Apache-2.0 (raw LICENSE) | General OCR; could be fine-tuned on plate crops | Not plate-specific | [F] |
| RT-DETRv2 plate detector `justjuu/rtdetr-v2-license-plate-detection` | model tagged Apache-2.0; trained on `justjuu/license-plate-detection` (6,176 train / 1,765 val / 882 test images), mAP 0.97 on its own test split; dataset licence not stated | country mix unknown | [F] |
| Many HF "MIT" YOLOv8/YOLO11 plate detectors | Often fine-tunes of Ultralytics (AGPL) with a different label | - | [I]: do not trust the label; AGPL lineage |

### 2.6 Synthetic plate generators

- **No open-source generator for Uzbek plates found.** [not found]
- Templates to adapt: `kstar1996/Car_Plate_DataGenerator` (Korean plates; scripts for plain, brightness, perspective, rain/sun/shadow noise, and pasting onto car photos); licence not read. [S]
- Research evidence that synthetic helps: a diffusion-generated Ukrainian plate set (10,000 images on Zenodo) improved LPR accuracy by about 3% over baseline when added with pseudo-labels. [S]
- fast-plate-ocr itself ships an Argentinian model trained with synthetic plates (`argentinian-plates-cnn-synth-model`, 94.19% vs 94.05% for the non-synthetic one), and has a fine-tuning notebook. [F]
- **[I] Build plan (about half a day):** Python + OpenCV/PIL generator from the formats in 2.1 (plate types III, I, EV-green region box, trailer two-line, H yellow), 520:110 aspect, region code left, flag + "UZ" right; character strings sampled from the region-code ranges in 2.3 with invented numbers (never copy real plate strings); perspective warp, blur, noise, JPEG, glare, dirt, night IR look; label = text. Also composite synthetic plates onto staged car photos (replace the plate area) for the detector. Font: try several FE-style candidates and verify each font's licence before shipping; keep the generator and fonts out of the product if the licence is unclear.

---

## 3. Models

All latency numbers below are **T4, TensorRT, FP16, batch 1**, which is what the vendors publish. I found **no published numbers for consumer RTX 30/40-series cards** for these detectors; RTX cards are generally several times faster than a T4 [I] but benchmark on our own box before promising anything.

### 3.1 Detectors (licence matters for the commercial pilot)

| Model | Licence | COCO AP50:95 / T4 latency | Notes | Tag |
|---|---|---|---|---|
| **RF-DETR Nano / Small / Medium / Large** | **Apache-2.0** (repo LICENSE; README) | 48.4 / 2.3 ms; 53.0 / 3.5 ms; 54.7 / 4.4 ms; 56.5 / 6.8 ms (30.5-33.9M params; 384, 512, 576, 704 px) | Roboflow-measured with pycocotools in its own benchmark (SAB); vendor-run. DINOv2 backbone. Also segmentation and keypoint (preview). `pip install rfdetr` v1.11.2 (4 Oct 2026). Atto/Femto/Pico/XL/2XL = PML 1.0, not Apache | [F] |
| **D-FINE N/S/M/L/X** | **Apache-2.0** | N 42.8 / 2.12 ms (4M params); S 48.5 / 3.49; M 52.3 / 5.62; L 54.0 / 8.07; X 55.8 / 12.89 (official zoo) | COCO-only checkpoints are the safe ones; **Objects365-pretrained (`*_obj365`, `*_obj2coco`) not commercially cleared**. HF Transformers port `ustc-community/dfine-*` is Apache-2.0 | [F] |
| DEIM (v1) | Apache-2.0 (LICENSE header) | 54.7 / 8.1 ms (L), 56.5 / 12.8 (X) per search | Fine | [F]/[S] |
| **DEIMv2** | **Custom non-commercial licence (2026)**; "No rights are granted for Commercial Use"; weights equally restricted; commercial licence from Intellindust (shenxi@intellindust.com) | S: 50.9 AP, 9.71M params (paper) | **Exclude from the pilot.** Several search summaries called it Apache; the repo says otherwise | [F] |
| LW-DETR T/S/M/L/X | Apache-2.0 (as listed in the RF-DETR README) | T 42.9 / 1.9 ms; S 48.0 / 2.6; M 52.6 / 4.4 | Alternative | [F] |
| RT-DETR / RT-DETRv2 | Apache-2.0 (repo LICENSE; HF `PekingU/rtdetr_r50vd` Apache-2.0) | not re-measured here | Solid, older | [F] |
| **YOLOX** | Apache-2.0 (repo LICENSE) | not re-measured | Old but permissive, ONNX/TensorRT well trodden | [F] |
| **Ultralytics YOLO11 / YOLO26** | **AGPL-3.0**; Enterprise licence for closed/commercial use, including internal tools, embedded/edge, SaaS, and fine-tuned models; price on request | YOLO26 N/S/M/L/X: 40.3/1.7 ms, 47.7/2.6, 52.5/4.4, 54.1/5.7, 56.9/9.6 (Roboflow-measured); YOLO11-N 37.4/2.5 | Best tooling, but licence cost or open-sourcing obligation for a dealer-group product | [F] |
| YOLOv9 (official) GPL-3.0; YOLOv10 AGPL-3.0 | copyleft | - | Exclude | [F] |
| YOLO-World | **GPL-3.0** (repo says "supported for commercial usage", commercial contact exists) | LVIS AP 35.4 at 52 FPS on V100 (paper abstract); model zoo S-X LVIS AP 18.5-28.7; TensorRT "coming soon" | Fine as an offline label bootstrapper (outputs are labels, not code); avoid shipping | [F] |

**Recommendation [I]:** RF-DETR-S/M as the main detector (best AP-per-ms among Apache-2.0 options in the vendor table, easy fine-tuning, ONNX/TensorRT export); D-FINE-S/M (COCO weights) as the fallback if RF-DETR fine-tuning disappoints. YOLOX if we need something small and boring.

### 3.2 Trackers

| Tracker | Licence | MOT17 HOTA (roboflow `trackers`, default params) | Notes | Tag |
|---|---|---|---|---|
| ByteTrack (`ifzhang/ByteTrack`) | MIT | `trackers` ByteTrackTracker: 60.1 (SportsMOT 73.0, SoccerNet 84.0, DanceTrack 50.2) | Two-stage association; no appearance model; best default for cars that are slow and well separated | [F] |
| OC-SORT (`noahcao/OC_SORT`) | MIT | 61.9 (DanceTrack 51.8) | Better after occlusions, e.g. people behind a lift | [F] |
| BoT-SORT (`NirAharon/BoT-SORT`) | MIT | not in the `trackers` table | Adds camera-motion compensation + optional ReID; our cameras are static so CMC adds little [I] | [F] |
| SORT | via `trackers` | 58.4 | Baseline | [F] |
| `roboflow/trackers` package | Apache-2.0; v2.6.1 (25 Sep 2026); CLI accepts video, webcam, RTSP | SORT, ByteTrack, OC-SORT | Plug into RF-DETR via supervision | [F] |
| `roboflow/supervision` | MIT | includes ByteTrack, zones, line counters | Polygon zones for entrance/bay/wash are built in | [F]/[I] |
| **BoxMOT** | **AGPL-3.0** | - | Avoid; also Ultralytics' bundled trackers | [F] |

### 3.3 Zero-shot / open-vocabulary models for bootstrapping labels

| Model | Licence | Size | Notes | Tag |
|---|---|---|---|---|
| Grounding DINO (IDEA-Research; HF `grounding-dino-tiny/base`) | Apache-2.0 (repo + HF API) | 172M / 233M params | Text-prompted boxes, e.g. "car on lift", "open bonnet". Speed numbers not found | [F] |
| OWLv2 (`google/owlv2-base-patch16-ensemble`, large) | Apache-2.0 (HF API) | 155M (base) | Open-vocab + image-conditioned queries; speed not found | [F] |
| YOLO-World | GPL-3.0 | - | see 3.1 | [F] |
| Florence-2 base / large (Microsoft) | MIT (HF API) | 232M / 777M | Captioning, grounding, OD, OCR; fast and small | [F] |
| SAM 2 / 2.1 | Apache-2.0 (repo + HF) | Tiny 38.9M ... Large 224.4M | Per README, A100 with `torch.compile`: Tiny 91.2 FPS, Small 84.8, Base+ 64.1, Large 39.5 (video segmentation). Good for mask labels and for tracking an object through frames during annotation | [F] |
| SAM 3 | Custom "SAM License" (gated on HF; HF API says `other`) | ~850M (paper) | Not Apache; read the terms (includes indemnification and termination clauses per a search summary) before any use | [F]/[S] |

**[I] Bootstrapping workflow:** run Grounding DINO / OWLv2 / Florence-2 on staged frames with prompts such as "car", "person", "raised car lift", "open car hood", "wheel", then correct by hand in CVAT (MIT) or Label Studio (Apache-2.0) [licences of these two tools not re-verified here]. Fine-tune RF-DETR on the corrected labels.

### 3.4 Vehicle re-ID models

- Code: **FastReID** Apache-2.0 [F]; **TransReID** MIT [F]; **CLIP-ReID** MIT [F]; deep-person-reid (Torchreid) MIT [F] but person-only.
- **Pretrained vehicle weights with a clean commercial licence: none found.** TransReID expects ImageNet backbones and user-trained checkpoints; several community weights on HF have zero downloads and were trained on VeRi-776 (non-commercial data). [S]/[F]
- **[I] Do not build a vehicle re-ID model.** Use plate + zone transitions + time windows; if appearance is needed, use a frozen generic embedding such as `facebook/dinov2-small` (22M params, Apache-2.0 [F]) or `google/siglip2-base-patch16-224` (Apache-2.0 [F]); avoid `facebook/dinov3-*` (custom DINOv3 licence, gated [F]).

### 3.5 Small VLMs that classify bay state from a crop

| Model | Licence | Params | Notes | Tag |
|---|---|---|---|---|
| **Qwen3-VL-2B-Instruct** | Apache-2.0 (HF API) | 2.13B | Released Oct 2025; strong OCR/doc/video claims; a forum post measured 0.89 QPS (transformers) / 0.53 QPS (llama.cpp) on Jetson Orin Nano Super for a 1280x720 image + 100-200 text tokens (prefill-only) | [F]/[S] |
| **Qwen3-VL-4B-Instruct** | Apache-2.0 | 4.44B | A setup guide estimates 8-9 GB VRAM at FP16 and ~3.5 GB at Q4; vLLM H200 forum numbers are saturated-server figures, not single-request latency | [F]/[S] |
| Qwen2.5-VL-3B-Instruct | **qwen-research (non-commercial)** | - | Exclude. Qwen2.5-VL-7B-Instruct is Apache-2.0 | [F] |
| SmolVLM2 2.2B / 500M / 256M | Apache-2.0 | 2.2B / 507M / 256M | Model card: 2.2B needs "only 5.2GB of GPU RAM for video inference"; Video-MME 52.1 (2.2B), 42.2 (500M), 33.7 (256M); no speed numbers published | [F] |
| Florence-2 base/large | MIT | 232M / 777M | See 3.3 | [F] |
| InternVL3.5-2B | Apache-2.0 (HF API) | - | Not benchmarked here | [F] |
| Gemma 3 / 3n | Gemma terms (not OSI) | - | Check terms before commercial use | [F] |

**Speed on a consumer RTX:** I found **no** credible measured numbers for Qwen3-VL-2B/4B or SmolVLM on RTX 4090/3090 for short-label classification. Benchmark ourselves (batch 1 and batch 8, a few image resolutions). For a short answer, time is dominated by image prefill, so image resolution is the main knob [S]/[I].

**[I] How to use them:** do **not** run a VLM per frame per bay. Use the VLM (a) offline to bootstrap labels on staged footage, (b) on demand for the jury's single photo ("describe bay states"), and (c) as a low-rate verifier. For live bay state, run a tiny classifier on the bay crop (frozen DINOv2-small or SigLIP2 embedding + linear head, or fine-tune a small CNN) trained on ~100-300 labelled crops per state.

---

## 4. Prior academic and industry work on bay states / service-process stages from video

**Academic (no direct match):** I found no peer-reviewed paper on recognising service-bay states or car-repair stages from CCTV. Adjacent work:

- **Video process mining**: Kratsch, Koenig and Roeglinger, "Shedding light on blind spots", arXiv 2010.11289 (Decision Support Systems): reference architecture; prototype automatically extracts **more than 70% of process-relevant events** from a real video dataset in a supervised scenario. [S] (abstract-level)
- RAVEE (Woerdehoff et al., Decision Support Systems 2025): unsupervised discovery of process steps from video. [S]
- **SnapLog** (Pegoraro, Seng, Heller, van der Aalst, Kersting; arXiv 2604.22476, 2026): frame embeddings + temporal segmentation + few-shot labels. I read the PDF: evaluated on the **TUM Kitchen and EPIC-KITCHENS-100** datasets, not workshops; **frame-wise segmentation accuracy 74.3% (TUM) and 67.7% (Epic Kitchens)**; code at github.com/J0nasSeng/SnapLog (licence not checked). [F]
- Egocentric factory event-log preprint (Preprints.org, Jul 2026; six factories from Egocentric-10K, no task-specific training): not peer-reviewed. [S]
- Passenger-car repair process mining with Celonis (Bitkowska et al., Foundations of Management 2020): event logs from the dealer system, no video. [S]
- Parking occupancy classification: PKLot (Almeida et al. 2015) and ACPDS (arXiv 2107.12207; baseline 98% accuracy on unseen lots, MIT licence). [F]/[S]
- Dealership drive-through 3D reconstruction (arXiv 2603.26638; 14-camera rig in service lanes of 10 dealerships, 25 vehicles, PSNR 28.66 dB): not a state-recognition paper, but shows dealership service-lane camera rigs exist. [S]

**Industry (vendor claims only, no independent accuracy numbers found):**

- **Leverege PitCrew**: runs on existing bay cameras + a small edge appliance; bay utilisation, idle time, bay time, labour minutes per service, delay-cause logging, delayed-service SMS alerts; privacy: no facial recognition, faces/full bodies blurred on edge and cloud. Vendor-reported: +30 monthly services per store, 11% fewer delayed services, 13% shorter service times, up to 8 more services/day at peak. No accuracy published for the AI outputs. [F: leverege.com]
- **IoT For All article**: vehicle arrivals/departures, service start/end, technician transitions and idle periods, bay utilisation; no accuracy numbers, no models named. [F]
- **SKAIVISION** (service-lane dwell time, missed handoffs), **EyeQ**, TVI: marketing pages only. [S]
- A USPTO application on tracking a vehicle across a dealership camera network exists (No. 12718563); the PDF is a scanned image and I could not read it. [S]
- Reported background issue: legacy IP/analog dealership cameras give footage too poor for activity recognition and a single customer's path spans several cameras. [S]

**Takeaway [I]:** the market already sells exactly this concept with bay-level states, so our differentiator is the Uzbek context (Uzbek plates, on-site box, local dealer KPIs) and an honest accuracy report on staged data. Report our own accuracy per state with a confusion matrix; competitors publish none.

---

## 5. GPU video decoding and export notes

### 5.1 Decoding options

| Option | What it is | Licence | RTSP / live | Pairing with Python/PyTorch | Tag |
|---|---|---|---|---|---|
| **FFmpeg `-hwaccel cuda` (NVDEC) in a subprocess** | One ffmpeg process per camera; flags such as `-rtsp_transport tcp -fflags nobuffer -flags low_delay`; `scale_cuda` + `hwdownload` to rawvideo pipe | FFmpeg (LGPL/GPL by build); NVIDIA codecs proprietary | Yes (RTSP is native to FFmpeg) | Pipe gives host memory, so a GPU->CPU->GPU copy; cheap at 2-10 fps analytics rates. Reported: TCP + nobuffer + low_delay cut latency from 1.5 s to 0.1 s (forum report) | [S]/[I] |
| **PyNvVideoCodec 2.2.3** (NVIDIA; PyPI, 15 Sep 2026) | Python bindings to Video Codec SDK; `SimpleDecoder`, `ThreadedDecoder` (background prefetch), low-level `CreateDecoder` with manual demux; supports H.264, HEVC, VP9, AV1 decode; zero-copy to PyTorch via DLPack (`torch.from_dlpack(frame)`) | MIT (PyPI metadata) | **Docs do not mention RTSP or URLs.** SimpleDecoder takes file paths, bytes, memoryview, seekable streams. Callback-based demuxing exists "for non-seekable network streaming". One forum thread reports RTSP decode returning no frames with the C++ SDK | Best fit for **jury video files** and for offline training-data decode; for live RTSP, demux with PyAV/FFmpeg and push packets to the low-level decoder (untested, [I]) | [F] |
| **TorchCodec 0.17.0** (PyTorch; 30 Sep 2026) | `VideoDecoder(..., device="cuda")` decodes via NVDEC into CUDA tensors | BSD-3 (CUDA build also under NVIDIA CUDA licence) | Page says nothing about network streams | Native PyTorch tensors; requires FFmpeg with `h264_cuvid`; torchaudio's StreamReader/torio (RTSP-capable) was deprecated in 2.8 and slated for removal in 2.9 with decoding moving to TorchCodec | [F]/[S] |
| **NVIDIA DeepStream 9.0 / 9.1** | GStreamer pipeline: decode, `nvstreammux` batching, `nvinfer` (ONNX/TensorRT), tracker, RTSP out | NVIDIA SDK EULA (proprietary, free); TAO models under NVIDIA AI Product License | Yes, designed for many RTSP sources | Python via pyservicemaker wheels; custom detectors need a bbox parser library (YOLO sample parser exists; DETR-style outputs need a custom parser [I]). Heavy setup; forum reports at 16-80 RTSP streams show decoder saturation tuning needed (batch size, `interval`) | [S] |
| **NVIDIA DALI 2.3.0** | Decode + augment pipelines | Apache-2.0 | File-oriented; no live RTSP found | Good for training data loading, not live ingest | [F]/[S] |
| OpenCV `cv2.VideoCapture` (FFmpeg) | CPU decode | Apache-2.0 | RTSP yes | Simple fallback | [I] |

**NVDEC capacity:** NVDEC engine counts per GPU: **RTX 4090: 2, RTX 3090: 1, A100: 5, T4: 2**, L40G 3. On an L40G (3 NVDECs) PyNvVideoCodec decoded 1080p H.264 sequentially at about 2,600 fps total with 3 threads (about 880 fps per NVDEC; synthetic clip). The SDK 12.2 application note (as reported by a search summary) lists the RTX 4090 at 883 fps for H.264 1080p. Concurrent NVDEC session count is not capped on consumer cards; the NVENC cap was the limited one (search summaries disagree; one APH Networks line says 1 stream, which contradicts Tom's Hardware coverage and I treat as an error). [F]/[S]
Arithmetic [I]: 883 fps / 25 fps = ~35 concurrent 1080p25 H.264 streams per RTX 4090 in the best case. A service centre with 8-32 cameras is therefore limited by RTSP ingest, CPU demux, VRAM and the detector, not by NVDEC.

**[I] Recommendation:** for the hackathon, one ffmpeg-NVDEC subprocess per camera (or PyAV) with frame skipping to 5 fps, newest-frame-wins queues, plus **PyNvVideoCodec or TorchCodec for the jury's file input**. Keep DeepStream as the scale-up path for the pilot only if we exceed ~16 cameras per box.

### 5.2 TensorRT / ONNX export notes

- **RF-DETR**: `rfdetr[tensorrt]` for the raw `.trt` path (`TRTInference`); ONNX for ONNX Runtime/OpenVINO/TensorRT; custom resolution must be divisible by `patch_size x num_windows`; INT8 is dynamic-range (no calibration data). On an L4, fp16 TensorRT run via rfdetr's `TRTInference` was 9.82 ms end-to-end (RFDETRSmall, batch 1) vs 3.41 ms with the `inference-models` managed engine; the vendor notes the two harnesses are not directly comparable. [F: rfdetr docs]
- **D-FINE**: `pip install onnx onnxsim`; `python tools/deployment/export_onnx.py --check -c <cfg> -r model.pth`; `trtexec --onnx=model.onnx --saveEngine=model.engine --fp16`; inference scripts `onnx_inf.py`, `trt_inf.py`; benchmark `trt_benchmark.py`; latencies from TensorRT 10.4.0. [F]
- **Ultralytics** (if we ever license it): built-in `export(format="onnx"|"engine")`; DeepStream needs correct `cluster-mode` for raw vs post-NMS output (community guidance). [S]
- **YOLO-World**: ONNX export documented, TensorRT "coming soon". [F]
- **Runtime packages (PyPI, 10 Oct 2026):** `onnxruntime-gpu` 1.31.0 (MIT); `tensorrt-cu12` 11.4.0.106 (PyPI classifier "Proprietary"); TensorRT OSS repo Apache-2.0. [F]
- **[I] General:** export with fixed input size and FP16 first; check that post-processing (DETR-style: top-k over queries, no NMS) is inside or outside the engine consistently; validate ONNX vs PyTorch outputs on 50 frames before trusting any speed number.

---

## 6. Recommended stack (licence compatibility for a commercial pilot)

| Layer | Pick | Licence | Commercial pilot OK? | Notes |
|---|---|---|---|---|
| Decode | FFmpeg-NVDEC subprocess (live), PyNvVideoCodec or TorchCodec (jury files) | FFmpeg (check LGPL vs GPL build), MIT, BSD-3 | Yes (watch FFmpeg build flags) | NVIDIA driver/CUDA are proprietary but free to deploy |
| Detector | **RF-DETR Small/Medium** (Apache-2.0), fine-tuned on our staged data; fallback D-FINE-S/M COCO weights (Apache-2.0) | Apache-2.0 | Yes | Do not use Atto/Femto/Pico/XL/2XL (PML), D-FINE Obj365 weights, DEIMv2, Ultralytics (AGPL) |
| Tracker + zones | `trackers` ByteTrack/OC-SORT (Apache-2.0) + `supervision` zones (MIT) | Apache-2.0, MIT | Yes | Avoid BoxMOT (AGPL) |
| Plate detection | open-image-models `yolo-v9-t-640-license-plate-end2end` (MIT-labelled) or our fine-tuned RF-DETR "plate" class | MIT | Yes with caveat: weights provenance undocumented | Replace with our own detector if the organisers ask for provenance |
| Plate OCR | fast-plate-ocr CCT-XS fine-tuned on **synthetic Uzbek plates**; regex validators from section 2.1; per-track voting over frames | MIT | Yes | Training data is ours (synthetic) |
| Bay state | ROI polygons + rules (car present, person count, lift/hood/wheel cues) + small classifier head on DINOv2-small/SigLIP2 embeddings | Apache-2.0 | Yes | |
| Labelling helpers | Grounding DINO (Apache-2.0), OWLv2 (Apache-2.0), Florence-2 (MIT), SAM 2 (Apache-2.0), Qwen3-VL-2B/4B (Apache-2.0) | permissive | Yes | Outputs reviewed by humans |
| Jury-photo explainer | Qwen3-VL-4B (Apache-2.0) or SmolVLM2-2.2B | Apache-2.0 | Yes | Low rate, on demand |
| Inference runtime | ONNX Runtime GPU (MIT), TensorRT (NVIDIA proprietary, free) | MIT / proprietary | Yes | |
| Optional scale-up | DeepStream 9.x | NVIDIA EULA | Yes (read EULA) | Only if more than ~16 cameras per box |

**Excluded for the commercial pilot [F]:** Ultralytics YOLO11/YOLO26 (AGPL-3.0, Enterprise licence needed), YOLOv9 official (GPL-3.0), YOLOv10 (AGPL-3.0), YOLO-World (GPL-3.0) as shipped code, BoxMOT (AGPL-3.0), nomeroff-net (GPL-3.0), DEIMv2 (non-commercial), Qwen2.5-VL-3B (research licence), anpr-uz weights (CC BY-NC), SAM 3 and DINOv3 (custom terms, read first), Assembly101 (CC BY-NC).
**Attribution duties:** every CC BY 4.0 item (PKLot, Roboflow sets, PhysicalAI-SmartSpaces) needs attribution in the data-source list.
**Caveat [I]:** permissive code licences do not clear the training-data licences behind pretrained weights (COCO, Objects365, LVIS). The hackathon asks for data sources and licences, so list "pretrained weights: RF-DETR (COCO-pretrained, Apache-2.0)" explicitly and say we did not review upstream image licences.

---

## 7. Data plan

### 7.1 What we can use (licence-clean, no real faces/plates)

| Source | Use | Licence | Real people/plates? |
|---|---|---|---|
| PKLot (UFPR) | Optional pretraining for empty/occupied slot classification; 12,417 images | CC BY 4.0 | Parked cars, elevated camera; plates unlikely readable [I] |
| NVIDIA PhysicalAI-SmartSpaces | Test multi-camera person tracking and zone-dwell code | CC-BY-4.0 | Synthetic only |
| Wikimedia Commons "Automobile repair shops" (166 files), "Car washes" (50 files) | Qualitative test images, per-file licence + attribution; reject any with identifiable faces/plates | per-file CC | Must screen manually |
| car-hood-open (Roboflow, 1,987 images) | Bonnet-open cue | CC BY 4.0 | Unknown; screen a sample for plates/faces before use |
| Pretrained weights: RF-DETR (Apache-2.0), fast-plate-ocr (MIT), Qwen3-VL (Apache-2.0), DINOv2/SigLIP2 (Apache-2.0), Grounding DINO/OWLv2/SAM 2 (Apache-2.0), Florence-2 (MIT) | Models only | as listed | not applicable |

### 7.2 What we must collect or generate

1. **Staged footage** (2-3 hours, several cameras if possible, in a real or mock bay): our own/borrowed cars, team members and dealer staff with **written consent** (one-page form: purpose, 90-day retention, on-site only). States: empty, car present, car on raised lift, wheel removed, bonnet open, people working, idle; plus wash, ready zone, handover. Vary light, time of day, car colour. Blur faces at ingest (edge) and keep only crops of cars/people-without-faces where possible.
2. **Own labels**: auto-label with Grounding DINO / Florence-2 / Qwen3-VL, correct by hand, ~150-300 frames per state; a quick 3-way split by time block (not random frames) to avoid leakage.
3. **Synthetic Uzbek plates**: generator per 2.6; 50-100k OCR crops, plus plates composited onto staged car frames for the detector. Invented numbers only.
4. **Jury-input robustness set**: 50-100 stills (our own phone photos + Commons) with varied angles, low light, motion blur; keep a separate test pot.
5. **Real dealer footage** only with a signed data-processing agreement from the dealer and notice to staff; keep processing on the on-site box. Ask the organisers whether footage from the dealer's existing cameras counts as "our data" and whether consented customers are allowed.

### 7.3 Forbidden or avoid

- Real faces or plates of non-consenting people: **Roboflow Uzbek_Plates, Mendeley Central Asia set, platesmania.com scrapes, VeRi-776, Ego4D, COIN/Kinetics/YouTube downloads, Open Images / COCO images with identifiable people or plates** (Open Images annotations are CC BY 4.0 and images "listed as" CC BY 2.0 with no warranty from Google [F]; plates appear in the wild) unless organisers waive the rule.
- Scraping YouTube (ToS, section 1.5) or platesmania (no stated permission, personal data) at any scale.
- Using any NC/research-only weights or datasets in the product (anpr-uz, Assembly101, VeRi, Qwen2.5-VL-3B, DEIMv2).
- Biometric processing (face recognition, gait) in the pilot: see section 8.

### 7.4 Ready-to-paste data-source list for the hackathon submission (draft)

| # | Source | Licence | URL | Real people/plates |
|---|---|---|---|---|
| 1 | Team staged bay footage (own, consented) | Proprietary / internal | - | consented only |
| 2 | Synthetic Uzbek plate generator output | Ours (CC0 if we release) | - | none |
| 3 | PKLot (optional) | CC BY 4.0 | https://web.inf.ufpr.br/vri/databases/parking-lot-database/ | parked cars, low res |
| 4 | NVIDIA PhysicalAI-SmartSpaces (optional) | CC-BY-4.0 | https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces | synthetic |
| 5 | Wikimedia Commons repair-shop images (optional) | per-file CC | https://commons.wikimedia.org/wiki/Category:Automobile_repair_shops | screened |
| 6 | car-hood-open (optional) | CC BY 4.0 | https://universe.roboflow.com/carhoodopen/car-hood-open | screen first |
| 7 | Pretrained: RF-DETR | Apache-2.0 | https://github.com/roboflow/rf-detr | - |
| 8 | Pretrained: fast-plate-ocr / open-image-models | MIT | https://github.com/ankandrew/fast-plate-ocr | - |
| 9 | Pretrained: Qwen3-VL | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | - |

---

## 8. Compliance notes for the pilot (short)

- **Personal data law:** Uzbekistan requires onshore storage of citizens' personal data collected online since April 2021 (Law on Personal Data, amended by ZRU-666); a **March 2026 amendment** eased cross-border transfer for most data but **biometric and genetic data must stay on servers in Uzbekistan**. Explicit consent is a core legal basis; employee data is covered by the Labor Code; a March 2025 Justice Ministry draft would bar use of employees' photos/video without consent and raise fines (adoption not confirmed). [S] (kun.uz, Mondaq, DataGuidance, Grata); verify on lex.uz.
- **[I] Design implications:** no face recognition, face blur at ingest, plates stored hashed or access-controlled, retention limits, on-site box (already planned), sign at the entrance, staff notice.

## 9. Not found / open questions

- Official Uzbek plate standard (size, font, letter alphabet, transit "T" format, military/police series, whether EVs use `M DDDDDD`). Ask Uzavtomotobelgi / traffic police; read Cabinet Resolutions 131/2007 and 429/2024 on lex.uz.
- Whether fast-plate-ocr's "global" model or open-image-models' detector saw Uzbek plates; training-data provenance of both. Test empirically on synthetic + consented plates.
- Consumer-RTX latency for RF-DETR, D-FINE, Qwen3-VL, SmolVLM (only T4/L4/Jetson numbers found).
- Licence of CNRPark+EXT, VeRi-Wild, AI City/CityFlow, HowTo100M, CrossTask, HOI4D.
- Whether PyNvVideoCodec accepts RTSP directly.
- Kaggle was not searched directly (no API access in this session).

---

## 10. Source list

Datasets and licences
- Roboflow Uzbek_Plates (rendered page): https://universe.roboflow.com/suxrobs-workspace/uzbek_plates
- Roboflow "Uzbek car license number OCR": https://universe.roboflow.com/uzbek-car-license-plate-number-recognition/uzbek-car-license-number-ocr
- Mendeley "Dataset Of Central Asia vehicle number plate with Annotations": https://data.mendeley.com/datasets/4c5z833m6m/2
- Roboflow car-hood-open: https://universe.roboflow.com/carhoodopen/car-hood-open
- Roboflow Carlifting: https://universe.roboflow.com/sobirin-bnpsx/carlifting-ppaom
- Roboflow Garage Car workspace: https://universe.roboflow.com/garage-xwypn
- PKLot (UFPR): https://web.inf.ufpr.br/vri/databases/parking-lot-database/ ; HF mirror https://huggingface.co/datasets/dronefreak/PKLot
- CNRPark+EXT code/README: https://github.com/fabiocarrara/deep-parking ; CNR record https://iris.cnr.it/handle/20.500.14243/409744
- ACPDS: https://arxiv.org/abs/2107.12207
- NVIDIA PhysicalAI-SmartSpaces: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
- VeRi-776: https://github.com/JDAI-CV/VeRidataset
- Ego4D licence PDF: https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf ; commercial-use commentary https://www.cloudpano.com/blog/is-ego4d-free-for-commercial-use
- Ego4D scenario grouping (secondary): https://openaccess.thecvf.com/content/ICCV2023/supplemental/Plizzari_What_Can_a_ICCV_2023_supplemental.pdf
- COIN: https://coin-dataset.github.io/
- Kinetics-700 label map (mmaction2): https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k700.txt
- Open Images facts and licence: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- Wikimedia Commons categories: https://commons.wikimedia.org/wiki/Category:Automobile_repair_shops ; https://commons.wikimedia.org/wiki/Category:Car_washes
- VinDynamics repair video datasets: https://vindynamics.net/en/data-factory/automotive-brake-rotor-replacement-video-dataset
- MIT SUN auto mechanics: https://groups.csail.mit.edu/vision/SUN/scenes/pages/a/auto_mechanics/indoor/index.html
- Assembly101: https://arxiv.org/pdf/2203.14712
- YouTube Terms: https://www.youtube.com/t/terms ; YouTube CC BY help: https://support.google.com/youtube/answer/2797468

Uzbek plates
- Wikipedia, Vehicle registration plates of Uzbekistan: https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan
- Golden Pages Uzbekistan (types, region table, font statement): https://www.goldenpages.uz/en/avto/
- gov.uz (meaning of letters on plates): https://gov.uz/en/advice/73/document/156
- Anhor, ambulance plates: https://anhor.uz/news/car-registration-number/
- Spot.uz EV plates 2021: https://www.spot.uz/ru/2021/11/15/ev-plates/ ; hybrids 2023: https://www.spot.uz/ru/2023/03/27/hybrid-plates/
- Kun.uz, digital plate system 2026: https://kun.uz/en/news/2026/03/24/uzbekistan-launches-new-digital-system-for-vehicle-registration-plates ; July 2024 special plates: https://kun.uz/en/news/2024/07/19/uzbekistan-introduces-special-license-plates-for-electric-motorcycles-mopeds-and-scooters
- lex.uz document 1249273 (Resolution 196/2007): https://lex.uz/docs/1249273
- FE-Schrift: https://en.wikipedia.org/wiki/FE-Schrift
- plate-recognition-uz: https://github.com/IzzatulloOne/plate-recognition-uz ; anpr-uz: https://huggingface.co/IzzatulloOne/anpr-uz
- fast-alpr: https://github.com/ankandrew/fast-alpr ; open-image-models: https://github.com/ankandrew/open-image-models ; fast-plate-ocr: https://github.com/ankandrew/fast-plate-ocr ; model zoo https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/
- Nomeroff-Net: https://github.com/ria-com/nomeroff-net
- justjuu RT-DETRv2 plate detector: https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection
- Korean plate generator (template): https://github.com/kstar1996/Car_Plate_DataGenerator

Models, trackers, VLMs
- RF-DETR: https://github.com/roboflow/rf-detr (README + LICENSE); docs https://rfdetr.roboflow.com/latest/exports/
- D-FINE: https://github.com/Peterande/D-FINE ; DEIM: https://github.com/Intellindust-AI-Lab/DEIM ; DEIMv2 licence: https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md
- Ultralytics licence page: https://www.ultralytics.com/license ; repo LICENSE https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX ; RT-DETR: https://github.com/lyuwenyu/RT-DETR ; YOLOv9: https://github.com/WongKinYiu/yolov9 ; YOLOv10: https://github.com/THU-MIG/yolov10
- YOLO-World: https://github.com/AILab-CVC/YOLO-World ; paper https://arxiv.org/abs/2401.17270
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO ; SAM 2: https://github.com/facebookresearch/sam2 ; SAM 3 licence: https://huggingface.co/facebook/sam3/blob/main/LICENSE
- ByteTrack: https://github.com/ifzhang/ByteTrack ; BoT-SORT: https://github.com/NirAharon/BoT-SORT ; OC-SORT: https://github.com/noahcao/OC_SORT ; trackers: https://github.com/roboflow/trackers ; supervision: https://github.com/roboflow/supervision ; BoxMOT: https://github.com/mikel-brostrom/boxmot
- FastReID: https://github.com/JDAI-CV/fast-reid ; TransReID: https://github.com/damo-cv/TransReID ; CLIP-ReID: https://github.com/Syliz517/CLIP-ReID
- Hugging Face model metadata (licence tags via API): https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct ; https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct ; https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct ; https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct ; https://huggingface.co/microsoft/Florence-2-large ; https://huggingface.co/google/owlv2-base-patch16-ensemble ; https://huggingface.co/facebook/dinov2-small ; https://huggingface.co/google/siglip2-base-patch16-224 ; https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- Qwen3-VL-2B on Jetson thread: https://forums.developer.nvidia.com/t/performance-inquiry-optimizing-qwen3-vl-2b-inference-for-2-qps-target-on-orin-nano-super/359639

Prior work
- Leverege PitCrew: https://www.leverege.com/usecases/service-intelligence-automotive
- IoT For All, service-bay intelligence: https://www.iotforall.com/bringing-ai-to-existing-camera-systems-unlocking-real-time-service-bay-intelligence
- SKAIVISION: https://skaivision.com/
- Kratsch et al., video process mining: https://arxiv.org/abs/2010.11289
- SnapLog: https://arxiv.org/abs/2604.22476 ; code https://github.com/J0nasSeng/SnapLog
- Drive-through reconstruction (dealerships): https://arxiv.org/abs/2603.26638

Decode and export
- PyNvVideoCodec API guide: https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html ; performance note https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/application-note.html ; PyPI https://pypi.org/project/PyNvVideoCodec/
- TorchCodec: https://github.com/meta-pytorch/torchcodec
- DeepStream 9.0 release notes: https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Release_notes.html ; custom model guide https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_using_custom_model.html
- NVDEC consumer-GPU session limits (news): https://tomshardware.com/news/nvidia-increases-concurrent-nvenc-sessions-on-consumer-gpus
- NVDEC application note: https://docs.nvidia.com/video-technologies/video-codec-sdk/12.2/nvdec-application-note/index.html (returned 404 to my fetch; figure comes from a search summary)

Legal context
- Uzbekistan personal data localisation: https://kun.uz/en/news/2026/03/27/uzbekistan-amends-personal-data-law-to-facilitate-global-payment-systems ; https://m.kun.uz/en/news/2021/01/26/personal-data-of-uzbek-citizens-are-now-required-to-be-stored-in-uzbekistan ; https://www.mondaq.com/privacy-protection/1522174/data-protection-and-employee-privacy-in-uzbekistan ; draft consent rules https://kun.uz/en/news/2025/03/17/justice-ministry-proposes-ban-on-filming-people-without-consent
