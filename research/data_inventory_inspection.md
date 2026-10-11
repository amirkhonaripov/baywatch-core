# Data inventory for automated vehicle inspection (dealer service entrance and car-wash lane)

Research date: 2026-10-10. Follows `stage_data.md` and `astana_tender.md` in this folder (read first; this file does not repeat their Pexels/Pixabay/Commons, Blender, plate-synthesis or Pulsar/UVeye material, it goes deeper on the inspection components and corrects a few of their numbers, see section 7).

Product context: (a) dealer service entrance, (b) car-wash lane with a drive-over "speed-bump" undercarriage camera plus side cameras, condition report on entry, weekly per-car history. We cannot film, so each inspection component must be proven on public data with a measured metric. Licences do not restrict us for this inventory (non-commercial, research-only and form-gated sets are allowed) but every licence is listed. Hackathon rule: real faces/plates of non-consenting people must not be shown, so every row says where they appear.

## How to read the tags and the columns

- **[F]** = read on the page, API, raw README/LICENSE or paper text, or seen in preview thumbnails.
- **[F thumbs]** = seen only in the 4-8 preview thumbnails of the project page (not a systematic scan). Anything I say about image type, faces or plates from thumbnails is a sample, not a guarantee.
- **[S]** = search-result summary only (lead, not verified).
- **[I]** = my inference or recommendation.
- Roboflow Universe "licence" is the uploader's label, not the image owner's. Nearly all Universe projects say "A description for this project has not been published yet", so image provenance is unknown. Several are re-uploads of other datasets (CarDD, Humans in the Loop, Shutterstock web photos) under a fresh CC BY label. Treat Universe sets as "label only, provenance unknown".
- Nothing was downloaded except small preview thumbnails. Roboflow pages return HTTP 403 to curl; I read them through a browser pane (page text, classes, image counts, licence label, and 4-8 preview thumbnails per project, about 340 small preview JPEGs in total, viewed only to judge image type, plates and faces). Roboflow returns HTTP 429 after bursts of a few dozen requests, so some searches were retried more slowly.
- Kaggle was queried through its own search endpoint (the same call the site makes), Hugging Face and Zenodo and Mendeley through their public APIs, GitHub and arXiv through their APIs. Mendeley's site search could not be driven (client-side), so Mendeley coverage relies on web search plus direct record reads; treat Mendeley as "not exhaustively searched".

---

## 0. Headlines

1. **There is no public labelled drive-over undercarriage defect set.** There are about 2,400 real stitched/composite drive-over undercarriage images on Roboflow (three Universe workspaces: 588 + 417 + 1,058 + 343 images), plus a 275-image good/bad image-quality set and 920 fisheye floor-camera frames, but their labels are only "car / under-car box", crop box, image quality, or "before/after-undercarriage"; none labels rust, leaks or damage. [F]
2. **Real underside defect labels exist only on ground-level phone photos**: the best is `undercarriage-rust-detection` (2,055 images, 3 rust classes, CC BY 4.0 label, Roboflow's own page shows mAP@50 60.8 %, precision 63.9 %, recall 60.6 %). The 668-image `Rust_detection_segmentation` uses the same image IDs (subset). Three 519-image `Underbody car defects` copies (rust, bolt, rust-bolts / steering rack) are a different set from the same family. Provenance undocumented, near-duplicate risk across siblings. [F]
3. **Best real vehicle oil-leak set found: `leak` by s Workspace (593 images, 1 class, CC BY 4.0 label)**; thumbnails show real car undersides/engine bays with oil drips. Second: `Oil and coolant leak` (49 images, with a Shutterstock-watermarked image). All the other "oil leak" sets are industrial, sea oil slicks, oil drops on paper, ultrasound images or UAV imagery. [F thumbs]
4. **A pit/lift-view parts set family exists in Indonesia-origin Roboflow workspaces**: 1,000 images of truck/bus underbody parts (differential, exhaust, leaf spring, propeller shaft, wheel, fuel tank, transmission), 1,000 suspension images, 603 images with `Kebocoran Oli` (oil leak) + shock absorber, 53 mask images (BY-NC-SA 4.0, Indonesian plate visible), 1,748 parts+Good/Bad images. Mostly heavy vehicles. [F]
5. **Several things that look like undercarriage data are not**: `Underbody Inspection` (436 images, class "Obstacles") shows rocks and minerals; `chassis` projects are VIN/chassis-number OCR; `computer-vision-for-undercarriag` (553 images) shows exterior dents on Shutterstock photos; `underbody` (6,068 images, 41 classes) and the VW set are factory-line part-variant/bracket checks; Kaggle "Underbody: Front Floor" is CAD metadata; EV-battery sets show pack internals. [F thumbs]
6. **The only genuine academic "underbody" benchmark is AutoVI (Renault, Zenodo, CC BY-NC-SA 4.0)**: `underbody_pipes` 506 and `underbody_screw` 765 images with segmentation masks, but on a factory production line with staged missing-part defects, built for unsupervised anomaly detection (AUROC/sPRO). Useful as a reference-vs-anomaly proxy, not as service-lane evidence. [F]
7. **Exterior damage is strong**: CarDD (4,000 images, 6 classes, masks, form-gated), VehiDE (13,945 images, 8 classes), Humans in the Loop (1,812 images, CC0, 21 parts + 8 damage classes), SYNDCAR (245 images with damage AND parts on the same images, CC BY 4.0), plus tens of thousands of Roboflow images. All are web/phone photos; **no fixed-camera or drive-through damage set is public**. Most big sets are copies/merges of each other, so leakage is the main risk. [F]
8. **Tyres: strong for condition classification and DOT-text OCR, thin for tread depth in mm.** One public set has gauge-measured depth in mm: `TyreCV` on Kaggle (508 stereo image pairs, 1.5-8 mm, MIT label, 14.5 GB). A Hugging Face Space reports MAE of about 0.435/32 in on 730 labelled tyres but the data is not released. [F]
9. **Car wash: thin.** A few Roboflow CCTV-style sets (237 / 2,439 / 27 images) with unclear labels, a 647-image car-dirt segmentation set, nothing for drive-over lanes, nothing on wet or dirty cars with damage labels. Several "wash" hits are hand-washing sets. [F thumbs]
10. **Change detection between two inspections of the same car: no public paired data.** Proxies are reference-based anomaly detection (AutoVI), re-identification sets for matching a car across cameras, and rendered pairs from the MIT-licensed CrashCar101 pipeline (the dataset itself is not downloadable as far as I found). [F/I]

---

## 1. Proof matrix

Access effort: **low** = free download or free account export; **form** = signed form / waiting; **sub** = paid subscription; **build** = needs rendering or compositing work. "Plates/faces" lists where real plates or people appear. Metric = what we could report with a leakage-safe split (see section 5). RF = Roboflow Universe, CC BY = CC BY 4.0 (uploader label on RF).

| Component | Dataset | Size | Image type | Licence | Plates / faces | Metric we could report | Access |
|---|---|---|---|---|---|---|---|
| 1 Undercarriage | RF `car-segmentation-k5akd/undercarriagecropping` | 588 | stitched drive-over composite, garage background [F thumbs] | CC BY (label) | none seen | box IoU / mAP50 of undercarriage crop | low |
| 1 Undercarriage | RF `copart-i960g/undercarriage-cropping` | 417 | stitched wide-strip composites, dark, edge streaks [F thumbs] | CC BY (label) | none seen | crop IoU | low |
| 1 Undercarriage | RF `car-segmentation-iq9jj/new-car-seg` + `undercarraige` | 1,058 + 343 | stitched composites (class `Under-car`, `car`) [F thumbs] | CC BY (label) | none seen | crop IoU | low |
| 1 Undercarriage | RF `car-segmentation-iq9jj/image_analysis_under` | 275 | stitched composites, Good Image / Bad Image | CC BY (label) | none seen | image-quality gate accuracy / AUROC | low |
| 1 Undercarriage | RF `car-segmentation-iq9jj/background-car` | 920 | fisheye floor/pit camera looking up at the underside; classes before-undercarriage, after-undercarriage, cars-background | CC BY (label) | people at frame edges in 2 of 6 thumbs | box mAP50 | low |
| 1 Undercarriage | RF `carp10/undercarriage-y5epg` | 193 | fixed floor camera, car passes over a mirror-like plate (class `mirror`) | none shown on page | a person and the car rear are visible | none (1 class `mirror`) | low |
| 1 Undercarriage | RF `anirudh-qjb2e/undercarriage-rust-detection-f9grj` | 2,055 | handheld/ground-level phone photos of used-car undersides | CC BY (label) | none seen in 8 thumbs | box mAP50, image-level rust AUROC; page shows mAP50 60.8 % | low |
| 1 Undercarriage | RF `anirudh-qjb2e/rust_detection_segmentation` | 668 | same images (identical image IDs) | CC BY (label) | none seen | mask AP50 / IoU for rust_normal, rust_severe | low |
| 1 Undercarriage | RF `karthik-z0q6k/underbody-car-defects` (+ `-2`, `-6`) and Anirudh forks | 519 each | ground-level phone photos, over-exposed backgrounds | CC BY (label) | none seen | mAP50 (rust, bolt, rust-bolts / steering rack) | low |
| 1 Undercarriage | RF `deteksi-objek-ciufr/bagian-bawah-kendaraan` | 1,000 | pit/lift-view phone photos, trucks and buses | CC BY (label) | none seen; hands/arms | mAP50 per part | low |
| 1 Undercarriage | RF `deteksi-objek-ciufr/system-suspension`, `sistem-suspensi-0jqsm` | 1,000 + 603 | pit-view phone photos (leaf springs, shocks) | CC BY (label) | none seen | mAP50; AP for `Kebocoran Oli` (oil leak) | low |
| 1 Undercarriage | RF `psbhatt-tekion-com/underbody-hard-parts-seg-sk6dq` | 53 | pit view; masks | **BY-NC-SA 4.0** | Indonesian plate readable in 1 of 6 thumbs | mask AP (tiny) | low |
| 1 Undercarriage | RF `psbhatt-tekion-com/car-parts-detection-eove9` | 1,748 | pit-view photos mixed with stock product photos; classes Good/Bad + 8 parts | CC BY (label) | not checked | mAP50 | low |
| 1 Undercarriage | RF `s-workspace-fz2uo/leak-4pyvb` (vehicle oil leak) | 593 | phone photos of car undersides and engine bays with oil | CC BY (label) | none seen in 8 thumbs | box AP / image-level AUROC for leak | low |
| 1 Undercarriage | RF `hamza-bashir-nzrcf/oil-and-coolant-leak-drzbm` | 49 | phone + one Shutterstock image | CC BY (label) | none seen | tiny, test only | low |
| 1 Undercarriage | RF `dataset-fod/tumpahan-oli` | 493 | ground close-ups of oil on asphalt (ASPAL / TUMPAHAN OLI) | CC BY (label) | none seen | IoU of oil on ground | low |
| 1 Undercarriage | RF `md-jubayar-alam-rafi/under-vehicle-surveillance-sys.` | 509 | handheld video frames of an underside with staged objects taped on | CC BY (label) | a person partly visible in 2 of 8 thumbs | box mAP50 (1 class) | low |
| 1 Undercarriage (factory) | RF `t-muywg/underbody` | 6,068 | factory-line flash photos, coloured paint marks, 41 part-variant classes | CC BY (label) | none seen (factory) | mAP50 per variant; off-domain for service | low |
| 1 Undercarriage (factory) | AutoVI `underbody_pipes` / `underbody_screw` (Zenodo 10459003) | 506 + 765 of 3,950 | production-line fixed cameras, staged missing-part defects, masks | **CC BY-NC-SA 4.0** | faces blurred (datasheet) | image AUROC, pixel sPRO | low (Zenodo, 0.95 + 1.35 GB zips) |
| 2 Exterior damage | CarDD | 4,000 images, 9,000+ instances | Flickr + Shutterstock web photos, high resolution | form: research/non-commercial only, no redistribution | faces/plates mosaicked or deleted (paper) | mask AP / box AP | form |
| 2 Exterior damage | VehiDE | 13,945 images, 32,000+ instances | web photos (Vietnam), about 1,395 x 1,038 px mean | none on paper; Kaggle copy labelled Apache-2.0 | unknown | AP, polygons (VIA JSON) | low (Kaggle 2.25 GB) |
| 2 Exterior damage | Humans in the Loop Car Parts and Car Damages | 1,812 (998 parts + 814 damage), 24,851 polygons | unknown web/phone photos | CC0 | `License-plate` is an annotated class, so plates are visible | mask AP; part IoU | low (Kaggle 3.1 GB) |
| 2 Exterior damage | SYNDCAR (Mendeley hzpj48krdt) | 245 | five capture devices, high resolution; called "synthetic" without a method | CC BY 4.0 | `license plate` is a parts class | damage mAP and damage-to-part assignment on the same images | low (3.2 GB zip) |
| 2 Exterior damage | RF `trainer-jaf9y/car-damage-v9lz4` | 17,607 | web/phone, CarDD-like classes plus hole, missing-part | CC BY (label) | plates visible in samples | mask AP | low |
| 2 Exterior damage | RF `changs-workspace-hnorg/vehicle-damage-gwmh4` | 10,931 | phone photos, Thai plate visible | CC BY (label) | **plate readable** | mask AP | low |
| 2 Exterior damage | RF `skillfactory/car-damage-c1f0i` | 8,857 | web photos, 31 classes | CC BY (label) | plates in samples | box AP | low |
| 2 Exterior damage | Kaggle Coco Car Damage (lplenka) | **80** (59/11/8) | web photos | CC0 | not checked | AP50 (tiny) | low |
| 2 Exterior damage | Kaggle Car Damage Assessment (hamzamanssor) | about 1,500, 224 x 224 | one car with one damage type per image (obtained from Peltarion) | CC0 | n/a at 224 px | 8-class accuracy | low |
| 2 Exterior damage | HF `tugberkkalay/autodamageiq-vehicle-damage-dataset` | 13,089 + 2,094 | merge of CarDD + VehiDE + HITL, GPT-4o labels | CC BY 4.0 label (inherits restricted sources) | inherits | do not use as a test set | low |
| 3 Tyres | `TyreCV` (Kaggle maxfinkenbeiner) | 508 stereo pairs | Raspberry Pi 5 + IMX219 stereo camera, tread depth measured with a gauge | MIT (label) | none | MAE in mm, bucket accuracy | low (14.5 GB) |
| 3 Tyres | TyreNet (Mendeley 32b5vfj6tc) | 1,698 | 48 MP phone at six service stations + two showrooms | CC BY 4.0 | low risk [I] | accuracy / F1 / AUROC defective vs good | low (1.9 GB RAR) |
| 3 Tyres | Kaggle `sameersambhare1/tyre-condition-classification-dataset` | 548 (168 / 306 / 73) | smartphone, Indian tyre workshops | CC BY-SA 4.0 | low risk [I] | 3-class accuracy | low (2.8 GB) |
| 3 Tyres | Kaggle `warcoder/tyre-quality-classification` | 1,856 files | not documented | CC BY 4.0 | not checked | accuracy | low (2.9 GB) |
| 3 Tyres | RF TyreAI `ocr-yphpv` (DOT/sidewall characters) | 4,054 | rectified sidewall crops, 39 character classes | CC BY (label) | none | exact-match DOT string, character accuracy | low |
| 3 Tyres | CAWDEC (Kaggle adamnovozmsk) | 1,500 frames + rim crops, 21 rim classes | Skoda factory conveyor, two fixed side cameras | CC BY-SA 4.0 | people may move in scene | wheel detection mAP, rim classification | low (1.6 GB) |
| 4 Car wash | RF `hoangdarkdao/wash-car-2` | 237 | fixed CCTV inside an automatic tunnel, timestamp overlay; classes 0-3 undocumented [F thumbs] | CC BY (label) | not checked | detection (class meaning unknown) | low |
| 4 Car wash | RF `um-84bee/washing-car` | 2,439 | fixed fisheye/CCTV wash-yard frames; normal car, open doors, open hood, washing car | CC BY (label) | plates readable (stage_data) | state accuracy | low |
| 4 Car wash | RF `workspace-eakuj/car-dirt-classification` | 647 | phone close-ups of mud, dust, bird droppings on paint and glass | CC BY (label) | none seen | mask IoU | low |
| 5 Parts segmentation | Ultralytics carparts-seg (from RF `car-seg`, Gianmarco Russo) | 3,833 (3,156 / 401 / 276), 23 classes | web photos, multiple views | CC BY 4.0 (doc page) | not checked | mask mAP50-95 | low (133 MB) |
| 5 Parts segmentation | RF `khpi/car-segmentation-ga8mc` | 9,011 | web photos, 15 classes | CC BY (label) | not checked | mask AP | low |
| 5 Parts segmentation | DSMLR Car-Parts-Segmentation (KMITL) | 500 | web photos, 18 part classes, COCO format | no licence file | anonymised (README) | mask AP | low (GitHub) |
| 5 Parts segmentation | Pascal-Part | 10,103 trainval + 9,637 test | PASCAL VOC 2010 Flickr photos, car parts masks | not stated | plate parts annotated | part IoU | low |
| 6 Change detection | AutoVI (reference vs anomaly proxy) | see above | fixed cameras, same station | CC BY-NC-SA 4.0 | faces blurred | AUROC, sPRO | low |
| 6 Change detection | CrashCar101 pipeline (MIT code, Blender 3.3, ShapeNet cars, Poly Haven HDRIs) | n/a (render ourselves) | synthetic paired renders [I] | code MIT; ShapeNet terms apply | none | change-mask IoU, instance recall | build |
| 6 Change detection | VeRi-776 / VehicleID / VeRi-Wild | about 50 k / 221,763 / 416,314 images [S] | surveillance cameras | not found | plates and drivers visible | re-identification rank-1 / mAP | form [S] |

---

## 2. Component 1: Undercarriage / underbody

### 2.0 What the search covered

Roboflow Universe (searches: underbody, under vehicle, undercar, undercarriage, car chassis, chassis rust, car underside, car bottom, under-car, undercarraige, car underbody scan, UVSS, under vehicle surveillance, suspicious object, oil leak family, exhaust family, suspension family, EV battery pack, brake pad/disc, plus Indonesian/Vietnamese/Turkish/Spanish/Russian/Chinese keywords, which returned nothing relevant because Universe search is semantic English-centric), plus the workspace pages of every relevant uploader. Kaggle (underbody, under vehicle, undercarriage, vehicle underside, car chassis, undercar, car bottom, oil leak, vehicle rust, exhaust, car suspension, AutoVI). Hugging Face (undercarriage, underbody, under vehicle, under-vehicle, vehicle underside, car bottom, undercar, chassis, rust, corrosion, oil leak, brake pad, wheel). Zenodo (undercarriage, underbody, under vehicle, vehicle underside, car chassis). GitHub repository search (English, Russian, Chinese terms). arXiv (about 15 query forms). Web search in English, Russian and Chinese. Result: **no dataset on Kaggle, Hugging Face, Zenodo, GitHub or arXiv is dedicated to car undercarriage defects; every real undercarriage image set is on Roboflow Universe or is the Renault AutoVI factory benchmark.** [F, negative results do not prove absence]

### 2.1 Drive-over / stitched composites and floor cameras (closest to the product)

These are the only public images that look like the speed-bump or in-ground drive-over product. None has defect labels. [F thumbs]

| Set | Link | Images | Classes / annotation | Licence | Updated | Notes |
|---|---|---|---|---|---|---|
| undercarriageCropping | https://universe.roboflow.com/car-segmentation-k5akd/undercarriagecropping | 588 | 1 class `undercarriage`, boxes | CC BY 4.0 (label) | a year ago | 3 versions; sibling `car-chips` (76 images, scratch/damage/paint, no licence shown). The page's model line says `undercarriagecropping/1 (0 images)`, so a search-snippet that "shows 0 images" is a display quirk. |
| Undercarriage-cropping | https://universe.roboflow.com/copart-i960g/undercarriage-cropping | 417 | 1 class named `0`, boxes | CC BY 4.0 (label) | a year ago | Workspace "copart" is the account name only; its other projects (label, svhn, car-detection) are off-domain. |
| new-car-seg | https://universe.roboflow.com/car-segmentation-iq9jj/new-car-seg | 1,058 | 1 class `Under-car`, boxes | CC BY 4.0 (label) | a year ago | largest composite set |
| undercarraige | https://universe.roboflow.com/car-segmentation-iq9jj/undercarraige | 343 | `car`, `under-car`, boxes | CC BY 4.0 (label) | a year ago | |
| image_analysis_under | https://universe.roboflow.com/car-segmentation-iq9jj/image_analysis_under | 275 | classification: Good Image / Bad Image (+Unlabeled) | CC BY 4.0 (label) | 3 years ago | an image-quality gate; matches the published University at Buffalo / ACV Auctions idea of classifying defective capture images [S] |
| background-car | https://universe.roboflow.com/car-segmentation-iq9jj/background-car | 920 | `before-undercarriage`, `after-undercarriage`, `cars-background`, boxes | CC BY 4.0 (label) | 3 years ago | fisheye frames from a floor/pit camera looking up; meaning of before/after is undocumented |
| top-position | https://universe.roboflow.com/car-segmentation-iq9jj/top-position | 108 | `car-top-view`, boxes | CC BY 4.0 (label) | 2 years ago | fixed overhead fisheye of a dark bay, a car entering (useful as a fixed-camera bay image) |
| Undercarriage (carp10) | https://universe.roboflow.com/carp10/undercarriage-y5epg | 193 | 1 class `mirror`, boxes | none shown | 2 years ago | frames show a flat mirror-like unit at ground level, a car passing over it, a person walking; closest to a "speed-bump" camera view but labels are about the mirror |
| UnderCar | https://universe.roboflow.com/under/undercar-xaflv | 14 (version 1 shown as 27 in a search snippet [S]) | 10 classes `BODY, AS, B, C, Chassis, CV, G, K, P, S` | CC BY 4.0 (label) | 2 years ago | wide-strip, sunlit ground-camera frames; abbreviations undocumented. Siblings: `under/test2-toki0` 234 images, semantic segmentation, 11 classes (`BODY, AS, B, C, CV, G, K, P, PER, pnp, S`); `under/kolong-mobil` 17 images (`ban-chassis-coverChas, chassis, cover`). |

Data-quality concerns: four different uploader accounts, probably one capture system or marketplace feed [I from identical image style]; strong edge streaks and rolling-shutter-like smearing in the composites; labels are for cropping, not for inspection; plate-free by design but background people exist in the fisheye and floor-camera frames. Per-project licence labels are uploader-set.

### 2.2 Ground-level phone photos with rust/defect labels

| Set | Link | Images | Classes / annotation | Licence | Updated | Notes |
|---|---|---|---|---|---|---|
| undercarriage-rust-detection | https://universe.roboflow.com/anirudh-qjb2e/undercarriage-rust-detection-f9grj | **2,055** | 3 classes `rust`, `rust__normal` (sic), `rust_severe`; boxes | CC BY 4.0 (label) | a month ago | 70 dataset versions, 62 trained models; page-reported best model `...-70-rfdetr-nano-t2`: mAP@50 60.8 %, precision 63.9 %, recall 60.6 % (own split, not leakage-checked). The earlier "about 1.2k images" is an older snapshot. |
| Rust_detection_segmentation | https://universe.roboflow.com/anirudh-qjb2e/rust_detection_segmentation | 668 | `rust_normal`, `rust_severe`; instance segmentation | CC BY 4.0 (label) | 6 months ago | first thumbnails carry the same image IDs as the 2,055 set, so it is a subset (not independent) [F thumbs] |
| Underbody car defects (original) | https://universe.roboflow.com/karthik-z0q6k/underbody-car-defects | 519 | `rust`, `bolt`, `rust-bolts`; boxes | CC BY 4.0 (label) | 2 years ago | variants `-2` (`rust`, `bolt`, `steering rack`, 519 images) and `-6` (`rust-bolts`, 519) by the same user; Anirudh holds forks `underbody-car-defects-2` (two copies) and `-6` (519 each) and `Underbody` (45) |
| Underbody (Lefa) | https://universe.roboflow.com/lefa/underbody-heyzf | 45 | `Defect`, `Bolts-196x`; boxes | CC BY 4.0 (label) | 2 years ago | thumbnails show factory/lift views with a yellow gantry, not service photos |
| underbody screw annotation | https://universe.roboflow.com/zaid-suimd/underbody-screw-annotation | 51 | `screw`, `underbody screw` | CC BY 4.0 (label) | a year ago | Kaggle copy `zaidmahmud/underbody-screw-annotated-dataset`: 104 files, MIT label. Images not viewed. |
| computer-vision-for-undercarriag | https://universe.roboflow.com/suhass-workspace-y4ouv/computer-vision-for-undercarriag | 553 | `dents`; instance segmentation | CC BY 4.0 (label) | 6 months ago | **mislabelled:** thumbnails are exterior damaged cars with Shutterstock watermarks and a person's name caption. GitHub `suhasrv/computer-vision-for-undercarriag-main-project-and-dataset` (no licence; code only) trains YOLOv8m-seg on its 1,617-image export (`corrosion`, `dents`) and states mAP50 50.5 %, mAP50-95 24.4 %. |

Image type of the 2,055/668/519 sets: looks like handheld low-angle phone photos taken from the side at ground level (wheels visible), outdoor or garage, used-car style. Not a drive-over. [F thumbs] Concerns: unknown provenance (probably marketplace/auction listing photos), near-duplicates across the sibling projects (several share one image style), class `rust_normal` vs `rust_severe` is subjective, no vehicle ID so a car can appear in train and test, bolts/steering rack labels are parts not defects. Dedupe by perceptual hash before any split.

### 2.3 Pit/lift-view parts and leak labels (Indonesia-origin, mostly trucks and buses)

| Set | Link | Images | Classes / annotation | Licence | Updated |
|---|---|---|---|---|---|
| bagian-bawah-kendaraan ("vehicle underside") | https://universe.roboflow.com/deteksi-objek-ciufr/bagian-bawah-kendaraan | 1,000 | 7 classes: `Gardan` (differential), `Knalpot` (exhaust), `Leaf Spring`, `Propeller Shaft`, `Roda` (wheel), `Tangki Bahan Bakar` (fuel tank), `Transmisi`; boxes | CC BY 4.0 (label) | 3 years ago |
| system-suspension | https://universe.roboflow.com/deteksi-objek-ciufr/system-suspension | 1,000 | `Ball Joint`, `Coil Spring`, `Leaf Spring`, `Shock Absorber`, `Stopper`; boxes | CC BY 4.0 (label) | 2 months ago |
| Sistem Suspensi | https://universe.roboflow.com/deteksi-objek-ciufr/sistem-suspensi-0jqsm | 603 | `Kebocoran Oli` (oil leak), `Shock Absorber`; boxes; 6 models | CC BY 4.0 (label) | 3 years ago |
| projek-suspensi | https://universe.roboflow.com/deteksi-objek-ciufr/projek-suspensi | 1 | `Coil Spring`, `Kebocoran Minyak Rem` (brake-fluid leak), `Roda` | CC BY 4.0 (label) | 2 years ago |
| underbody-hard-parts-seg | https://universe.roboflow.com/psbhatt-tekion-com/underbody-hard-parts-seg-sk6dq | 53 | 13 classes: `tire`, `Tire`, `brake caliper`, `Brake Caliper`, `brake disc`, `Brake Drum`, `Brake Pads`, `Brake Rotor`, `coil spring`, `exhaust tail pipe`, `stabilizer bar`, `Strut Assembly`, `Wheel Cylinder`; instance masks | **BY-NC-SA 4.0** | 2 months ago |
| car-parts-detection | https://universe.roboflow.com/psbhatt-tekion-com/car-parts-detection-eove9 | 1,748 | `Good`, `Bad`, `Ball Joint`, `BrakePad`, `Coil Spring`, `Leaf Spring`, `MountPad`, `Shock Absorber`, `SidePad`, `Stopper`; boxes | CC BY 4.0 (label) | 2 months ago |

Image type: phone photos taken inside an inspection pit or under a lift, looking up at trucks, buses and pickups (red/yellow pit rails visible), some with a hand gauge or hands in view. [F thumbs] The Tekion sets (the workspace name suggests the dealer-software company Tekion [I]; purpose unstated) include a readable Indonesian plate in 1 of 6 thumbnails and stock product photos (a brake pad); their class names equal those of the Deteksi-Objek `system-suspension` set, so they look derived from the same photos [I]. Concerns: heavy-vehicle bias (leaf springs, propeller shafts) so little transfer to passenger-car underbodies; Indonesian/English class aliases and duplicate-case class names (`tire` vs `Tire`); label names mix parts and defects; the three Deteksi-Objek projects share one image source, so do not split across them.

### 2.4 Factory-line underbody sets (off-domain but real, measurable)

| Set | Link | Images | Classes | Licence | Notes |
|---|---|---|---|---|---|
| underbody (T) | https://universe.roboflow.com/t-muywg/underbody | **6,068** | 41 classes of part variants, e.g. `BRAKE FLEXIBLE HOSE FR-LH:ORANGE`, `EG MOVING CONTROL ROD-PINK`, `ENGINE UNDER COVER-WITH`, `EXHAUST FR PIPE-F1`..`F6`, `-T1`..`T5`, `FR BRAKE CALIPER-LH:55` / `:K`, `FR DRIVE SHAFT-LH:GREEN/ORANGE/PURPLE/WHITE`, `FR LOWER ARM-LH:H XL/XL`, `FR SHOCK ABSORBER-LH:DOT BLUE/DOT YELLOW/GREEN/WHITE`, `REAR OXYGEN SENSOR-RED GREEN`, `RR E/G MOUNTING (TORQUE ROD)-BLUE/GREEN`; boxes | CC BY 4.0 (label) | updated 3 years ago, 2 models. Flash-lit assembly-line photos, coloured paint marks identify variants (a "which variant is fitted" check), so classes are not damage. A Chinese blog repo (`share2code99/chassis_component_yolo11_detection`) describes this exact set (v12, 2024-01-31, 6,068 images, 41 classes, via the "qunshankj" platform). |
| Underbody Inspection System (VW) | https://universe.roboflow.com/vw-nfqmp/underbody-inspection-system | 1,185 | `crooked bracket`, `no washer friction`, `straight bracket`, `washer friction` | CC BY 4.0 (label) | close-ups of brackets and washers on a production line |
| AutoVI | https://zenodo.org/records/10459003 ; https://autovi.utc.fr | 3,950 in v1.0.0 (1,530 train, 2,420 test); `underbody_pipes` **506**, `underbody_screw` **765**, plus engine_wiring 892, pipe_clip 532, pipe_staple 524, tank_screw 731 | missing/misplaced-part defects, **created on purpose by Renault**, segmentation masks per defect; train folder is defect-free | **CC BY-NC-SA 4.0** (Zenodo and datasheet); commercial use needs a request | zip sizes: underbody_pipes 950 MB, underbody_screw 1,347 MB. Datasheet: faces of operators blurred. The project website now states 11 classes and 7,184 images (a newer release); Zenodo v1.0.0 says 3,950. Kaggle mirrors with different licence labels exist (`blank1508/autovi-dataset-version-1-0-0` CC BY-NC-SA 4.0; others "mit"/"apache-2.0"/"CC BY-SA 4.0" labels that do not override the original). |

### 2.5 Security-style under-vehicle (UVSS) sets

| Set | Link | Images | Classes | Licence | Notes |
|---|---|---|---|---|---|
| Under Vehicle Surveillance Sys. | https://universe.roboflow.com/md-jubayar-alam-rafi/under-vehicle-surveillance-sys. | 509 | `suspicious-object`; boxes | CC BY 4.0 (label) | handheld video frames with a white box or paper taped under the car (staged), a bystander partly visible in 2 of 8 thumbnails. Student-project style. [F thumbs] |
| V-Sentry (GitHub) | https://github.com/fengfenghencai/V-Sentry | 3 sample images | n/a | none | code only; YOLOv8n with colour/shape rules for trackers and explosives |
| other GitHub "underbody inspection" repos | `pavinvelan/Automatefd-Vehicle-Underbody-Monitoring-System` (Apache-2.0 app), `hrishikesh1427/Vehicle-Underbody-Inspection` (React UI), `VinishP-03/...` (empty) | none | n/a | | application code only, no dataset |

Commercial UVSS (ISS SecurOS, Dahua, Hikvision, Vehant, Rapiscan) publish no datasets; web search found only datasheets. [S]

### 2.6 Oil / fluid leak images (any context)

| Set | Link | Images | What it actually shows | Licence | Verdict |
|---|---|---|---|---|---|
| leak (s Workspace) | https://universe.roboflow.com/s-workspace-fz2uo/leak-4pyvb | 593 | **real car undersides / engine bays with oil drips and ground stains** [F thumbs]; 1 class `leak`; boxes | CC BY 4.0 (label) | best vehicle leak set; updated 4 months ago |
| Oil and coolant leak | https://universe.roboflow.com/hamza-bashir-nzrcf/oil-and-coolant-leak-drzbm | 49 | car undersides, one Shutterstock image; instance masks | CC BY 4.0 (label) | tiny, usable only as a cross-check |
| Sistem Suspensi (`Kebocoran Oli`) | see 2.3 | 603 | truck pit-view photos | CC BY 4.0 (label) | independent test domain for a leak model trained on `leak` |
| tumpahan-oli | https://universe.roboflow.com/dataset-fod/tumpahan-oli | 493 | oil spill on asphalt, close-up phone photos (`ASPAL`, `TUMPAHAN OLI`) | CC BY 4.0 (label) | the "oil on the ground" case; no car in frame |
| oil-leak (Tryout) / oil-leak-detection (Chiranjeevi) / oil-leak-detection (oil-detection) | `tryout/oil-leak` 68 (MIT label), `chiranjeevi-p2jmx/oil-leak-detection-axcq4` 311, `oil-detection/oil-leak-detection` 26 | | industrial plants, engines, oil-field scenes | CC BY / MIT | off-domain |
| oil-leak-discovery | `custom-yolov5-1lqs1/oil-leak-discovery` | 118 | oil slicks on the sea | CC BY | off-domain |
| engine-oil-detection / engine-oil-analyse | `newproject-7c1cc/...` | 185 / 327 | oil drops on paper (blotter test) | CC BY | off-domain |
| fluid | `dhip-tupart/fluid-j85l3` | 412 | **ultrasound images** | BY-NC-SA 4.0 | off-domain |
| Kaggle `dimpleb/image-dataset-for-surface-oil-and-water-leakage` | https://www.kaggle.com/datasets/dimpleb/image-dataset-for-surface-oil-and-water-leakage | 1,544 files | UAV multispectral imagery (IIT Bombay) | CC BY-SA 4.0 | off-domain |

Published results on proprietary data: ACV Auctions oil-defect models on undercarriage/dipstick/cap photos reported preliminary AUROC 0.84 for oil condition and 0.73 for leaks [S, TechConnect 2023 summary]. University at Buffalo / ACV: 1,100 undercarriage images labelled with 8 image-defect tags, ResNet-34 average ROC area 0.90 [S, TechConnect 2022]. Neither dataset is released.

### 2.7 Exhaust, EV battery pack, suspension

- **Exhaust/muffler damage: none.** Universe "exhaust" hits are motorcycle exhaust legality (`knalpot standar / tidak standar`, 98 images), factory pipe checks (`exhaust-pipe`, 1,869 images, class `pipe_ng`), or plain `exhaust` boxes (400 images). `Knalpot` as a part class exists in the Indonesian underside set (2.3) and exhaust-pipe variants in the factory `underbody` set (2.4). [F]
- **EV battery-pack underbody damage: none.** Universe EV-battery sets show pack internals (modules, busbars, bolts, casings: 145, 198, about 1,000, 643, 1,010 and 94 images); no enclosure-impact images. A 2026 SAE paper and patents discuss underbody impact sensors, not image sets. [F/S]
- **Suspension parts: yes, in 2.3** (leaf spring, coil spring, ball joint, shock absorber, stopper, strut assembly, stabilizer bar). Brake wear: `brake-shoe/brake-pad-wear-znaqd` (408 images, classification 0/25/50/75/100 % worn + irrelevant, CC BY 4.0 label, web-style product/hand photos). [F thumbs]

### 2.8 Non-image and proprietary work worth knowing

- arXiv 2601.14208 (ACV Auctions + University at Buffalo): three-camera drive-over rig, videos about 8 s, 3D Gaussian-splatting undercarriage models, tested on 10 vehicles; no dataset or code release stated. [F]
- arXiv 2509.01719: underbody impacts detected from windshield-mounted IMU + microphone (ROC-AUC 92 %), a non-image alternative; data release not stated. [F abstract]
- Pulsar.ai SmartRail, UVeye Helios, Ravin: see `astana_tender.md`; no public data.

### 2.9 Off-domain / mislabelled hits to avoid

| Hit | Why |
|---|---|
| RF `srikarthicks-workspace/underbody-inspection` (436 images, class `Obstacles`) | all 8 thumbnails are rocks and mineral specimens |
| RF `odometer-zig37/chassis-t8cl4` (3,696), `chassis-2-3c9q7` (3,394), `saad-qylvv/numero-chassis-mpgbr` (1,127), `abc-sp3v2/chassis-number` (475) | chassis/VIN number OCR |
| RF `rep2/under` (2,575; scratch, dent, broke, decay, NoDamageCar) | exterior damage on Thai vehicles, plates readable |
| RF `suhass-workspace-y4ouv/computer-vision-for-undercarriag` | exterior dents (see 2.2) |
| Kaggle `viniciusbonicio/underbody-front-floor` | one Ford CAD metadata text file |
| Kaggle `rainyseason72/underbody-pipe-autovi`, `underbody-screw-autovi` | mirrors of AutoVI |
| HF `structmeshdata/underbody-impact-data` (MIT) | 3D/time-series simulation data |
| HF `ybli/yolo-car-*-flaw-detection` | download-stub READMEs pointing to data2.cn, factory parts |

### 2.10 Verdict for component 1: **thin**

Real, usable: rust/corrosion detection on ground-level underside photos (about 2,000 images, weak provenance), vehicle oil-leak detection (593 images plus a 603-image truck pit-view cross-domain test), parts localisation under the vehicle (about 2,600 pit-view images, trucks), crop/localisation and quality-gating on drive-over composites (about 2,400 + 275 images). Not available: any defect label on drive-over composites, passenger-car lift-view damage, exhaust damage, EV battery enclosure damage, weekly/paired scans. Claim only "measured on public ground-level photos and pit-view photos, not on drive-over scans".

---

## 3. Component 2: Exterior damage

| Dataset | Link | Size | Classes | Annotation | Licence | Image type | Plates / faces | Notes |
|---|---|---|---|---|---|---|---|---|
| **CarDD** | https://cardd-ustc.github.io/ ; paper https://arxiv.org/abs/2211.00945 | 4,000 images, 9,000+ instances; average 684,231 pixels (13.6x prior sets) | `dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat` | boxes + masks + salient-object maps | access form; research and educational non-commercial only; copyright stays with Flickr/Shutterstock; no redistribution (paper + licence PDF read in `stage_data.md`) | web photos from Flickr and Shutterstock, wide variety of viewpoints and light | **faces and plates mosaicked or the image deleted** (paper) | over 10,000 candidates were collected; rusty, burned and smashed cars were attached **without annotations**. Third-party copies: Kaggle `issamjebnouni/cardd` (4,001 files, 3 GB, licence "Unknown"), `gabrielfcarvalho/cardd-with-yolo-annotations-images-labels` (CC BY-NC 4.0), Roboflow `car-damages-djnvi/car-damage-l7wgd` (4,000 images, CC BY label). The paper tests Mask R-CNN, Cascade Mask R-CNN, GCNet, HTC, DCN and its DCN+; the extracted table shows overall mask AP roughly 48-57 and small-object AP only about 14-35 [F, read from PDF text, row alignment not verified]. |
| **VehiDE** | paper https://doaj.org/article/6e0eb600059440b390431a42dfe8d138 (J. Information and Telecommunication 2024); Kaggle https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection | 13,945 images, 32,000+ instances | 8 damage categories per paper; a third-party processed copy shows Vietnamese source classes `mat_bo_phan` (missing part), `rach` (tear), `mop_lom` (dent), `tray_son` (paint scratch), `vo_kinh` (broken glass), `thung` (puncture), `be_den` (broken lamp) and mean image size about 1,395 x 1,038 | VIA JSON polygons (Kaggle), tasks: classification, detection, instance segmentation, salient objects | no licence in the paper; Kaggle copy labelled Apache-2.0 (uploader); HF copies carry no card | high-resolution web photos of damaged cars, Vietnam | unknown, plates likely [I] | 2.25 GB on Kaggle. The 8th category name was not found. Derived Kaggle `m4rcuseryx/vehide-dataset-preprocessed` (27,283 files). |
| **Humans in the Loop** Car Parts and Car Damages | https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/ ; https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages | 1,812 images (998 parts, 814 damage); 24,851 polygons | parts (21): Windshield, Back-windshield, Front-window, Back-window, Front-door, Back-door, Front-wheel, Back-wheel, Front-bumper, Back-bumper, Headlight, Tail-light, Hood, Trunk, License-plate, Mirror, Roof, Grille, Rocker-panel, Quarter-panel, Fender. Damage (8): Dent, Cracked, Scratch, Flaking, Broken part, Paint chip, Missing part, Corrosion | polygons; each image has parts **or** damage, not both | CC0 1.0 | segmented by Beetroot Academy trainees (Ukraine pilot); image origin not described | `License-plate` is a class, so real plates are present [F class list] | Kaggle 3.1 GB (5,436 files + 1,813 JSON). Many derivatives: HF `DrBimmer/car-parts-and-damage-dataset` (MIT label) and about ten copies, Roboflow `raja-nidft/car-damage-hyplq` (814), `urfu-kcxlq/damage-car-lh7s3` (2,280), `car-segmentation-iq9jj/car-damage-haa6l` (817), `car-parts-9vig8` (998), `shinhatest/car-parts-segmentation-0gniu` (2,796; thumbnails contain mask montages beside the photos, a data-quality flaw), IEEE DataPort `Pixemantic Vehicle Damage-29` (5,244 images = 21 parts + 8 damage; subscription, no licence stated). |
| **SYNDCAR** | https://data.mendeley.com/datasets/hzpj48krdt | 245 images, 3.2 GB zip | damage (4): broken glass, broken lights, cracks, scratches; parts (28): windows, doors, quarter panels, wheels, windshields, mirrors, `license plate`, lights, panels, roof, rocker panels | YOLO boxes (damage) + polygons (parts) **on the same images**; prepare script for COCO | CC BY 4.0 | titled "synthetic" but described as captured with five imaging devices; method not explained (staged or simulated damage?) | `license plate` is a parts class | the only public set where damage and part are labelled together; Digital Transformation CoLAB / University of Minho, 2025-10-31 |
| CarDamagePanel (CMU course project) | https://huggingface.co/datasets/kadireks/cardamagepanel-originals | 565 (415 / 103 / 47) | 21 exterior panels incl. `license-plate` (250 instances) | polygons, YOLO-seg | `other` | photos of damaged cars | plates labelled | small, recent (2026-10-09) |
| Kaggle Coco Car Damage Detection | https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset | **80 images** (59 / 11 / 8) | one `damage` category, and 5 part-of-damage categories `headlamp, front_bumper, hood, door, rear_bumper` | COCO | CC0 | web photos | not checked | the earlier note of "152 files" counts JSON and thumbnails; only 80 images |
| Kaggle Car Damage Assessment | https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment | about 1,500, 224 x 224 | broken headlamp, broken tail lamp, glass shatter, door scratch, door dent, bumper dent, bumper scratch, unknown | classification | CC0 (data from Peltarion) | cropped web photos | not visible at 224 px | |
| Kaggle Comprehensive Car Damage Detection | https://www.kaggle.com/datasets/samwash94/comprehensive-car-damage-detection | 2,300 (F_Breakage 500, F_Crushed 400, F_Normal 500, R_Breakage 300, R_Crushed 300, R_Normal 300) | 6 classes | classification | CC0 | front/rear views | not checked | |
| Kaggle Car Damage Severity | https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset | 1,631 files | minor / moderate / severe | classification | CC BY-NC-SA 4.0 | scraped "from multiple sources" | not checked | |
| Kaggle `anujms/car-damage-detection` | https://www.kaggle.com/datasets/anujms/car-damage-detection | 2,299 files | training/validation folders, no description | folders (classification) | Unknown | not described | not checked | 77 k views, licence unknown |
| Roboflow large sets (all CC BY label, plates in samples) | `trainer-jaf9y/car-damage-v9lz4` 17,607 (8 classes: crack, scratch, dent, hole, glass-shatter, lamp-broken, missing-part, tire-flat; seg); `changs-workspace-hnorg/vehicle-damage-gwmh4` 10,931 (scratch, damage, broken, dents; seg; Thai plate readable); `ayhan-gul-hgudf/car-damage-rlogo` 9,900 (1 class; seg); `skillfactory/car-damage-c1f0i` 8,857 (31 classes; boxes); `truck-damage/car-damage-dataset-iqcfm` 7,133 (7 numeric classes); `accident-k7ls5/vehicle-damage-dataset` 6,668 (CarDD classes + damage; seg); `dt-xjexo/car-damage-uffwb` 4,972 (13 classes incl. Rust, Paint chip-peel, Normal Glass/Tyre); `car-damages-djnvi` 4,000 (CarDD classes) | | | | | web and phone photos, some with date stamps | plates visible [F thumbs] | classes overlap heavily with CarDD/VehiDE/HITL, so these are probably re-uploads or merges [I] |
| HF `tugberkkalay/autodamageiq-vehicle-damage-dataset` | https://huggingface.co/datasets/tugberkkalay/autodamageiq-vehicle-damage-dataset | 13,089 train + 2,094 val; about 42,419 labels | crack 3,427, dent 11,204, glass_shatter 2,696, lamp_broken 776, scratch 23,967, tire_flat 349 | YOLO boxes; labels by GPT-4o on CarDD + VehiDE + HITL | cc-by-4.0 (label; does not override CarDD terms) | | inherits | convenient but contaminated by construction |
| HF `Naiscorp/car-damage-dataset` | https://huggingface.co/datasets/Naiscorp/car-damage-dataset | 22,491 unlabeled images (3.8 GB) | none | raw images | `other` | insurance-style photos | unknown | unlabelled source material |
| IEEE DataPort Pixemantic Vehicle Damage-29 | https://ieee-dataport.org/documents/pixemantic-vehicle-damage-29 | 5,244 | the 29 HITL class names | YOLO/COCO masks | none stated; subscription | | | derivative |
| CrashCar101 | https://crashcar.compute.dtu.dk/ ; code https://github.com/JensPars/CrashCar_procedural_generation | 101,050 synthetic images per search snippet [S] | part + damage categories (not enumerated) | pixel-accurate masks | code MIT | Blender 3.3 render of damaged 3D cars; needs ShapeNet cars + Poly Haven HDRIs | none | dataset file not found; regenerate [I] |
| Other Kaggle damage sets (provenance unchecked) | `moeeznisar/car-dent-and-parts-detection-using-yolov8` (6,006 files, licence Unknown), `albertjoseph2002/car-damage-detection-on-13-different-car-parts` (17,698 files, MIT label), `aryaniggeh/damage-car-images-for-vision-training` (75 files, CC BY-NC-ND 4.0), `nasimetemadi/car-damage-detection` (16,004 files, "Other") | listed for completeness | | | | | | not opened |

Faces/plates: CarDD removes them; HITL, SYNDCAR, CarDamagePanel label plates as a class; most Roboflow sets show plates. Plan to run a plate detector and replace/blur (see `stage_data.md` plate pipeline) before any image is shown.

**Fixed camera vs phone:** every public damage set above is web or phone photography. No fixed-camera, drive-through, car-wash-entrance or dealer-gate damage set is public. SYNDCAR (five devices) and CAWDEC (factory conveyor cameras, wheels only) are the nearest. [F]

**Masks and part labels:** CarDD (boxes + masks, damage only); VehiDE (polygons, damage only); HITL (polygons, parts or damage per image); SYNDCAR (damage boxes + part polygons on the same 245 images); CarDamagePanel (panels only); Roboflow `shinhatest`/`car-parts-9vig8` (parts only).

**Verdict: strong.** Enough diverse labelled data to train and to report cross-source numbers. Caveats: duplicates across mirrors (use one source per split), CarDD terms, web-photo domain gap to a fixed lane camera, no wet/dirty condition labels, no severity-in-mm.

---

## 4. Components 3, 4, 5, 6 in detail

### 4.1 Component 3: Tyres

**Tread depth with measured ground truth (mm)**

| Set | Link | Size | Image type | Labels | Licence | Notes |
|---|---|---|---|---|---|---|
| **TyreCV - Two Views Are All You Need?** | https://www.kaggle.com/datasets/maxfinkenbeiner/tyrecv-two-views-are-all-you-need | **508 stereo pairs** (6,938 files, 14.49 GB; raw cropped + augmented) | Raspberry Pi 5 16 GB with a Waveshare IMX219-83 stereo camera (8 MP) | tread depth measured manually with a gauge, range 1.5 mm to 8 mm, bucketed `<=3`, 4, 5, 6, 7, 8 mm; left and right labels; stereo calibration file | MIT (uploader) | the only public set with gauge mm; close-range lab-style capture, few low/high depths (hence buckets); no paper found. Updated about a year ago. [F] |
| HF Space `UnCodeSpecial/tire-tread-depth` | https://huggingface.co/spaces/UnCodeSpecial/tire-tread-depth | 730 labelled tyre photos (not released) | close-up straight-on tread photos | depth 2/32 in to 10/32 in (about 1.6-7.9 mm) | CC BY-NC 4.0 (app/model) | reports leakage-free 5-fold MAE of about 0.435/32 in (about 0.35 mm), 70 % exact, 95 % within one step (earlier 89.6 % exact was leaked). Use as an external benchmark number only. [F README] |
| RF `automatics-number-plate-recognition/tyre-thread-depth` | https://universe.roboflow.com/automatics-number-plate-recognition/tyre-thread-depth | 36 | tread with a **green laser line** and depth numbers drawn on frames [F thumbs] | `Damage, Anamoly, curly_thread, Thread_depth, Thread_depth1, Thread_depth2, tyre_design, Tyres` | CC BY (label) | tiny |
| RF `group5-odr67/tread-dbn2s` | https://universe.roboflow.com/group5-odr67/tread-dbn2s | 369, classification | frames of a tread with a hand-held gauge-like object [F thumbs] | `tread_0.0`, `tread_0.9`, `tread_1.8` (meaning undocumented) | CC BY (label) | Kaggle `numberfive/tire-tread-photos` also has 369 images (Good 234 / Bad 135, licence Unknown); probably the same photos, near-duplicate video frames [I] |
| QBurst case study, Goodyear/Taazaa smartphone POC, patents | see source list | n/a | | | proprietary | QBurst claims +-1.5 mm on 90 % of tests (50,000 images, unreleased) [S]; Ajou University record builds its own groove-depth video set because none exists [S] |

**Condition (good / worn / defective), cracks, flats**

| Set | Link | Size | Classes | Licence | Notes |
|---|---|---|---|---|---|
| TyreNet (Mendeley 32b5vfj6tc) | https://data.mendeley.com/datasets/32b5vfj6tc | 1,698 (866 defective + 832 good); 1.9 GB RAR | defective (cracks, wear, others) vs good | CC BY 4.0 | 48 MP Android phone, six car/bike service stations + two showrooms; annotated by experts; published 2024-01-16. [F] |
| Mendeley `dfzbm2ktss` Annotated Tyre Image Dataset (life prediction) | https://data.mendeley.com/datasets/dfzbm2ktss | image count not stated; 889 MB zip | defective/good plus usage metadata for residual-life | CC BY 4.0 | four service stations in Meerut; v2 2025-12-11. [F] |
| Kaggle Tyre Condition Classification (sameersambhare1) | https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset | 548 (NEW 168, SERVICEABLE 306, UNUSABLE 73) | 3 usability classes | CC BY-SA 4.0 | smartphone photos from Indian tyre-repair workshops. [F] |
| Kaggle Tyre Quality Classification (warcoder) | https://www.kaggle.com/datasets/warcoder/tyre-quality-classification | 1,856 files, 2.86 GB | good / defective | CC BY 4.0 | 4,362 downloads; sources undocumented; `rpjinu/tyre-faultfindy` (1,857 files, same 2.86 GB, MIT label) looks like a copy [I] |
| Kaggle Tire Texture Image Recognition | https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition | 1,028 files, 742 MB | normal / cracked textures [S, from the dataset title] | CC0 | 3,575 downloads; classes not opened |
| Kaggle Full vs Flat Tire Images | https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images | 900 files | full / flat | CC BY-NC-SA 4.0 | Roboflow `yolov8-daldb/full-vs-flat-tire` (737) and `national-formosa-university-9ftdd/tire-flat-normal` (759) are similar |
| Kaggle aggregator `prakarshawasthi/tyre-dataset-detection-severity-classification-rul` | https://www.kaggle.com/datasets/prakarshawasthi/tyre-dataset-detection-severity-classification-rul | 100,752 images, 33.5 GB | detection, health (Healthy/Moderate/Poor), severity, damage (cracks, bulges, cuts, tread wear, sidewall), RUL table | MIT (uploader) | collects Roboflow exports (e.g. `Tire_Tread.v1i.yolov8`) from IIT Dhanbad; provenance and licence unreliable |
| Kaggle `nethhari/sri-lanka-bus-tire-condition` | https://www.kaggle.com/datasets/nethhari/sri-lanka-bus-tire-condition | 1,800 images + CSV, 4 GB | condition | Unknown | bus tyres |
| RF tire-tread sets | `mark-aft7n/tire-tread` 6,316 (BAD_Tyres, BALD_Tyres, NORMAL_Tyres, public-domain label, mosaic/augmented crops); `eva-chen-ilht2/tire-tread-segmentation-orjof` about 6.14k (same classes, segmentation); `asmaakhaled/car-tire-zwml5` 2,459 (Good, Defective, used; classification); `tyre-tj9mt/tyre-q73y8` 167 (Hairline Crack, Puncture, Sidewall Crack, Small Cracks, Defect); `muhammad-anif-zuhrul-anam-eaufq/crack-tyre-detection` 874 (class `crack`, mosaic tiles); `car-parts-straight-photos/tire-flat` 685 (`tire-crack`, `tire-flat`; street photos of flat tyres on cars, plates in background) | | | CC BY label unless noted | qualitative labels, not mm; heavy augmentation in some (mosaics) |
| HF `NMiriams/Good_Tires`, `Defective_Tires` | https://huggingface.co/datasets/NMiriams/Defective_Tires | size not checked (Defective_Tires is tagged 1K-10K); 78 and 69 downloads | good / defective | cc-by-4.0 | small, not opened |

**DOT code / sidewall text:** RF TyreAI `ocr-yphpv` 4,054 images, 39 character classes (0-9, A-Z, hyphen, slash, Boundary), CC BY label, rectified unwrapped sidewall strips plus natural DOT close-ups [F thumbs]; other sets (83 / 97 / 294 / 180 / 129 / 494 images) are in `stage_data.md`. Tyre-brand recognition: `car-segmentation-iq9jj/cartire_classification` 425 images, 43 brand classes; `cartire-oxtxu` 160 images, 15 brand classes (Bridgestone, Continental, Falken, Firestone, Goodyear, Hankook, Kenda, Landgolden, Michelin, Pirelli, ...). Michelin DOT format notes are in `stage_data.md`.

**Wheel/rim/brake:** CAWDEC (https://www.kaggle.com/datasets/adamnovozmsk/cawdec, CC BY-SA 4.0): Skoda factory conveyor with two fixed side cameras at 1 fps FullHD, 34 h of video; CWD1500 = 1,000 train / 250 val / 250 test frames for car and wheel detection (91 background-only images), rim classification on 21 rim classes from 31 identified. `dataclusterlabs/vehicle-wheel-detection` (CC0, 1,354 files). Brake pads: `brake-shoe/brake-pad-wear-znaqd` (see 2.7).

**Faces/plates:** tyre close-ups carry little risk [I]; `tire-flat` street photos and CAWDEC frames show cars and sometimes people.

**Data-quality concerns:** many "tyre quality" sets come from one or two workshops (domain narrow); mosaics/augmentation baked into Roboflow exports inflate image counts; TyreCV has few extreme depths and needs a stereo rig, so a phone-photo depth model cannot be validated on it directly; no set pairs a phone photo with a mm gauge reading.

**Verdict: strong for condition classification and DOT OCR; thin for mm tread depth** (one 508-pair stereo set). Say "tread depth validated on 508 gauge-measured stereo pairs" and nothing about single-camera phone accuracy.

### 4.2 Component 4: Car-wash context

| Set | Link | Images | Classes | Licence | What it shows | Faces / plates |
|---|---|---|---|---|---|---|
| wash-car-2 | https://universe.roboflow.com/hoangdarkdao/wash-car-2 | 237 | `0`, `1`, `2`, `3` (undocumented) | CC BY 4.0 (label) | fixed CCTV inside an automatic tunnel (brushes, wet floor, timestamp overlay) [F thumbs] | not checked |
| washing car (UM) | https://universe.roboflow.com/um-84bee/washing-car | 2,439 | `normal car`, `open doors`, `open hood`, `washing car` | CC BY 4.0 (label) | fixed fisheye/CCTV frames of a wash yard with timestamps [F thumbs] | readable Arabic-script plate in a night frame (`stage_data.md`) |
| car-wash (Ravshanbek's Workspace) | https://universe.roboflow.com/ravshanbeks-workspace/car-wash-00e9w | 27 | `car`, `truck`, `mini truck` | CC BY 4.0 (label) | fixed CCTV of a manual pressure-wash bay, workers visible; updated 2 months ago [F thumbs]; the workspace name sounds Uzbek but the scene's country was not verified [I] | workers (small), plates unclear |
| Car Dirt Classification | https://universe.roboflow.com/workspace-eakuj/car-dirt-classification | 647 (instance segmentation) | `birdpoop`, `body dust`, `mud`, `window dust` | CC BY 4.0 (label) | phone close-ups of dirty paint and glass [F thumbs] | none seen |
| Car dirt family by uploader "fall" | https://universe.roboflow.com/fall-on7ds | Car Dirt Detection 1.33k, clean-or-dirty 992, Clean cars 656, clean/dirty/damaged 1.3k [S] | | CC BY (label) [S] | not opened | |
| car-washing-polygonal-detection | https://universe.roboflow.com/bhumins-workspace/car-washing-polygonal-detection | 751 | 19 panel classes (wheel, hood, trunk, bumpers, doors, lights, glass...), polygons | CC BY 4.0 (label) | per-panel masks, handy for localisation | not checked |
| carwash-repair, Wash Bay Presence, car-wash (elsoudy `0/1`), car-wash (emstone smoke/flame) | see `stage_data.md` | | | | plates by design / unclear / off-domain | |
| HF `jjjlimaus/car-wash-dataset-manual` | https://huggingface.co/datasets/jjjlimaus/car-wash-dataset-manual | 6,278 files listed, YOLO yaml, PNG test/images | unknown | gated (restricted access) | cannot be inspected | |

Traps: `abc-3onz4/wash-643po` (1,050 images, `Step1`..`Step7`) and `objectdetection-vttse/wash` (415) are **hand-washing** sets; `national-chung-cheng-university-csie/dirty-vov5p` (5,101 images, dirty/clean) shows dirty windows/panels, not cars; `vuong-thanh/washing-car-adz5l` (80 images) is stock photos with watermarks; `aaryas-workspace/dirty-n4msw` (375 images, 23 part classes) shows dealer-lot cars with Indian plates readable. [F thumbs]

Not found: any drive-over or tunnel-entrance wash-lane dataset, any wet/dirty-car damage-labelled set, Kaggle car-wash image sets (16 hits, all business listings), any wash-lane video dataset (Commons clips are in `stage_data.md`). The interior-cleanliness set of Jayawickrama et al. (1,379 images, J. Big Data 2023) covers cabin trash, not exteriors [S].

**Verdict: thin.** Proof on public data is limited to single cues: wash-state / door-open detection (2,439 images), dirt-region segmentation (647), and vehicle-in-lane detection on a few CCTV frames. Damage-under-dirt robustness cannot be measured on real labelled data; build it by compositing mud/water onto damage images [I].

### 4.3 Component 5: Car-parts segmentation (to localise damage)

| Set | Link | Size | Classes | Annotation | Licence | Notes |
|---|---|---|---|---|---|---|
| Ultralytics carparts-seg | https://docs.ultralytics.com/datasets/segment/carparts-seg/ | 3,833 (3,156 / 401 / 276), 133 MB | 23: back_bumper, back_door, back_glass, back_left_door, back_left_light, back_light, back_right_door, back_right_light, front_bumper, front_door, front_glass, front_left_door, front_left_light, front_light, front_right_door, front_right_light, hood, left_mirror, object, right_mirror, tailgate, trunk, wheel | instance masks | CC BY 4.0 per docs page; origin Roboflow `car-seg` by Gianmarco Russo (Nov 2023) | the docs card says "223 classes" in one place (typo); the yaml header carries the AGPL-3.0 tag of the config file only |
| RF khpi/car-segmentation | https://universe.roboflow.com/khpi/car-segmentation-ga8mc | 9,011 | 15: Wheel, object, Bonnet, Frontbumper, Frontdoor, Frontfender, Headlights, Rearbumper, Reardoor, Rearfender, Rearlamp, Rockerpanel, Sidemirror, Trunklid, Windshield | instance masks | CC BY (label) | largest parts set; 3 years old |
| DSMLR Car-Parts-Segmentation | https://github.com/dsmlr/Car-Parts-Segmentation | 500 (sedans, pickups, SUVs; front/back/angled views) | 18: back_bumper, back_glass, back_left_door, back_left_light, back_right_door, back_right_light, front_bumper, front_glass, front_left_door, front_left_light, front_right_door, front_right_light, hood, left_mirror, right_mirror, tailgate, trunk, wheel | COCO masks + boxes | none in repo (GitHub licence field empty) | **plates and faces anonymised** (README); V2 re-annotation; paper Complex & Intelligent Systems 2021, 110 stars |
| Humans in the Loop (parts half) | see section 3 | 998 images | 21 parts | polygons | CC0 | |
| RF `ai-proyect/car-parts-segmentation-q1xai` | https://universe.roboflow.com/ai-proyect/car-parts-segmentation-q1xai | 1,023 | 17 | masks | CC BY (label) | |
| RF `fleetblox-car-damage/car-parts-bzaux` | https://universe.roboflow.com/fleetblox-car-damage/car-parts-bzaux | 2,256 | 33 classes, includes `License-plate` | boxes | CC BY (label) | web/stock photos, plates visible [F thumbs] |
| RF `team-data/car-parts-ybiev` | https://universe.roboflow.com/team-data/car-parts-ybiev | 8,739 | 50 mechanical parts (alternator, battery, brake caliper...) | boxes | public-domain label | parts-shop catalogue photos, not vehicles [F thumbs] |
| Pascal-Part | https://roozbehm.info/pascal-parts/pascal-parts.html | 10,103 trainval + 9,637 test (all PASCAL VOC 2010 classes) | car: back/front license plate, back/front/left/right/roof side, door, headlight, left/right mirror, wheel, window | part masks | no licence on page; VOC images are Flickr | images from the VOC 2010 site |
| LSCP | https://arxiv.org/abs/2311.11754 | 84,162 | 12 part types | auto-labelled boxes (semi-supervised) | paper CC BY 4.0, no data release found | |
| SYNDCAR | see section 3 | 245 | 28 parts | polygons | CC BY 4.0 | |
| Underbody parts | see 2.3 and 2.4 | | | | | for parts seen from below |

Faces/plates: DSMLR anonymised; HITL/SYNDCAR/Pascal-Part label plates; Roboflow sets not scanned. 

**Verdict: strong** for side/exterior panel localisation (20+ thousand images, 15-23 classes); thin for underside parts (see component 1).

### 4.4 Component 6: Change detection between two inspections of the same car

- **No public paired set** (same vehicle photographed at two times with new-damage ground truth) was found: web search for "before/after paired vehicle damage dataset", arXiv `vehicle damage change detection`, Kaggle, Roboflow (`vehicle before after damage`: nothing relevant). Rental/lease vendors (Hertz+UVeye, Click-ins, Tiliter, Ravin) and patents describe before/after pipelines on proprietary data; an ECML-PKDD 2020 lease-return paper notes that distinguishing new from pre-existing damage was a core problem and its data was not captured to a standard [S].
- Nearest public proxies:
  1. **AutoVI** (reference-based anomaly detection): train folder = good items, test folder = anomalies; metrics AUROC and sPRO. Fixed cameras at one station, the right shape for "compare against the car's own reference". Factory domain; CC BY-NC-SA 4.0. [F]
  2. **Roboflow `background-car`** (`before-undercarriage`, `after-undercarriage`): the labels hint at before/after states of a scan but their meaning is undocumented. [F]
  3. **Vehicle re-identification** for matching the same car across cameras/visits (the plate is the real key in our product): VeRi-776 (about 50,000 images, 776 vehicles, 20 cameras), VehicleID (221,763 images, 26,267 vehicles, front/back only), VeRi-Wild (416,314 images, 40,671 IDs, 174 cameras, 1 month) [S]; licences not found, plates and drivers are visible. 
  4. **Synthetic pairs** [I]: render the same 3D car, camera and HDRI with and without procedural damage using CrashCar101's MIT-licensed Blender pipeline (needs ShapeNet cars, whose terms are research-only, or Sketchfab CC BY cars, and Poly Haven CC0 HDRIs); pixel-exact "changed" mask for free.
  5. **Image-space compositing** [I]: paste HITL/CarDD damage crops onto clean images of the same car to make an "after" frame; weaker realism, no 3D.
- **Verdict: none (public).** Weekly per-car history cannot be validated on real data; claim only reference-based anomaly detection (AutoVI AUROC/sPRO) plus synthetic paired renders with declared method.

---

## 5. Recommended component proofs (what we can honestly measure)

General protocol: dedupe by perceptual hash within and across sibling projects; split by source (project/workspace) not by frame; report n and 95 % Wilson intervals (for a proportion near 0.9: n=100 gives about +-6 points, n=300 about +-3.4); keep real and synthetic numbers separate; declare every licence and the Roboflow "label only" caveat in the repo list; run a plate/face detector at ingest and never display raw frames from sets marked "plates visible".

| # | Component proof | Data | Split | Metric to report | What it can claim | What it cannot claim |
|---|---|---|---|---|---|---|
| P1 | Exterior damage detection + mask | CarDD if the form is approved, else HITL damage (814, CC0) + VehiDE; test on a source never used in training (SYNDCAR 245 CC BY, Coco Car Damage 80, CarDamagePanel 565) | by dataset | mask AP50 / box mAP50 per class; recall at a fixed false-positive rate per image | "damage found on phone/web photos of cars" | fixed-camera or wet/dirty performance |
| P2 | Damage localised to a part | SYNDCAR (damage and 28 parts on the same 245 images); train parts on Ultralytics/khpi/DSMLR/HITL parts | parts trained on others, evaluated on SYNDCAR | part-assignment accuracy of damage instances, part mask mAP | "damage reported with its panel" | |
| P3 | Undercarriage rust | `undercarriage-rust-detection` (2,055) + `underbody-car-defects` (519) after dedupe; own held-out 150-200 images drawn from a different project | by project, not by image | box mAP50; image-level AUROC and recall of `rust_severe` | "rust detection on ground-level underside photos; page-level baseline mAP50 about 61 %" | drive-over scan accuracy |
| P4 | Undercarriage oil leak | train `leak-4pyvb` (593) [+49 `oil-and-coolant-leak`]; test cross-domain on `Kebocoran Oli` in `sistem-suspensi` (603, trucks) | by source | box AP at IoU 0.5, image-level leak AUROC | "leak found on two independent sources" | leak vs dirt/mud separation (Pulsar says this needed thousands of examples) |
| P5 | Under-vehicle localisation + capture-quality gate | drive-over composites (588 + 417 + 1,058 + 343) for crop/under-car box; `image_analysis_under` (275 Good/Bad) for the gate | by workspace | crop IoU / mAP50; gate accuracy and AUROC | "we find the underbody in a composite scan and reject bad scans" | defect accuracy on composites |
| P6 | Underbody parts | Indonesian pit-view sets (about 2,600 images: `bagian-bawah` 1,000 + `system-suspension` 1,000 + `sistem-suspensi` 603); the Tekion sets look derived from the same photos, so hold out by perceptual hash and evaluate on an own 150-image pit-view set | by image hash and workspace | per-class mAP50 (differential, exhaust, leaf spring, shock absorber, wheel) | parts localisation under heavy vehicles | passenger-car lifts |
| P7 | Tyre condition | train TyreNet (1,698, CC BY); test cross-dataset on Kaggle Tyre Condition (548; map NEW+SERVICEABLE vs UNUSABLE) and `warcoder` | by source | accuracy, F1, AUROC, confusion matrix | "worn/defective tyre screen on phone close-ups" | mm depth |
| P8 | Tread depth in mm | TyreCV (508 stereo pairs, gauge mm) | grouped k-fold by tyre; report bucket accuracy and MAE in mm; compare with the HF Space's 0.35 mm external reference | MAE (mm), within-1-mm rate | "mm estimate from a calibrated stereo pair, on a small public set" | single-camera phone accuracy |
| P9 | DOT code reading | TyreAI OCR (4,054) + DOT validator regex (week/year range) | by split in the project | exact-match of the 4-digit date code, character accuracy | "tyre age from DOT" | night/rain |
| P10 | Car-wash cues | `washing-car` (2,439: normal / open doors / open hood / washing) and `car-dirt-classification` (647 masks) | by project | state accuracy per class; dirt mask IoU | "wash-state and dirt regions" | wet/dirty damage recall |
| P11 | Change detection | AutoVI underbody categories (reference-based AD) + rendered paired scenes (CrashCar101 pipeline or own Blender) | AutoVI official split; synthetic by car model/HDRI | image AUROC and sPRO (AutoVI); change-mask IoU and new-damage instance recall (synthetic) | "reference-based change detector tested on factory data and on synthetic pairs" | weekly real-world history |

Cheapest high-value order [I]: P2 (SYNDCAR, uniquely joint labels) and P1, then P3/P4 (undercarriage) and P7/P8/P9 (tyres), then P5/P6; P10 and P11 are only honest if framed as single-cue and synthetic respectively.

---

## 6. Verdict summary

| Component | Verdict | One-line reason |
|---|---|---|
| 1 Undercarriage / underbody | **thin** | real rust (about 2,000 images) and leak (593) labels on ground-level photos, drive-over composites exist (about 2,400) but unlabelled for defects, no EV-battery or exhaust-damage data |
| 2 Exterior damage | **strong** | CarDD, VehiDE, HITL, SYNDCAR and 50 k+ Roboflow images with masks; domain is phone/web photos only |
| 3 Tyres | **strong (classification, DOT) / thin (mm depth)** | six condition sets and a 4,054-image DOT OCR set; one 508-pair gauge-measured stereo set |
| 4 Car wash | **thin** | three CCTV-style wash sets with unclear labels, a dirt-segmentation set, nothing wet/dirty with damage labels |
| 5 Parts segmentation | **strong** | Ultralytics 3,833, khpi 9,011, DSMLR 500, HITL 998 images, SYNDCAR 245 |
| 6 Change detection | **none** | no paired public data; use reference-based anomaly detection and declared synthetic pairs |

---

## 7. Corrections to earlier notes in this folder

- `stage_data.md` said Kaggle Coco Car Damage has "152 files"; the dataset has 80 images (59 / 11 / 8).
- `stage_data.md` left the licence of `underbody-hard-parts-seg` unread: it is BY-NC-SA 4.0.
- `stage_data.md` and the brief list `undercarriage-rust-detection` as about 1.2k images in older snapshots; the live page shows 2,055.
- `astana_tender.md` said no undercarriage dataset exists beyond the Roboflow leads; this search found the drive-over composite family, the Indonesian pit-view family, a vehicle leak set, AutoVI underbody categories, and several mislabelled "undercarriage" projects.
- UnderCar shows 14 images on its project page (27 in a search snippet).
- AutoVI: Zenodo v1.0.0 = 3,950 images / 6 categories; the website advertises 7,184 images / 11 classes.

## 8. Not found / not verified

- Any labelled drive-over (speed-bump) undercarriage defect set; any EV-battery enclosure damage set; any exhaust-damage set; any fixed-camera exterior-damage set; any paired before/after vehicle set; any wet/dirty-car damage set; any wash-lane video data set. (Searched Roboflow, Kaggle, Hugging Face, Zenodo, GitHub, arXiv, web search in English/Russian/Chinese.)
- Mendeley full-text search (site search not drivable here) and IEEE DataPort listings beyond two hits; Chinese platforms (Baidu AI Studio, Tianchi, Gitee) were not accessible through search.
- Image provenance and true licence of every Roboflow set; whether the stitched composites come from one commercial system; meaning of classes `0-3` in `wash-car-2`, `tread_0.0/0.9/1.8`, `before/after-undercarriage`; the 8th VehiDE class; image counts of Mendeley `dfzbm2ktss`.
- Plates/faces were checked only in 4-8 thumbnails per project.
- Numbers from VeRi/VehicleID/VeRi-Wild, ACV/UB metrics, QBurst claims and CrashCar101's 101,050 images are [S].
- The "fall" Roboflow car-dirt family (992 / 656 / 1.3k / 1.33k) was not opened.

---

## 9. Source list

Roboflow Universe (prefix https://universe.roboflow.com/): `anirudh-qjb2e/undercarriage-rust-detection-f9grj`, `anirudh-qjb2e/rust_detection_segmentation`, `anirudh-qjb2e` (workspace: 6 projects, 4.33k images), `karthik-z0q6k/underbody-car-defects`, `/underbody-car-defects-2`, `/underbody-car-defects-6`, `lefa/underbody-heyzf`, `zaid-suimd/underbody-screw-annotation`, `t-muywg/underbody`, `vw-nfqmp/underbody-inspection-system`, `srikarthicks-workspace/underbody-inspection`, `psbhatt-tekion-com/underbody-hard-parts-seg-sk6dq`, `psbhatt-tekion-com/car-parts-detection-eove9`, `car-segmentation-k5akd/undercarriagecropping`, `copart-i960g/undercarriage-cropping`, `car-segmentation-iq9jj` (workspace: `new-car-seg`, `undercarraige`, `image_analysis_under`, `background-car`, `top-position`, `car-damage-haa6l`, `car-parts-9vig8`, `cartire_classification`, `cartire-oxtxu`), `carp10/undercarriage-y5epg`, `under/undercar-xaflv`, `under/test2-toki0`, `under/kolong-mobil`, `deteksi-objek-ciufr` (workspace: `bagian-bawah-kendaraan`, `system-suspension`, `sistem-suspensi-0jqsm`, `projek-suspensi`), `md-jubayar-alam-rafi/under-vehicle-surveillance-sys.`, `suhass-workspace-y4ouv/computer-vision-for-undercarriag`, `s-workspace-fz2uo/leak-4pyvb`, `hamza-bashir-nzrcf/oil-and-coolant-leak-drzbm`, `dataset-fod/tumpahan-oli`, `tryout/oil-leak`, `chiranjeevi-p2jmx/oil-leak-detection-axcq4`, `custom-yolov5-1lqs1/oil-leak-discovery`, `newproject-7c1cc/engine-oil-detection`, `dhip-tupart/fluid-j85l3`, `brake-shoe/brake-pad-wear-znaqd`, `rep2/under`, `odometer-zig37/chassis-t8cl4`, `trainer-jaf9y/car-damage-v9lz4`, `changs-workspace-hnorg/vehicle-damage-gwmh4`, `ayhan-gul-hgudf/car-damage-rlogo`, `skillfactory/car-damage-c1f0i`, `truck-damage/car-damage-dataset-iqcfm`, `accident-k7ls5/vehicle-damage-dataset`, `car-damages-djnvi/car-damage-l7wgd`, `dt-xjexo/car-damage-uffwb`, `shinhatest/car-parts-segmentation-0gniu`, `khpi/car-segmentation-ga8mc`, `ai-proyect/car-parts-segmentation-q1xai`, `fleetblox-car-damage/car-parts-bzaux`, `team-data/car-parts-ybiev`, `tyreai/ocr-yphpv`, `group5-odr67/tread-dbn2s`, `automatics-number-plate-recognition/tyre-thread-depth`, `mark-aft7n/tire-tread`, `eva-chen-ilht2/tire-tread-segmentation-orjof`, `asmaakhaled/car-tire-zwml5`, `tyre-tj9mt/tyre-q73y8`, `muhammad-anif-zuhrul-anam-eaufq/crack-tyre-detection`, `car-parts-straight-photos/tire-flat`, `hoangdarkdao/wash-car-2`, `um-84bee/washing-car`, `ravshanbeks-workspace/car-wash-00e9w`, `workspace-eakuj/car-dirt-classification`, `bhumins-workspace/car-washing-polygonal-detection`, `abc-3onz4/wash-643po`, `objectdetection-vttse/wash`, `national-chung-cheng-university-csie/dirty-vov5p`, `aaryas-workspace-qvuhz/dirty-n4msw`, `vuong-thanh/washing-car-adz5l`, `fall-on7ds`.

Kaggle (prefix https://www.kaggle.com/datasets/): `maxfinkenbeiner/tyrecv-two-views-are-all-you-need`, `sameersambhare1/tyre-condition-classification-dataset`, `warcoder/tyre-quality-classification`, `numberfive/tire-tread-photos`, `jehanbhathena/tire-texture-image-recognition`, `rhammell/full-vs-flat-tire-images`, `prakarshawasthi/tyre-dataset-detection-severity-classification-rul`, `nethhari/sri-lanka-bus-tire-condition`, `adamnovozmsk/cawdec`, `humansintheloop/car-parts-and-car-damages`, `hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection`, `lplenka/coco-car-damage-detection-dataset`, `hamzamanssor/car-damage-assessment`, `samwash94/comprehensive-car-damage-detection`, `prajwalbhamere/car-damage-severity-dataset`, `anujms/car-damage-detection`, `issamjebnouni/cardd`, `gabrielfcarvalho/cardd-with-yolo-annotations-images-labels`, `viniciusbonicio/underbody-front-floor`, `dimpleb/image-dataset-for-surface-oil-and-water-leakage`, `zaidmahmud/underbody-screw-annotated-dataset`, `blank1508/autovi-dataset-version-1-0-0`, `rainyseason72/underbody-pipe-autovi`.

Zenodo / Mendeley / IEEE DataPort: AutoVI https://zenodo.org/records/10459003 ; https://autovi.utc.fr ; datasheet https://autovi.utc.fr/download/datasheet.md ; TyreNet https://data.mendeley.com/datasets/32b5vfj6tc ; tyre life prediction https://data.mendeley.com/datasets/dfzbm2ktss ; SYNDCAR https://data.mendeley.com/datasets/hzpj48krdt ; Pixemantic https://ieee-dataport.org/documents/pixemantic-vehicle-damage-29 ; port inspection (not relevant) https://ieee-dataport.org/documents/port-wide-view-vehicle-inspection.

Hugging Face: https://huggingface.co/datasets/tugberkkalay/autodamageiq-vehicle-damage-dataset ; https://huggingface.co/datasets/Naiscorp/car-damage-dataset ; https://huggingface.co/datasets/kadireks/cardamagepanel-originals ; https://huggingface.co/datasets/DrBimmer/car-parts-and-damage-dataset ; https://huggingface.co/datasets/jjjlimaus/car-wash-dataset-manual ; https://huggingface.co/datasets/thejashr/vehiDE_processed ; https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study (experiment artefacts, about 120 JPEG) ; https://huggingface.co/spaces/UnCodeSpecial/tire-tread-depth ; https://huggingface.co/datasets/NMiriams/Defective_Tires.

GitHub: https://github.com/suhasrv/computer-vision-for-undercarriag-main-project-and-dataset ; https://github.com/dsmlr/Car-Parts-Segmentation ; https://github.com/JensPars/CrashCar_procedural_generation ; https://github.com/fengfenghencai/V-Sentry ; https://github.com/share2code99/chassis_component_yolo11_detection ; https://github.com/pavinvelan/Automatefd-Vehicle-Underbody-Monitoring-System.

Papers and project pages: CarDD https://cardd-ustc.github.io/ and https://arxiv.org/abs/2211.00945 ; VehiDE https://doaj.org/article/6e0eb600059440b390431a42dfe8d138 ; CrashCar101 https://arxiv.org/abs/2311.06536 and https://crashcar.compute.dtu.dk/ ; LSCP https://arxiv.org/abs/2311.11754 ; Pascal-Part https://roozbehm.info/pascal-parts/pascal-parts.html ; Ultralytics https://docs.ultralytics.com/datasets/segment/carparts-seg/ ; Humans in the Loop https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/ ; DSMLR paper https://doi.org/10.1007/s40747-021-00397-8 ; rig-aware undercarriage 3D https://arxiv.org/abs/2601.14208 ; IMU underbody damage https://arxiv.org/abs/2509.01719 ; multi-view factory inspection https://arxiv.org/abs/2509.26454 ; small-dent dataset https://arxiv.org/abs/2508.15431 ; tyre-pattern recognition https://arxiv.org/abs/2607.23979.

[S] leads: ACV Auctions oil defects https://techconnectworld.com/World2023/a.php?i=242 ; Buffalo undercarriage image classification https://techconnectworld.com/World2022/a.php?i=268 ; QBurst tyre inspection https://www.qburst.com/downloads/tire-inspection-system.pdf ; Taazaa/Goodyear https://www.taazaa.com/success-story/mobile-app-for-tire-tread-analysis ; leasing return paper https://mlanthology.org/ecmlpkdd/2020/jameel2020ecmlpkdd-automation ; vehicle re-identification surveys https://arxiv.org/abs/2401.10643 ; VeRi-Wild https://openaccess.thecvf.com/content_CVPR_2019/papers/Lou_VERI-Wild_A_Large_Dataset_and_a_New_Method_for_Vehicle_CVPR_2019_paper.pdf ; interior cleanliness https://doaj.org/article/593d1bb186a449e5bd4e8d22f2202e94 ; Click-ins/Tiliter/Hertz before-after pages (via search results).
