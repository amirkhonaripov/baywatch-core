# Astana Hub "VisionFlow AI" tender, under-vehicle (speed-bump) inspection, and related dealer/service CV tenders

Research date: 2026-10-10. Scope: the Astana Hub tender, who (if anyone) won it, whether the winner built speed-bump undercarriage inspection, related tenders/pilots in KZ/UZ/CIS, and public datasets.

Tag legend: **[F]** = read on the page itself (fetched with curl and text-extracted), **[S]** = search-engine summary only (page not read by me), **[I]** = my inference. "Not found" means I searched and found nothing; it does not prove absence.

Working files (raw HTML/text of every page read) are in `/private/tmp/claude-501/-Users-diznect-Downloads-hackathon-final/d15fe6b0-783c-41a4-937a-85a23a467fba/scratchpad/astana/`.

---

## 0. Bottom line (read this first)

1. **The teammate's claim is partly true but is attached to the wrong company.** Speed-bump undercarriage scanning at a service-centre entrance is real and is being sold in Kazakhstan by **Pulsar.ai (product "SmartRail", Almaty)**, not by anyone I could link to the VisionFlow tender. Pulsar's pilot is at a **Hyundai Premium Center in Almaty** (March 2026). [F: Digital Business, Kapital, Forbes.kz, Kursiv]
2. **The VisionFlow tender winner / executor: NOT FOUND.** Astana Hub's tech-task pages show no winner, no status flag and no results page; per Astana Hub's own description the customer picks the contractor privately from submitted applications. [F]
3. **No evidence links Pulsar.ai to the VisionFlow tender, and no evidence links "ИП Smart Tech" to undercarriage scanning.** The tender scope (bay occupancy + plate reading + ETD + queue) contains nothing about undercarriage or inspection. [F]
4. The only Astana Hub link to Pulsar I found: co-founder Baglan Olzhabayev says he completed "a programme of Astana Hub" (an accelerator, unspecified) among others. That is not a tender win. [F: Forbes.kz]
5. Possible source of the confusion [I]: (a) Pulsar's speed-bump design is in the Kazakh press in April-August 2026, the same months the VisionFlow tender was open; (b) a different Astana Hub tender (PROMAT-SYSTEMS, night plate reading) describes a speed bump before a barrier, which is only there to slow cars down. Neither is an inspection system.

---

## 1. The VisionFlow tender

### 1.1 Page facts [F]

URL: https://astanahub.com/ru/tech_task/tz-platforma-avtomatizatsii-servisnykh-zon-visionflow-ai-mvp (also /en/ and /kk/ versions exist; internal task id 1192, from the page's "submit solution" link).

| Field | Value on page |
|---|---|
| Title | ТЗ: Платформа автоматизации сервисных зон «VisionFlow AI» (MVP+) |
| Customer | "ИП Smart Tech" (sole proprietor); the customer avatar is just the letter "И" and links to a personal Astana Hub user profile, not a company profile |
| Geography | "Только РК" (Kazakhstan only) |
| Budget ("Сумма") | 1500000, **no currency shown** on the page in ru/en/kk. [I] Almost certainly tenge (KZT): other KZ tasks on the same site show 100000 / 200000 / 1000000 for small jobs, and the page says Kazakhstan-only. At the 9 Oct 2026 rate quoted on Kapital.kz (about 453 KZT per USD) that is roughly **USD 3,300**. |
| Reward form | "Денежное вознаграждение (оплата по этапам/спринтам)" (money, paid per milestone/sprint) |
| Number of applications | 10 (as of fetch, 2026-10-10) |
| Submission deadline | 20.04.26 inclusive. **Passed about 5.7 months ago.** |
| Product status | MVP |
| Task type / sphere / area | ICT task / Automotive industry / Intelligent control systems |
| Product type | Software/IS and mobile application |
| Responsible person | Александр Мещеряков (name as shown; contact details are on the page, not reproduced here) |
| Attached TZ file | None visible on this page (the car-wash and LiTRO tasks have a "Скачать файл" link; this one does not) |
| Winner / result / status | **None shown.** The page still renders a "Отправить решение" (submit solution) button after the deadline; I could not tell whether it is functional. |

### 1.2 Full requirements [F]

**Goal.** Software to automate service-post occupancy using computer vision: determine zone status (free/occupied) in real time, identify vehicles (LPR) and predict completion time (ETD) with no manual input.

**Architecture.** "Hybrid Web/Mobile": single backend with all AI logic and DB; frontend a PWA mirrored as a Telegram Mini App (works by direct browser link if the messenger is blocked).

**Module 1, video analytics / CV.**
- Detection with YOLOv8/v10 of cars in 6+ zones simultaneously.
- LPR via Nomeroff Net "or analogues", stable reading of Kazakh plates in any lighting.
- Activity analysis: detect stages start / process / finish from visual data (motion analysis and ROI change).
- RTSP streaming with automatic reconnect.

**Module 2, ETD engine ("Zero-Manual Calculation").** ETD from (1) chosen service type (Standard/Express), (2) current stage confirmed visually by AI, (3) historical average service time at that site.

**Module 3, queue dispatcher.** Dynamic rescheduling (cascading recalculation on early exit or delay); notifications (Web Push / Telegram) about queue shifts; history (plate, session time, photo of entry/exit).

**Stack.** Python 3.10+ FastAPI, PostgreSQL, Redis; React/Next.js (PWA + Telegram Mini App); OpenCV, PyTorch, YOLO; Docker, NVIDIA GPU servers, MQTT.

**Acceptance criteria.**
1. Bay status changes in the DB within **5 seconds** of the physical event, no human involved.
2. Kazakh plate recognition (including dirty plates) **>= 90%** in daytime/evening.
3. App works in the browser when the messenger is blocked.
4. Waiting time in the UI updates in real time.

**Milestones.** S1: plate + car detection prototype on 2 streams. S2: smart-queue logic and predictive algorithm (backend). S3: PWA UI and Telegram Mini App. S4: field test on site and finalise notifications.

**Code/architecture requirements.** Source code handed to the customer's git repo; Swagger/Redoc API docs; scalable to 50+ cameras; REST API fully decoupled from the front end for a later Flutter/React Native app.

**Expected effect.** Full automation of bay-occupancy accounting; higher throughput via ETD and smart queue; transparent attendance/service-time analytics for the owner; fewer queue conflicts.

**Note from customer.** Looking for a team or an experienced full-stack developer with CV expertise; "not a business-card website", the value is the AI core (plates + bay status from RTSP); backend/frontend separation mandatory; "Zero-Manual" principle; wants real cases with YOLO / OpenCV / Nomeroff Net, streaming and GPU-load optimisation, and readiness for milestone-based payment; applicants should briefly describe video-analytics experience.

**What the tender does NOT mention [F]:** undercarriage, speed bump, vehicle inspection, damage detection, dealer, brand, or any named site/business type.

### 1.3 Results / winner page

- **Astana Hub results page: not found.** I read the ru/en/kk tender pages, the customer profile page, and crawled the full public tech-task listing (about 862 tasks over 87 pages). The listing shows only title, customer, deadline, area and application count; there is no winner/"executor" field anywhere. [F]
- Astana Hub's own article on the module says contractor selection "is carried out by the customer from among the submitted applications" (https://astanahub.com/en/article/vozmozhnosti-dlia-razrabotchikov-astana-hub-zapustil-ploshchadku-tekhnologicheskikh-zadach, 25 Jul 2025). [F] So winners are not published by the platform.
- **Winner/executor of VisionFlow: not found anywhere** (web search in ru/en, GitHub, domain probes, news).

### 1.4 Who is "ИП Smart Tech"? Not found

- The Astana Hub customer record is an individual Astana Hub user (name Александр Мещеряков), level "Newcomer", location Kazakhstan, 0 followers, "no activity on the platform"; no company description, no business type. [F]
- Web searches (ru/en) for "Smart Tech" + Мещеряков + СТО / автомойка / видеоаналитика, plus LinkedIn/Telegram variants, returned nothing relevant. [S]
- Business type (car wash / service-station chain / dealer / developer-reseller): **not determinable.** [I] The wording ("ищем команду или Fullstack-разработчика", a 1.5M budget, "поэтапная оплата") reads like a small operator or integrator commissioning an MVP, but that is inference.
- Domain probes (not conclusive): visionflow.kz and smarttech.kz / smart-tech.kz did not connect; visionflow.ai redirects to an unrelated product (hipilot.ai); visionflow.io and visionflow.app are unrelated businesses (gym websites; architects' software). [F, negative]
- GitHub search for "VisionFlow" returned only unrelated third-party repos (traffic analytics / YOLO workflow projects by other authors; no Telegram Mini App, PWA, bay-occupancy or ETD). [S] I found no repo, demo video, LinkedIn or Telegram post for a "VisionFlow AI" service-bay product.

### 1.5 Is the product "VisionFlow AI" live anywhere? Not found

No news, site, repo or demo for a service-bay product under this name. The tender is the only trace.

---

## 2. Under-vehicle inspection with speed-bump cameras

### 2.1 Verdict on the teammate's claim

| Sub-claim | Status |
|---|---|
| Someone in Kazakhstan built entrance undercarriage inspection for a service centre/dealer, using a speed-bump-style unit | **Verified: Pulsar.ai SmartRail.** [F] |
| That company won / implemented the VisionFlow tender | **Not verified; no evidence found.** Nothing in four press articles on Pulsar mentions VisionFlow, Smart Tech, Astana Hub tenders or a service-bay product. [F, grep of all four articles] |
| The VisionFlow tender included undercarriage inspection | **Refuted for the tender text** (nothing on undercarriage in the TZ). [F] |

### 2.2 Pulsar.ai / SmartRail: what the press says

Sources: Digital Business (2 Apr 2026), Kursiv (6 Apr 2026), Kapital.kz (11 Jun 2026), Forbes.kz (25 Aug 2026). All are interviews with the founders; **every number below is a vendor claim, not independently validated.** [F]

- **Company.** Pulsar.ai, Almaty. Co-founders Baglan Olzhabayev (ex-founder of the Beep roadside-mechanic app; says he went through MOST Ventures, an Astana Hub programme and Draper University) and CEO Zhanar Galymzhan. Started Feb 2026 on self-funding (tens of millions of tenge; they sold personal assets); now open investment round. Hardware patent in process. [F: Digital Business, Forbes]
- **Device.** SmartRail: a flat/low unit on the road surface or at the entrance to the service area; the car drives over it at 5-10 km/h; cameras plus a small AI computer take a burst of photos (Kapital: 10-500 images in about 3 s; Digital Business: report in 30-60 s; Kursiv: 30 s to a couple of minutes). Kapital says it "looks like a speed bump". Forbes: the first pilot used a flat unit with a section of paving stones removed before the service entrance; "over time we realised it is better to build SmartRail into a speed bump, where drivers slow down: sharper images and the bump protects the device". Digital Business (2 Apr) called the speed-bump version a *plan* for toll-road entries. So: flat unit in the dealer pilot, speed-bump form is the direction being adopted. [F]
- **Hardware cost.** First prototype about **USD 2,000**; upgraded version (better camera, heating and air-blowing for rain/snow, Raspberry Pi as the "brain") about **USD 3,700**; built from off-the-shelf parts bought in Almaty electronics shops. [F: Forbes]
- **What it detects.** Visual defects on the underside: oil/coolant/brake-fluid leaks, exhaust damage, boot/dust-cover condition, deformed underbody protection, corrosion of brake lines, dents/cracks; Forbes also lists tyre wear in the founders' list of defects. Traffic-light output (green/yellow/red); low-confidence (<60%) findings marked "needs observation". Does not look inside the engine. Plate number is linked automatically to the report. [F]
- **Claimed accuracy (inconsistent across articles).** ">80%" (Digital Business body), "85-90%" (headline/Kursiv), "85-95%" by defect type (Kapital: leaks to 90%, corrosion to 95%, others to 88%), "83-95.5%" (Forbes). Known failure cases: full underbody shields, thick mud, very fresh micro-cracks, defects hidden by other parts. [F]
- **Pilot results (claims).** Kapital: of the first 2,000 cars, 23.6% had potentially critical defects; Forbes: "every fourth car"; Forbes: 150k data sets in two weeks; Digital Business: 1,000+ sessions processed (these figures do not reconcile). [F]
- **Pilot site.** Hyundai Premium Center, Almaty (scanner at entrance, records the car's state before acceptance; Digital Business). I found no confirmation from Hyundai itself. [F/S]
- **Business model (inconsistent).** Digital Business: free install, subscription to 300,000 KZT/month for dealers, taxi fleets, car-sharing; paid detailed report from 500 KZT. Kapital/Forbes: operator pays per scan or CPA (dealer pays a lead fee when a defect brings the car to service), Pulsar pays for hardware. [F]
- **Next steps claimed.** Aport malls in Almaty (parking entries), toll roads via KazAvtoZhol (written reply that they will consider a presentation/demo), Smart Bridge/eGov talks via NIT (Digital Business), DraperU "Hero Training" (US). **Expansion plan names Uzbekistan, Kyrgyzstan, Azerbaijan** (Kapital); Forbes says Central Asia and the Caucasus. [F]
- Also: free Telegram bot where a driver sends a phone photo of the underbody for AI feedback. [F]

I found no Pulsar presence in Uzbekistan, with Toyota or BYD, or as an Astana Hub resident (search summaries only). [S]

### 2.3 Other suppliers and what they detect

| Vendor | What it is | Detects / notes | Price evidence |
|---|---|---|---|
| **UVeye** (Israel/US) | Drive-through tunnel/floor systems: Helios (underbody), Artemis (tyres), Atlas / Atlas Lite (exterior) | Underbody: leaks, corrosion, rust, missing/broken parts. Tyres: tread depth, wear patterns, DOT data, sidewall bulges, broken belts, mismatched/aged tyres, nails, rim damage. Exterior: scratches, dents, paint, cracked lights, windshield chips. Vendor claims for dealerships: 70%+ fewer false damage claims with timestamped before/after scans, 30%+ more profit per repair order, 4+ CSI points. [F: uveye.com/how-it-works, /dealerships, /products] | No list price. Secondary source: lease of about **USD 6,000-7,000 per lane per month**, one Orlando dealer about 5,000; extra scans USD 2.50 over the limit; early-removal fee USD 8,500 per site. [S: thepricer.org, treat with caution] GM dealer incentive programme exists (2024 PDF). [S] Investors reported to include GM, Hyundai, Volvo and (2019) Toyota Tsusho. [S/F: Forbes.kz names GM, Hyundai, Volvo] |
| **Dahua UVSS** | Security-oriented fixed/portable under-vehicle scanners | Built to find hidden objects (threats), not mechanical defects. | US reseller listings (both "Discontinued - not sold in US"): portable DH-MV-VDM5021E-00 **USD 22,140**; fixed DH-MV-VDF5020CE-00 **USD 32,400**. [F: networkcamerastore.com] |
| **Hikvision UVSS** | Fixed and mobile UVSS; mobile unit has speed-sync module up to 30 km/h | Threat/contraband detection; stored images can be linked to plates. | No list price found. [S] |
| **ISS SecurOS UVSS, Vehant DuoScan, Rapiscan IVUS, Northrop (Perceptics)** | Security UVSS | Foreign-object / contraband detection, "digital fingerprint" comparison with earlier scans, plate linkage | No prices found. [S] |
| **Contrastech** (via Moscow office), Russian vendors | Camera-based underbody check systems | Security use; I found **no Russian vendor of dealer-grade mechanical-defect underbody scanners and no prices** (searches in Russian). [S] |
| **AUTOSCAN (Russia)** | Arch of cameras at a taxi-park entrance: full body inspection in 15 s, plate recognition, comparison with earlier scans, edge processing | Exterior damage, not underbody. Vendor blog/case. [S] |
| **Skaivision (US)** | Not an inspection scanner but dealership **service-flow analytics** from existing CCTV (see section 3.4) | n/a | n/a |

Take-aways [I]: (a) the market splits into security UVSS (expensive, hidden-object oriented, tens of thousands USD) and AI "vehicle health" scanners (UVeye by subscription, Pulsar as a USD 2-4k DIY-class device); (b) a cheap, speed-bump-type unit is technically plausible per Pulsar's own build cost, but the accuracy numbers are unproven; (c) the Uzbek market has no reported vendor.

---

## 3. Other tenders, pilots and comparable products (dealer / service / car wash / plates)

### 3.1 Astana Hub tech tasks (all fetched 2026-10-10)

I crawled the whole listing (862 tasks) and keyword-filtered titles/descriptions, then read the relevant pages. **No task on undercarriage, speed bumps or entrance vehicle inspection exists on the platform.** Related tasks:

| Task | Customer | Deadline / state | Apps | Notes |
|---|---|---|---|---|
| [VisionFlow AI service zones](https://astanahub.com/ru/tech_task/tz-platforma-avtomatizatsii-servisnykh-zon-visionflow-ai-mvp) | ИП Smart Tech | 20.04.26, **closed** | 10 | Section 1. Winner unknown. |
| [Intelligent system for automatic accounting/control of car washes](https://astanahub.com/ru/tech_task/intellektualnaia-sistema-avtomaticheskogo-ucheta-i-kontrolia-avtomoek) | ONE AI | **19.10.26, still open (9 days)** | 7 | SaaS on the wash's existing IP cameras: recognise arriving cars, build a digital visit route, match service to orders and payments, flag services with no payment record, photo/video evidence; product status "Idea"; budget field 0; attachment with TZ not downloaded (file download not authorised). Customer-side contact named on page: Ким Дмитрий Евгеньевич. Closest live tender to our idea. [F] |
| [Night plate recognition at parking entrance (IR, headlight glare, blur)](https://astanahub.com/ru/tech_task/ustoichivoe-nochnoe-raspoznavanie-gosnomerov-rk-na-vezde-parkovki-ik-podsvetka-zasvetka-farami-raz) | ТОО PROMAT-SYSTEMS (Promat Parking) | 05.10.26, **closed 5 days ago** | 12 | Ordinary 4 MP IP camera over the entrance; **speed bump before the barrier slows cars**; today ~**98% day, ~87% night** on Kazakh plates (format NNN ABC NN); target >=95% night; payment under a development contract after a successful pilot at a live Astana site; recordings only under NDA. Useful as a real-world LPR baseline. [F] |
| [Car damage assessment from photo/video](https://astanahub.com/ru/tech_task/razrabotka-intellektualnogo-servisa-dlia-avtomaticheskoi-otsenki-povrezhdenii-avtomobilia) | ТОО LiTRO RSA | 13.11.25, closed | 4 | MVP, mobile app; budget field 200000; negotiated reward; for insurers and partner service stations. [F] |
| [STO module in app (booking, ratings, mechanic load)](https://astanahub.com/ru/tech_task/razrabotka-modulia-dlia-stantsii-tekhnicheskogo-osmotra-v-prilozhenii) | LiTRO RSA | 09.09.25, closed | 3 | Shows "current workload of mechanics" per station, no CV. [F] |
| Predictive breakdown AI | LiTRO RSA | 21.04.25, closed | 1 | [S: listing only] |
| [Predictive technical-fault system](https://astanahub.com/ru/tech_task/razrabotka-sistemy-prognozirovaniia-tekhnicheskikh-neispravnostei-avtomobilei) | Agrostream | 10.09.26, closed | 4 | Telemetry-based, no CV; status Idea. [F] |
| Video analytics of car and visitor flows at EXPO | АО НК QazExpoCongress | 27.03.26, closed | 5 | [S: listing only] |
| [Orion Pro + Hikvision (access control, ANPR, speed)](https://astanahub.com/ru/tech_task/orion-pro-hikvision) | NF Labs | 25.12.25, closed | 1 | Plant-site ANPR/access integration; budget 1,000,000. [F] |

Winners/executors: **not published for any of them.**

### 3.2 Kazakhstan, outside Astana Hub

- **Pulsar.ai SmartRail at Hyundai Premium Center, Almaty**, pilot from March 2026 (section 2.2). [F]
- **Kazakhstan technical-inspection reform:** transport ministry order No. 58 of 16 Mar 2026 (in force 29 Mar 2026): authenticity of passing inspection to be checked by an automated system with AI elements (Digital Business, Tengrinews, Bizmedia summaries). I did not read the order; it concerns state inspection stations, not dealers. [S]

### 3.3 Uzbekistan / IT Park Uzbekistan

- **No tender, pilot or vendor for dealer/service-centre computer vision found** (IT Park, UzAuto Motors, Toyota/BYD dealers; searches in ru/en). [S, negative]
- Unrelated or adjacent items: Streetpark Systems won a Tashkent city parking information system tender (Feb 2025, Tashkent Invest); Tass Vision is an Uzbek computer-vision/AI-analytics startup (no automotive client found); customs uses scanning complexes (Z-Portal) for border inspection. [S]
- Pulsar.ai names Uzbekistan as a target market (Kapital, 11 Jun 2026). [F] **Possible competitor or partner within 6-12 months** [I].
- Context only: BYD's first Uzbek dealer opened 16 Mar 2026; Toyota Qushbegi dealership opened May 2026. [S]

### 3.4 Closest international analogue to the service-flow idea

**Skaivision (US)**, case study "Fred Anderson Toyota of Raleigh" (vendor page). Existing camera infrastructure, about 30 cameras watching technician downtime, customer greeting time, vehicle time in bays, customer-contact timing, overall repair and guest cycle time; dealership has about 250 vehicles/day and about 40 bays; reports basic service time from 75-90 min to under 60 min (customer-reported), about USD 1M added annual revenue "attributed to operational efficiencies", since Jan 2024. Reynolds & Reynolds partnership. Vendor/partner marketing, no independent validation; the page itself notes the service-time figure is customer-reported. [F: skaivision.lovable.app; S: Reynolds coverage]

---

## 4. Public datasets (brief; licences as read where possible)

**Bottom line: I found no public dataset of vehicle undercarriage images labelled for leaks/damage/rust/foreign objects.** All undercarriage work found uses proprietary data.

| Dataset | Content | Licence | Status |
|---|---|---|---|
| **TyreNet** (Mendeley Data, doi 32b5vfj6tc) https://data.mendeley.com/datasets/32b5vfj6tc | 1,698 tyre images: 866 defective (cracks, wear patterns, other), 832 good; classification | **CC BY 4.0** [F] | Usable for tyre condition classification; no tread-depth measurements |
| Roboflow Universe "Car Damages Kaggle" https://universe.roboflow.com/ai-proyect/car-damages-kaggle | General car-body damage (scratch, dent, corrosion classes) | CC BY 4.0 per search summary; **page returned 403 to my fetch** [S] | Body photos, not undercarriage |
| Roboflow "tire_rim 2", "tire defect detection", "defect detection" (tire-defect projects) | Tyre/rim annotation | tire_rim 2 CC BY 4.0, two others Public Domain per summary [S]; pages 403, contents unverified | Check content before use |
| Roboflow "undercarriageCropping" https://universe.roboflow.com/car-segmentation-k5akd/undercarriagecropping | Title suggests undercarriage cropping/segmentation | unknown; 403 [S] | Worth opening in a browser |
| Kaggle cracked/normal tyre set (mentioned in a 2025 paper, 240+60 images per class) | Tyre cracks | unknown [S] | Check Kaggle licence tag |
| Tyre sidewall defect set from a Symmetry 2024 paper (scratches/cuts/dents) | Sidewall defects | unknown; not a public release [S] | |
| CAD 100K (arXiv 2604.09023) https://arxiv.org/abs/2604.09023 | 100k+ images, 7 vehicle domains, car manufacturing anomaly detection | not checked | Manufacturing QA, not undercarriage [F abstract] |
| Mendeley "Dataset of Central Asia vehicle number plate with Annotations" https://data.mendeley.com/datasets/4c5z833m6m | Central Asian plates | not checked here | Real plates; privacy rules in the brief apply |

Undercarriage research (proprietary, not downloadable): ACV Auctions oil-leak models on its Virtual Lift images; University at Buffalo image-quality classifier on 1,100 labelled underbody images; a corrosion study on 369 inspected vehicles. [S]

Implication [I]: for any undercarriage claim we would have to collect and label our own data (and Pulsar says it needed thousands of dirty-car examples to separate mud from oil). That is a strong data moat for whoever does it, but impossible to build credibly in a hackathon without hardware and sites.

---

## 5. Implications for our product choice

Constraints from our hackathon brief (memory file): judged on technical complexity, innovation (replication test / moat), impact (real user, payer, evidence, 3-month pilot), no faked demos or invented metrics; the user already rejected photo damage-inspection ideas because procedures exist, losses are insured and UVeye-class vendors exist.

**Evidence for the full service-flow product**
- A Kazakh customer put a service-bay occupancy + LPR + ETD + queue system out to tender (10 applicants), so the need is being articulated locally; but the budget is small (about USD 3,300 if KZT) and the buyer is an unverifiable sole proprietor. [F/I]
- A US Toyota dealer deployed the same category (Skaivision) with reported (vendor) results; so there is a known paying customer type and a known KPI set. [F]
- No competitor found in Uzbekistan for it. [S]
- Technical bar is credible and testable: the TZ's own acceptance criteria (5-second status change, >=90% plate accuracy day/evening) are concrete targets we can measure on staged footage. PROMAT's tender gives a real-world LPR baseline (98% day / 87% night with ordinary IP cameras), which we can use to set honest expectations for night/rain plate reads.

**Evidence for focusing on entrance inspection**
- A local team already has a live pilot, a USD 2-4k hardware design, and press coverage, and says it is heading to Uzbekistan. A hackathon entry in this niche would face a better-documented local competitor and would need hardware plus roadworks, which we cannot demonstrate on a judge-supplied input. [F/I]
- No public undercarriage dataset exists; accuracy claims in the market (80-95%) are all unverified. Under the rules (no invented metrics, own test set), we could not defend an undercarriage accuracy number. [F/I]
- UVeye-class vendors already sell it to dealer groups (and the user's brief already treats inspection as saturated).

**Recommendation [I]**
1. Keep **service-flow analytics** as the core (entry plate to bay states to idle/parts-wait to handover, with customer Telegram notification). It matches the open Kazakh demand and a proven US case, and it is measurable on staged footage.
2. Treat the **entrance as the first event of the flow**, not a separate product: plate read, timestamp, a few standard-camera photos of the car on arrival as an evidence record (UVeye's own claimed value for dealers is fewer false damage claims). This borrows the useful part of "incoming inspection" without hardware. Do not claim undercarriage capability; if mentioned, present it as a roadmap item with a partner such as Pulsar.ai or UVeye.
3. Position the speed-bump idea honestly: say it exists locally (Pulsar.ai) and that we integrate rather than replicate; this also defuses a judge's "UVeye/Pulsar already do this" objection.
4. Reuse tender requirements as our test plan: bay state latency <5 s, plate accuracy at day/evening/night on Uzbek-format plates (synthetic or consented), reconnect on RTSP loss; report measured numbers and failure cases.
5. The still-open **car-wash tender (ONE AI, deadline 19.10.26)** is a nearby reference: same stack (existing IP cameras, plate-based visit routing, evidence clips). It could be a useful design reference, and a possible small side channel if the team wants an external customer signal.

**Open questions worth one phone call each, if time allows**
- Ask the dealer group whether they already have or have evaluated any entrance scanner (UVeye, Pulsar), and whether service cameras already cover bays (hardware is the cost driver).
- Ask Astana Hub / the tender contact directly about the VisionFlow outcome; the platform will not tell us.

---

## 6. What I could not verify

- Winner/executor of VisionFlow (not published anywhere I could reach).
- Currency of the 1.5M budget (inferred KZT).
- Identity and business of ИП Smart Tech.
- Whether the Pulsar.ai pilot is confirmed by Hyundai, and how its accuracy was measured.
- Prices for UVeye (secondary source only), Hikvision, Russian vendors; Uzbek market prices.
- Licences on several Roboflow/Kaggle datasets (Roboflow pages returned HTTP 403).
- ONE AI and LiTRO tender attachments (TZ files) were not downloaded because file downloads need explicit user permission.

---

## 7. Sources

Astana Hub (all [F] unless stated)
- VisionFlow tender (ru): https://astanahub.com/ru/tech_task/tz-platforma-avtomatizatsii-servisnykh-zon-visionflow-ai-mvp (also /en/, /kk/)
- Customer profile: https://astanahub.com/account/v2/user/720477/profile/activity/
- Tech-task listing (crawled pages 1-87): https://astanahub.com/ru/tech_task/?page=N
- Tech Tasks module description (25 Jul 2025): https://astanahub.com/en/article/vozmozhnosti-dlia-razrabotchikov-astana-hub-zapustil-ploshchadku-tekhnologicheskikh-zadach
- Car-wash tender (ONE AI): https://astanahub.com/ru/tech_task/intellektualnaia-sistema-avtomaticheskogo-ucheta-i-kontrolia-avtomoek
- PROMAT night LPR: https://astanahub.com/ru/tech_task/ustoichivoe-nochnoe-raspoznavanie-gosnomerov-rk-na-vezde-parkovki-ik-podsvetka-zasvetka-farami-raz
- LiTRO damage assessment: https://astanahub.com/ru/tech_task/razrabotka-intellektualnogo-servisa-dlia-avtomaticheskoi-otsenki-povrezhdenii-avtomobilia
- LiTRO STO module: https://astanahub.com/ru/tech_task/razrabotka-modulia-dlia-stantsii-tekhnicheskogo-osmotra-v-prilozhenii
- Agrostream: https://astanahub.com/ru/tech_task/razrabotka-sistemy-prognozirovaniia-tekhnicheskikh-neispravnostei-avtomobilei
- NF Labs Orion Pro + Hikvision: https://astanahub.com/ru/tech_task/orion-pro-hikvision

Pulsar.ai / SmartRail
- Digital Business, 2 Apr 2026: https://digitalbusiness.kz/2026-04-02/shazam-dlya-avtomobilya-kazahstanskiy-startap-nauchil-ii-nahodit-polomki-za-2-sekundi/
- Kursiv, 6 Apr 2026: https://kz.kursiv.media/2026-04-06/cntr-v-kazakhstane-sozdali-ii-skaner-dlya-poiska-utechek-i-korrozii-v-avto/
- Kapital.kz, 11 Jun 2026: https://kapital.kz/tehnology/149021/startap-iz-kazahstana-hochet-snizit-dtp-s-pomoshyu-ii-diagnostiki-avto.html
- Forbes.kz, 25 Aug 2026: https://forbes.kz/articles/kak-kazahstanskiy-startap-nauchil-iibystromu-diagnostirovaniyu-avtomobiley
- Instagram handle quoted by Digital Business: @pulsar.ai.thecar (not opened)

Vendors / analogues
- UVeye how it works: https://uveye.com/how-it-works/ ; dealerships: https://uveye.com/dealerships/ ; products: https://www.uveye.com/products/ ; GM programme: https://uveye.com/gm
- UVeye cost summary (secondary): https://www.thepricer.org/how-much-does-uveye-machine-cost/
- Toyota Tsusho investment in UVeye (2019) [S]: https://www.toyota-tsusho.com/english/press/detail/190619_004400.html ; https://coverager.com/uveye-raises-31m/
- Dahua UVSS reseller listings: https://networkcamerastore.com/products/dahua-dh-mv-vdf5020ce-00-fixed-under-vehicle-scan-system ; https://networkcamerastore.com/products/dahua-dh-mv-vdm5021e-00-portable-under-vehicle-surveillance-system
- Hikvision UVSS [S]: https://content.hikvision.com/hikvision-under-vehicle-surveillance-system
- Skaivision case study: https://skaivision.lovable.app/case-studies/fred-anderson-toyota-raleigh ; Reynolds partnership [S]: https://www.autoremarketing.com/?p=70219
- AUTOSCAN (Russia) [S]: https://www.sostav.ru/blogs/280924/98970

Other
- Kazakhstan technical inspection AI control [S]: https://digitalbusiness.kz/2026-03-20/u-kazahstantsev-bolshe-ne-poluchitsya-kupit-tehosmotr-iz-za-vnedreniya-ii-kontrolya/
- Tashkent parking tender / Tass Vision / IT Park [S]: https://www.spot.uz/ru/2025/02/12/parking/ ; https://www.it-park.uz/en/itpark/news/uzbekistan-strengthens-business-ties-in-the-it-sector

Datasets
- TyreNet: https://data.mendeley.com/datasets/32b5vfj6tc
- CAD 100K: https://arxiv.org/abs/2604.09023
- Central Asia plates: https://data.mendeley.com/datasets/4c5z833m6m
- Roboflow Universe: https://universe.roboflow.com/ai-proyect/car-damages-kaggle ; https://universe.roboflow.com/car-segmentation-k5akd/undercarriagecropping ; https://universe.roboflow.com/tai-lam/tire_rim-2 (all 403 to my fetch)
