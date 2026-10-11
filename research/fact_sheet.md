# Neutral fact sheet: CV track, Uzbekistan (as of 2026-10-10)

Facts only, gathered by our research agents. It deliberately contains **no idea rankings or recommendations**.
Tags: [F] read on a primary page, [S] from search summaries only, [I] inference. Numbers are as reported by the cited sources (in the detailed reports in this folder); conflicting figures are shown side by side.

---

## 1. The challenge (organisers' brief, key text)

- **Task:** build a working AI product that turns images or video into a decision or an action for the automotive business or for road safety. The user can be a dealer, a service centre, an insurer, a fleet owner, a driver or a city.
- **Track partners:** Toyota Royal Motors, BYD Royal Motors and Ducati.
- **Modes (name it on slide 1; judged identically):**
  - *Startup:* an independent product with its own customers and revenue.
  - *Partner product:* runs inside a partner company; name the unit that owns it and the process it changes.
- **Example directions (not limits):**
  - Vehicle inspection: damage and defect detection at handover, trade-in, service intake or an insurance claim; appraisal of a used vehicle from photos.
  - Dealer and service operations: service-bay and stock-yard analytics, plate recognition linked to the customer record, parts identification, pre-delivery inspection.
  - Driver and road safety: driver monitoring, motorcycle rider safety, dashcam analytics, road-condition mapping, "traffic events that continue your elimination task".
  - EV and infrastructure: charging-station occupancy and state, parking, fleet monitoring.
- **Deliverables:** (1) a prototype that runs end to end, live, **including on an input the jury gives you**; (2) a deck; (3) a repo with code, README and list of data sources with licences. Web version must be publicly deployed and testable by judges on laptops/phones.
- **Engineering constraint:** state where inference runs (edge, server, cloud), speed on your hardware, cost per camera or per inspection, and how quality holds at night, in rain, and on local vehicles and plates.
- **Not allowed:** faked demos, hard-coded answers, invented metrics; real personal data that is not public (customer records; faces/plates of people who did not consent); pre-final work presented as new; unattributed code/models. "A single prompt to a general-purpose model behind a UI is not forbidden, but it scores low on both technology criteria."
- **Judging (0-10 per criterion per judge):**
  - Technical complexity 25%: a system, not a call to someone else's model; architecture, data pipeline, quality measured on a test set you built, speed and cost per request, known failure cases.
  - Innovation and originality 25%: what is technologically new; "the replication test: what stops another team with the same LLM tools from rebuilding this in a weekend? Name your advantage: data, model, pipeline, integration or domain knowledge."
  - Impact and practicality 25%: is the problem real and is this the right solution; who is the user, who pays, what is your evidence; can it be piloted in three months within regulation and existing processes.
  - Presentation 15%: clear story, live demo + deployed website, honest numbers, direct answers, timing.
  - Business potential 10%: pilot budget, unit economics, payback period with assumptions.
  - Ties broken by the sum of the two technology criteria.

## 2. Team constraints

- Team is in Uzbekistan; Uzbekistan market is the priority.
- Mode preference: partner product slightly preferred; startup mode acceptable.
- **Representatives of the partner companies are expected to sit on the jury.**
- Organisers, partners and jury provide **no data**. The team **cannot film real footage at the moment**.
- Public datasets are acceptable even if non-commercial / research-only and even if they contain faces or plates. Using ideas or products that already exist abroad is acceptable if they don't exist in Uzbekistan.
- Compute: one GPU workstation (currently offline) plus cloud GPUs if needed.
- The team's elimination-round task is unknown to this sheet; do not assume reusable code.

## 3. The partners

### Royal Motors group (dealer group, not a distributor)
- Royal Motors is a **multi-brand dealer group** in Tashkent. Its role as Toyota and BYD dealer is confirmed on dealer/distributor sites [F]; the Ducati link is **inferred** from a shared address (Zargar st. 4) and phone prefix, not stated anywhere [I].
- A likely legal employer, "OLTIN TULPOR MOTOR", appears on hh.uz vacancies across Toyota, BYD (Namangan, Samarkand) and Hyundai sites [I].
- No sales volumes, staff counts, bay counts or DMS/CRM names found.

### Toyota
- Distributor: **Toyota Central Asia** (formerly Summit Motors Central Asia, renamed July 2026, "managed by Sumitomo Corporation") [F]. Royal Motors runs 2 of 5 Toyota points in Uzbekistan: **Toyota Tashkent City** (opened Aug 2023) and **Royal Motors Qushbegi** (opened 8 May 2026, "built to Toyota Motor Corporation standards", express-service zone with panoramic glass so customers can watch maintenance) [F].
- Models: Camry Hybrid, RAV4 Hybrid, Prado, Land Cruiser 300 / 300 HEV. 8-year hybrid battery warranty; free maintenance to 20,000 km; Kapital Bank loans and 0% instalments [F].
- 2025-26: Toyota Academy Central Asia opened Nov 2025 ("one standard of service quality at all dealers"); national technician skill contest; two free "10-point diagnostic" campaigns with service vouchers in 2026; Ziebart paint protection rolling out to all dealers (Jun 2026) [F].
- Distributor sales head (2023): "car sales are not the main thing, service is" [F].
- Not found: mobile app, online service booking, CRM, trade-in / certified-used programme in Uzbekistan (Kazakhstan has "Toyota Tested") [F/NF].
- Toyota Tsusho invested in UVeye (AI drive-through vehicle scanners) in 2019 [S]. Ravin AI's site describes a service-centre product using standard CCTV at the entrance gate built "in partnership with Toyota service centers" [F].

### BYD
- "BYD Royal Motors" = **BYD Chigatoy** (byd-rm.uz, Tashkent), one of ~30 official BYD dealers in Uzbekistan (15 in Tashkent city/region) [F]. Online service booking with slots 09:00-17:00 and a question "bought from official dealer?" [F].
- BYD Uzbekistan factory in Jizzakh: UzAuto 60% / BYD 40% JV; production from Jan 2024; ~20,200 cars in 2025 [F]. BYD share ~8.6-9.4% of sales in 2026 [F].
- **Grey imports:** BYD said official dealers will not give warranty service to grey-import cars and that origin of some cannot be verified; a reporter estimated about two-thirds of Song DM-i seen in Tashkent were unofficial imports [F].
- Royal Motors BYD vacancies: service consultants, diagnostician-electricians (ECU flashing), a mechanic doing "damage assessment", detailing master, parts specialist, Head of Service (Namangan), Service Trainer [F].
- BYD discussed building a fast-charging network with the ministry (July 2026, proposal only) [S].
- BYD Europe job ads list dealer KPIs: NPS, CCR, ticket resolution time, parts availability, workshop productivity, warranty quality, DMS/TIS rollout, audit readiness [F].

### Ducati
- "Ducati Tashkent City" (Zargar st. 4), calls itself the official importer; opened late 2024 / grand opening May 2025 [F]. Instagram ~13.1k followers, Telegram ~504 subscribers [F].
- Service Mon-Fri; Ducati Diagnosis System; 2-year warranty; service paused ~2 weeks in Jun-Jul 2025; sells ex-test-fleet bikes with 500-1,150 km; list prices fell 10-17% during 2025 [F].
- Test rides require category A licence, gear and 2+ years' experience [F]. Short riding season (mid-April opening) [F].
- Ducati globally: radar ARAS (Bosch), airbag vests, V2X demos, Safety Road Map; no public camera-vision programme found [F/NF].

## 4. Uzbekistan market facts

- Vehicle parc: ~4.9M individually owned vehicles (1 Apr 2026), of which ~4.55M passenger cars [F].
- New sales 2025: 427,097 incl. light commercial; ~40.7k EVs [F].
- EVs: 109.2k registered (1 Apr 2026), 122.7k (1 Jul 2026), ~72-73% in Tashkent [F]; EVs ~71% of Jan-Jul 2026 passenger-car imports [F].
- Official dealerships: >190 (1 Jan 2024), 88 of them Chevrolet [F]. Number of independent service stations: not found. Car-wash legal entities: 3,122 (1 Feb 2026) [F].
- Used cars: 810,000 transactions in 2024 (-33%) [F]; ~50% of 2023 car sales listed on OLX/Avtoelon [F]; Avtoelon launched an "official dealers" section Nov 2025 [F]. Petrol-car registration fee rose 68x (to 5.4M UZS) in April 2024, so many owners sell by power of attorney without re-registration [F]. Used-EV prices fell 2024-mid-2025 [F].
- Imports: the Piskent certification-lab testing of imported cars (Nov 2024, 100-120 cars/day, long queues) **ended on 2 Sep 2025** per the Technical Regulation Agency (kun.uz) [F, found by the independent check]; do not cite the queue as current.
- President (16 Sep 2025): servicing of foreign brands "insufficiently developed"; ordered regional service centres and technician training [F]. Competition Committee (Oct 2025): 350+ car-service complaints in 2025, showrooms without service centres, warranty mismatches [F].
- EV policy: subsidised EV loans (<=16%) for 15k buyers from 1 Sep 2026; drivers pay 300 UZS/kWh from 1 May 2026; EV taxi benefits from 1 Mar 2026; recycling-fee exemption for locally built EVs to 2030 [F].
- Charging: 1,399 stations (1 May 2025), target 4,000 by end-2026; operators Tokbor (~1,000), Megawatt, Spectre, Quwatt [F]. March 2026: 101 Tashkent stations disconnected for missing documents, 62 without grid contracts [F]. April 2025: drivers leaving cars on chargers after charging; idle fees (~500 UZS/min) called illegal by the consumer committee; petrol cars blocking charging bays [F].
- Plates: private `01 A 123 BC`, legal entity `01 123 ABC`; EV plates have a **green region-code field** (since Nov 2021, hybrids since 2023); square motorcycle plates; taxi series TTA-TTG [F].
- Insurance: OSAGO reformed 1 Jan 2026 (limit 40 -> 80M UZS); the regulator said ~2,700 complaints (>50% of all) concerned payout delays and low limits [S/F]; EV KASKO products exist (EUROASIA, Uzbekinvest "Green KASKO") [F]; state app "Sug'urta Servis" for accident notices; no AI photo inspection by any Uzbek insurer found [F/NF].
- Road safety: 2,188 deaths / 8,901 injured in 2025 (traffic police), vs 1,854 deaths (Kun.uz) - unreconciled [F]; WHO 2021 estimate 3,155 deaths [F]. Tashkent had 420 crashes in H1 2026, highest of any region [F]. Camera enforcement widened (law of 4 Feb 2026: seat belt, phone use, small speeding, stopping/parking); **cameras issue penalty points since 1 Apr 2026** (12 points -> vehicle to impound lot) [F]. Tashkent "Safe City": 42,750 intelligent cameras, Huawei-led, plans for four new AI systems by end-2026 [F/S].
- Two-wheelers: moped/scooter registration mandatory from 1 Oct 2025; A1 licence category added Mar 2026; courier raids logged 23,785 violations (Oct 2026); 563 moped/scooter crashes and 89 deaths in 9 months of 2026 [F]. No motorcyclist death statistics published [NF]. Pothole complaints and a city "photo the potholes" campaign in Tashkent [F].
- Legal: personal-data law ZRU-547 amended 26 Mar 2026 - only biometric, genetic and telecom-subscriber data must be stored in Uzbekistan; biometrics need consent; employers must inform staff about data collection [F].

## 5. Customer reviews of Tashkent dealer service (read Oct 2026)

- ~490 reviews read across 32 dealer/service listings (Yandex Maps, 2GIS; Google ratings only). Mostly non-5-star reviews were read, so counts show the *distribution* of complaints, not prevalence [F].
- Headline ratings: Yandex 4.7-5.0 almost everywhere; Google 3.7-4.6 for the same BYD dealers; 2GIS 1.2-2.8 for some independent networks [F].
- Yandex "waiting time" aspect, share positive: BYD Yashnobod 38% (n=17), Kia Yakkasaroy 52% (34), BYD Sergeli 58% (29), Chery Rohat 69%, BYD Chigatoy 73% (11), BYD Yunusabad 74%, BYD Oloy 75%, BYD Megawatt 89-90% [F].
- Complaint counts by topic (all reviews read; a review can count twice): price/upsell 55, repair quality/repeat visits 46, time in bay / total time 31, status updates / phone / callbacks 29, reception wait 23, parts waits 21, handover problems 14, booking 8 [F].
- Recurring themes: promised short times turning into hours without updates; "leave the car, we'll call" without a call; parts waits of weeks to months for warranty work; paid line items not performed; customers barred from the repair area at some dealers; no shared customer database across BYD branches ("not in our base") [F].
- Some customers say they check dealer bay cameras themselves: e.g. "on the cameras nobody is working on the cars, they stand for 40 minutes" (Chery), "watched on the cameras: [paid items] were not done" (BYD) [F].
- Royal Motors sites: Toyota Tashkent City, Qushbegi and Ducati - no waiting/status complaints found; BYD Chigatoy - a few waiting complaints and a 1 Oct 2026 review saying 4 of 5 requested items were not done [F].

## 6. Existing products and players (non-exhaustive)

- Service-lane / bay analytics from CCTV: SKAIVISION (US; a Toyota dealer customer since 2024), Leverege PitCrew (tyre/oil-change chains), Autoflow Bayflow, EyeQ (ANPR cycle time), Detect Auto [F].
- Drive-through / drive-over inspection: UVeye (body, tyres, underbody; ~$5-7k per lane per month reported [S]), ProovStation (France; court restructuring Jan 2026 over hardware financing [F]), DeGould, **Pulsar.ai SmartRail** (Almaty; speed-bump underbody scanner piloted at a Hyundai centre since Mar 2026; names Uzbekistan as an expansion target) [F].
- Phone photo inspection: Ravin AI, Tchek, Monk (ACV), Inspektlabs, WeProov; tyres: Anyline [F].
- CIS: Ivideon + AvtoVAZ (200+ LADA dealers on cloud video analytics) [F]. Uzbekistan: TASS Vision (retail on-device CV), VEON AdTech video analytics; **no dealer/service-centre or auto CV startup found** [F/NF].
- Astana Hub (Kazakhstan) tech tasks: "VisionFlow AI" service-bay automation MVP (~1.5M, likely KZT; winner unpublished); "ONE AI" car-wash visit-to-payment matching on existing cameras (open until 19 Oct 2026); night ANPR task quoting 98% day / 87% night accuracy on Kazakh plates with ordinary IP cameras; photo damage assessment for roadside assistance [F].

## 7. Public data availability (summary of our inventories)

- **Strong:** exterior car damage (CarDD 4k, VehiDE 14k, several Roboflow sets, SYNDCAR damage+parts); car-parts segmentation (3.8k-9k); tyres (TyreCV 508 stereo pairs with tread depth in mm; TyreAI 4k DOT-code OCR images; tyre condition sets); vehicle detection incl. night/rain (UA-DETRAC, BDD100K, DAWN, ExDark, MIO-TCD, NDISPark); plates (CCPD, RodoSol-ALPR, UFPR-ALPR, AUTO.RIA crops incl. Kazakhstan 9.9k and Kyrgyzstan 31.9k; Roboflow Uzbek plate detection 3k); parking occupancy (PKLot, CNRPark+EXT incl. rain); vehicle tracking/re-ID (CityFlow, VeRi); real video of car-service actions (Kinetics-700: changing wheel 971 clips, changing oil 913, checking tires 788; CrossTask with step timestamps: Jack Up a Car 89, Change a Tire 99, Add Oil 137).
- **Thin:** undercarriage (rust on phone photos 2,055; real car oil leaks 593; drive-over composites without defect labels ~2,400; pit-view truck parts); car-wash scenes (a few hundred to 2.4k images); fixed-CCTV footage of service bays (a handful of public timelapse clips).
- **None found:** labelled service-bay CCTV with states; paired before/after images of the same car; EV battery-pack underbody damage.
- Not inventoried by us (check yourself if relevant): driver-monitoring / distraction datasets, road-damage datasets (e.g. RDD2022), helmet datasets (e.g. AI City Challenge helmet track), dashcam event datasets, charger datasets.

## 8. Engineering anchors

- Edge hardware in Tashkent: PC with RTX 4060 Ti ~$925-1,345 (OLX, Oct 2026); used RTX 4060 Ti card ~$390; Jetson Orin Nano Super kit $399 after July 2026 price rise; Hikvision IP camera from ~$32; 16-channel NVR ~$247 [F/S].
- Russian cloud video analytics ~$2.4-10 per camera per month [F].
- Exchange rate used: 11,897.90 UZS per USD (10 Oct 2026) [F].
