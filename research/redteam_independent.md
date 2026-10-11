# Independent red-team strategy: CV track final, Uzbekistan (2026-10-10)

Prepared by an isolated strategist. Inputs: `research/fact_sheet.md` only (no other team file was opened) plus about 40 web lookups of my own (web searches, page fetches and two GitHub API calls).

**Tags.** [F] = read on the page itself. [S] = search-result summary only. [I] = my inference or assumption. **FS** = taken from the fact sheet, with its own tag kept. "Not found" = I searched and could not verify it.

---

## 0. Bottom line

- **Winner: "Qabul", an AI vehicle reception and handover act** (partner product).
  - **Owner:** Royal Motors Aftersales, Service Reception, at Toyota Tashkent City / Royal Motors Qushbegi, BYD Chigatoy and Ducati Tashkent City.
  - **Process it changes:** vehicle reception (the acceptance act, *акт приёма-передачи*), active delivery, and after-hours drop-off.
  - **How it works:** a 60-second phone walkaround plus one dashboard photo becomes a signed digital act. The act holds the plate, mileage, warning lights and a panel-level damage map. At pick-up, a second capture runs a "what changed" check and the system decides *release / hold for review*.
  - **Score:** weighted estimate **6.6/10** (medium confidence). It stays at ≥6.15 even if partners say disputes are rare and reception is already digital.
- **Runner-up: "Show-me inspection", an evidence-backed multipoint inspection with customer approval in Telegram.** Scores **6.2**, rising to about 6.4 if partners confirm that customers approve few recommendations.
- **Sleeper to test with partners: warranty-claim photo QA.** Scores 5.8 today and about 6.4 if partners confirm that the OEM/distributor rejects claims over photos.
- **Service-bay CCTV analytics ranks 9th (5.6).** The pain is real, but there is no public labelled bay footage and the live demo on jury input is weak. It reaches about 6.1 only if partners hand over CCTV.
- **Main risk of the winner:** it sits in the most crowded lane. Damage detection is the brief's first example, and 20 teams per track reach the final [F]. The pitch must therefore open on the *act + handover diff + dashboard* system and on a *measured* cut in false alarms, not on "we detect scratches".

---

## 1. New facts found that the fact sheet does not have, or contradicts

| # | Finding | Tag | Implication |
|---|---|---|---|
| 1 | The event is **WIUT Hackathon 2026**. The CV-track final runs on site **10–11 Oct 2026**, with the top 20 teams per track and a separate panel per track. Weights are identical: Innovation 25, Technical 25, Impact 25, Presentation 15, Business 10. | [F] hackathon.wiut.uz | The build window is ~1.5 days. Expect several damage-detection teams, so innovation has to come from the system and process. |
| 2 | **Piskent import testing ended on 2 Sep 2025** (Technical Regulation Agency). The stated reason is growth in official dealers: 21 foreign companies, >100 models type-approved, versus 3 centres / 15 models before 2024. | [F] kun.uz | The FS §4 line "certification lab… queue fills in seconds" is **outdated**. Import/certification ideas are dead, and the official dealer network, and with it service competition, is growing. |
| 3 | The acceptance act is what real disputes turn on. At BYD Megawatt (Tashkent), the customer-service head said the client *"left the car overnight with security without an acceptance act, without recording any claims"*. The same car had broken down 3 times in 10 days, and parts were being "shipped from China". | [F] kun.uz 13 Mar 2026 | This is direct local evidence for an act-centred product and for night drop-offs. |
| 4 | **Two-wheelers (spot.uz, 6 Oct 2026):**<br>• SBDD: 4,312 moped/scooter violations in 9 months of 2026, ~20× all of 2025.<br>• Yandex Uzbekistan VP: 23,785 violations and ~4.7 bn UZS of fines. Only 8% concerned documents; 92% were missing reflectors/headlights, tinting or carried passengers.<br>• SBDD: 563 crashes, 89 killed, 488 injured.<br>• No platform measures are mentioned. | [F] | This is the strongest fresh road-safety evidence I found. It supports a courier-compliance idea (§4.5). |
| 5 | **Crash causes, Q1 2026:** 1,102 crashes, 346 fatal. Speeding 32.9%, red light 22.4%, failure to yield to pedestrians 13.6%, falling asleep 4.2%. Tyre failure is not listed. | [S] kun.uz | A tyre-safety pitch has no local evidence. |
| 6 | **3DRealCar** has 2,500 real cars with ~200 360° RGB-D views each, captured in **reflective, standard and dark** lighting, plus 3D car-parsing maps. Its README says it is "allowed commercial usage for free… license is Apache-2.0". | [F] GitHub README; size/lighting [S] arXiv | This is the best real multi-view walkaround data for the winner: false alarms on clean cars, reflections, night. |
| 7 | **TRODO** has 2,389 odometer images with mileage ground truth, under CC BY 4.0. **CarDD** has 4,000 images, 9,000+ instances and 6 categories, but download needs a signed licence form. **Uzbek_Plates** (Roboflow) has 8,434 images under CC BY 4.0. Dashboard warning-light sets on Roboflow are CC BY 4.0, one of them with 31 classes. **VehiDE** has 13,945 images and 8 classes, licence not found. **RDD2022** has 47,420 images; its CC BY-SA 4.0 licence is stated only by Dataset Ninja. **TyreCV** (listed in FS) was not found by my search. | TRODO, CarDD [F]; the rest [S] | Data plan in §4.1. The CarDD form may not come back in time, so the team needs fallbacks. |
| 8 | **Price anchors:** DAMAGE iD charges $1.25 per vehicle. Ravin's AWS Marketplace package costs $84,000/yr for up to 20,000 inspections (~$4.20 each). Ravin markets "Autoscan… transforms CCTV cameras into vehicle scanners" and shows a Toyota logo among its clients. | prices [S]; Ravin page [F] | Unit-economics anchors. A Toyota-network precedent exists abroad. |
| 9 | **Photo/video and approvals:** with photo/video documentation, ~51% of customers approve recommended work versus ~24% without (J.D. Power, reported second-hand). Vendor anecdotes such as a Toyota dealer going from 32% to 78% are unverified. | [S] | Supports the runner-up and the upsell part of the winner. Do not cite as primary. |
| 10 | **Toyota dealer practice:** "Active Reception" (a walkaround with a technician) and "active delivery". Job postings ask advisors to do the walkaround and take an accurate odometer reading. No official Toyota standard document was found. | [S] | Good process fit for the winner. Ask partners whether Toyota Central Asia mandates it. |
| 11 | **Pulsar AI (Kazakhstan):** the startup launched in Feb 2026, and its underbody scanner has been piloted at Hyundai Premium Center Almaty since Mar 2026. It claims 85–90% accuracy. | [S] | Dealers in the region adopt CV fast. Pulsar is also a potential entrant (FS says it names Uzbekistan as a target). |
| 12 | **No Uzbek dealer or insurer was found using AI photo inspection** (searches in Uzbek and Russian). Uzbekinvest uses AI in underwriting with GlowByte, but on data, not images. | [S] | "Exists abroad, not in Uzbekistan" holds. |
| 13 | **Insurance:** 2025 premiums were 13.46 trn UZS (+38%) and payouts ~3 trn UZS. KASKO share not found. *Sug'urta Servis* is the official app for accident notices without police and is owned by the OSAGO payment guarantee fund. | [S] | Context for the startup alternative in §4.4. |
| 14 | **Other context:**<br>• Taxi: >450k self-employed drivers; white taxi market ~$800M in 2025.<br>• Tokbor holds >40% of EV charging and said EV owners complain about stations being blocked.<br>• The Tashkent hokim ordered all potholes on main roads fixed within 10 days and asked residents to report them (date unclear). | [S] | Context for the long list. |

---

## 2. Long list (35 ideas, all user groups, both modes)

P = partner product (Royal Motors Toyota / BYD / Ducati). S = startup.

| # | Idea | User → payer | Mode |
|---|---|---|---|
| 01 | **Reception & handover act**: phone walkaround + dashboard photo → act (plate, mileage, warnings, damage map), handover diff, release/hold | Service advisor, customer → dealer aftersales | P |
| 02 | **Service-bay status & productivity** from existing CCTV: bay states, live customer status, idle time | Service manager, customer → dealer | P |
| 03 | **Proof-of-work**: action recognition on bay video ties each repair-order (RO) line to a video clip | Customer, manager → dealer | P |
| 04 | **Evidence-based multipoint inspection (MPI)**: tyres, glass, body measured from photos; priced red/amber/green items; approval in Telegram | Technician, advisor, customer → dealer | P |
| 05 | **Tyre safety & age check** (DOT date OCR, tread condition, sidewall) for drivers and fleets | Drivers, fleets → tyre sellers, dealers | S |
| 06 | **Instant body-repair estimate** from photos (body-shop lead generation, insurer approvals) | Body shop → dealer | P |
| 07 | **Trade-in / certified-used appraisal** (condition + local price) | Used-car manager → dealer | P |
| 08 | **Gate ANPR** → customer record, arrival timer, greeting | Reception → dealer | P |
| 09 | **Pre-delivery inspection (PDI)** of new cars (transport damage, accessories, films) | PDI tech → dealer | P |
| 10 | **Stock-yard audit** from phone/drone video (count, VIN/plate, damage) | Logistics → dealer | P |
| 11 | **Showroom footfall** & salesperson-response analytics | Sales manager → dealer | P |
| 12 | **Paint-condition scan** → Ziebart/detailing upsell | Advisor → dealer | P |
| 13 | **Warranty-claim photo QA** (VIN, odometer, part label, defect visible, completeness) | Warranty admin → dealer | P |
| 14 | **Grey-import / warranty-eligibility check** at intake | Reception → dealer, BYD | P |
| 15 | **Dashboard warning-light triage bot** → booking and parts pre-order | Owner → dealer | P |
| 16 | **Ducati riding coach** (lean angle and body position from video) | Riders → Ducati dealer | P |
| 17 | **Motorcycle pre-season check** (chain, tyres) from photos | Riders → Ducati dealer | P |
| 18 | **Remote KASKO self-inspection** with anti-fraud | Underwriting → insurer | S |
| 19 | **OSAGO claim photo triage** for Sug'urta Servis accident notices | Claims → insurers, guarantee fund | S |
| 20 | **Dashcam accident evidence pack** (collision moment, plates, signal state, trajectories) | Drivers, claims → insurers | S |
| 21 | **Ride-hailing vehicle photo-control** automation (class, cleanliness, damage, plate match) | Aggregator → aggregator | S |
| 22 | **Courier two-wheeler pre-shift compliance** (helmet, headlight, reflectors, plate) | Couriers → delivery platforms | S |
| 23 | **Driver monitoring** (phone, belt, fatigue) against the camera penalty-point system | Taxi/bus drivers → fleets, aggregators | S |
| 24 | **Road-damage mapping** from taxi/bus dashcams | Road agency → khokimiyat | S |
| 25 | **Two-wheeler helmet/violation detection** on city cameras | Police → city | S |
| 26 | **Pedestrian-yield violation detection** at crossings | Police → city | S |
| 27 | **EV charger bay blocking/occupancy** (green plate field marks EV vs combustion car) | Charge-point operator (CPO) → CPO | S |
| 28 | **EV underbody / battery-scrape check** at the service entrance | Service → dealer, OEM | P |
| 29 | **Car-wash visit-to-payment reconciliation** (ANPR count vs POS) | Owner → car wash | S |
| 30 | **Used-car listing verification** (Avtoelon/OLX) | Marketplace → marketplace | S |
| 31 | **Bank auto-loan collateral** remote inspection | Bank → bank | S |
| 32 | **Counterfeit-parts detection** from packaging photos | Parts department → distributor | P |
| 33 | **Repainted-panel ("kraska") detection** for used cars | Buyers → marketplaces, inspectors | S |
| 34 | **Import certification inspection** automation (Piskent) | Lab → state | S |
| 35 | **Parts identification** from photo at the parts counter | Parts counter → dealer | P |

---

## 3. Screening

### 3.1 Criteria and rubric mapping

Each idea is scored 1–10 on:

- **a.** Evidence that the pain is real in Uzbekistan
- **b.** Reaction of partner judges
- **c.** Whether CV is the right tool, as opposed to a form, button, sensor or process change
- **d.** Real-data provability
- **e.** Live demo on jury input
- **f.** Hard to copy (data, pipeline, integration or domain knowledge, not effort)
- **g.** 3-month pilot feasibility and economics
- **h.** Buildable in 1–2 days with a test-set metric
- **N.** Novelty versus other finalist teams and the Uzbek market

These map to the rubric through transparent heuristics, so expect about ±0.5 of noise:

- Technical = 0.3d + 0.3h + 0.2c + 0.2f
- Innovation = 0.6f + 0.4N
- Impact = 0.35a + 0.25b + 0.2c + 0.2g
- Presentation = 0.75e + 0.25·mean(a, b)
- Business = 0.6g + 0.4b
- **Weighted** = 0.25 Technical + 0.25 Innovation + 0.25 Impact + 0.15 Presentation + 0.10 Business

Conf = confidence in the weighted estimate: H high, M medium, L low.

### 3.2 Scores

| Rank | # | Idea | Mode | a | b | c | d | e | f | g | h | N | Tech | Innov | Impact | Pres | Bus | **Weighted** | Conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 01 | Reception & handover act | P | 6 | 7 | 7 | 8 | 9 | 5 | 8 | 7 | 4 | 6.9 | 4.6 | 6.8 | 8.4 | 7.6 | **6.60** | M |
| 2 | 04 | Evidence-based MPI + Telegram approval | P | 6 | 7 | 5 | 6 | 8 | 5 | 7 | 6 | 6 | 5.6 | 5.4 | 6.2 | 7.6 | 7.0 | **6.16** | M |
| 3 | 13 | Warranty-claim photo QA | P | 3 | 6 | 6 | 6 | 7 | 5 | 6 | 7 | 7 | 6.1 | 5.8 | 5.0 | 6.4 | 6.0 | **5.77** | L |
| 4 | 05 | Tyre safety & age check | S | 3 | 5 | 7 | 6 | 8 | 5 | 6 | 7 | 6 | 6.3 | 5.4 | 4.9 | 7.0 | 5.6 | **5.76** | M |
| 5 | 09 | PDI / new-car delivery inspection | P | 4 | 6 | 7 | 6 | 8 | 4 | 7 | 7 | 4 | 6.1 | 4.0 | 5.7 | 7.2 | 6.6 | **5.70** | M |
| 6 | 07 | Trade-in / certified-used appraisal | P | 5 | 6 | 5 | 5 | 8 | 6 | 5 | 5 | 5 | 5.2 | 5.6 | 5.2 | 7.4 | 5.4 | **5.66** | L |
| 7 | 24 | Road-damage mapping from dashcams | S | 7 | 3 | 8 | 8 | 8 | 4 | 4 | 8 | 4 | 7.2 | 4.0 | 5.6 | 7.2 | 3.6 | **5.65** | M |
| 8 | 20 | Dashcam accident evidence pack | S | 5 | 4 | 7 | 6 | 6 | 7 | 4 | 4 | 6 | 5.8 | 6.6 | 5.0 | 5.6 | 4.0 | **5.58** | L |
| 9 | 02 | Service-bay status & productivity (CCTV) | P | 7 | 6 | 6 | 3 | 3 | 6 | 6 | 5 | 7 | 4.8 | 6.4 | 6.4 | 3.9 | 6.0 | **5.57** | M |
| 10 | 06 | Instant body-repair estimate | P | 5 | 6 | 7 | 5 | 7 | 5 | 6 | 5 | 4 | 5.4 | 4.6 | 5.9 | 6.6 | 6.0 | **5.56** | M |
| 11 | 27 | EV charger bay blocking (green plate field) | S | 5 | 4 | 6 | 6 | 6 | 5 | 5 | 7 | 7 | 6.1 | 5.8 | 5.0 | 5.6 | 4.6 | **5.52** | M |
| 12 | 18 | Remote KASKO self-inspection | S | 4 | 4 | 7 | 7 | 8 | 5 | 5 | 7 | 4 | 6.6 | 4.6 | 4.8 | 7.0 | 4.6 | **5.51** | M |
| 13 | 12 | Paint-condition scan → Ziebart upsell | P | 4 | 6 | 6 | 4 | 8 | 4 | 6 | 6 | 6 | 5.0 | 4.8 | 5.3 | 7.2 | 6.0 | **5.46** | L |
| 14 | 03 | Proof-of-work (action recognition) | P | 6 | 5 | 5 | 5 | 4 | 6 | 5 | 4 | 8 | 4.9 | 6.8 | 5.3 | 4.4 | 5.0 | **5.42** | L |
| 15 | 22 | Courier two-wheeler compliance | S | 8 | 3 | 6 | 4 | 6 | 4 | 6 | 6 | 7 | 5.0 | 5.2 | 6.0 | 5.9 | 4.8 | **5.40** | M |
| 16 | 15 | Dashboard warning-light triage bot | P | 3 | 5 | 6 | 6 | 9 | 3 | 6 | 8 | 5 | 6.0 | 3.8 | 4.7 | 7.8 | 5.6 | **5.35** | M |
| 17 | 08 | Gate ANPR → customer record | P | 5 | 5 | 6 | 7 | 7 | 3 | 7 | 8 | 3 | 6.3 | 3.0 | 5.6 | 6.5 | 6.2 | **5.32** | M |
| 18 | 16 | Ducati riding coach | P | 2 | 5 | 8 | 4 | 5 | 6 | 4 | 5 | 8 | 5.5 | 6.8 | 4.3 | 4.6 | 4.4 | **5.30** | L |
| 19 | 19 | OSAGO claim photo triage | S | 6 | 4 | 6 | 6 | 6 | 5 | 4 | 6 | 5 | 5.8 | 5.0 | 5.1 | 5.8 | 4.0 | **5.24** | L |
| 20 | 28 | EV underbody / battery-scrape check | P | 4 | 6 | 8 | 2 | 2 | 7 | 4 | 3 | 8 | 4.5 | 7.4 | 5.3 | 2.8 | 4.8 | **5.19** | L |
| 21 | 21 | Ride-hailing photo-control | S | 4 | 3 | 7 | 6 | 8 | 4 | 4 | 7 | 5 | 6.1 | 4.4 | 4.3 | 6.9 | 3.6 | **5.10** | L |
| 22 | 31 | Bank collateral inspection | S | 3 | 3 | 7 | 7 | 8 | 4 | 4 | 7 | 5 | 6.4 | 4.4 | 4.0 | 6.8 | 3.6 | **5.07** | L |
| 23 | 29 | Car-wash reconciliation | S | 4 | 2 | 7 | 6 | 6 | 3 | 7 | 8 | 6 | 6.2 | 4.2 | 4.7 | 5.2 | 5.0 | **5.06** | M |
| 24 | 11 | Showroom analytics | P | 2 | 6 | 7 | 6 | 6 | 3 | 6 | 7 | 5 | 5.9 | 3.8 | 4.8 | 5.5 | 6.0 | **5.05** | L |
| 25 | 30 | Used-car listing verification | S | 4 | 3 | 6 | 6 | 8 | 4 | 4 | 7 | 5 | 5.9 | 4.4 | 4.2 | 6.9 | 3.6 | **5.00** | L |
| 26 | 25 | Two-wheeler helmet detection (city cameras) | S | 7 | 3 | 8 | 6 | 7 | 3 | 3 | 7 | 4 | 6.1 | 3.4 | 5.4 | 6.5 | 3.0 | **5.00** | M |
| 27 | 33 | Repainted-panel detection | S | 5 | 5 | 4 | 1 | 5 | 7 | 4 | 2 | 8 | 3.1 | 7.4 | 4.6 | 5.0 | 4.4 | **4.97** | L |
| 28 | 26 | Pedestrian-yield violations | S | 7 | 2 | 8 | 5 | 6 | 4 | 3 | 5 | 6 | 5.4 | 4.8 | 5.2 | 5.6 | 2.6 | **4.94** | L |
| 29 | 23 | Driver monitoring vs penalty points | S | 5 | 4 | 7 | 6 | 8 | 3 | 4 | 7 | 3 | 5.9 | 3.0 | 5.0 | 7.1 | 4.0 | **4.93** | M |
| 30 | 14 | Grey-import eligibility check | P | 7 | 5 | 2 | 4 | 6 | 3 | 5 | 6 | 7 | 4.0 | 4.6 | 5.1 | 6.0 | 5.0 | **4.82** | M |
| 31 | 32 | Counterfeit-parts detection | P | 3 | 5 | 6 | 2 | 4 | 6 | 4 | 3 | 8 | 3.9 | 6.8 | 4.3 | 4.0 | 4.4 | **4.79** | L |
| 32 | 35 | Parts identification | P | 3 | 5 | 6 | 3 | 5 | 5 | 5 | 4 | 6 | 4.3 | 5.4 | 4.5 | 4.8 | 5.0 | **4.76** | L |
| 33 | 17 | Motorcycle pre-season check | P | 3 | 5 | 6 | 3 | 6 | 4 | 5 | 5 | 6 | 4.4 | 4.8 | 4.5 | 5.5 | 5.0 | **4.75** | L |
| 34 | 10 | Stock-yard audit | P | 2 | 4 | 6 | 5 | 6 | 4 | 5 | 6 | 6 | 5.3 | 4.8 | 3.9 | 5.2 | 4.6 | **4.75** | L |
| 35 | 34 | Import certification (Piskent) | S | 1 | 2 | 5 | 4 | 5 | 4 | 1 | 5 | 6 | 4.5 | 4.8 | 2.1 | 4.1 | 1.4 | **3.60** | H |

### 3.3 Sensitivity (same formulas)

| Scenario | Weighted |
|---|---|
| Winner, base case | **6.60** |
| Winner, if partners say reception is already digital and disputes are rare (a 4, b 5) | 6.15 |
| Winner, if judges see it as pure commodity damage detection (f 4, N 3) | 6.30 |
| Warranty QA, if partners confirm photo-related claim rejections (a 7, b 8) | 6.44 |
| MPI, if partners confirm low approval rates (a 7, b 8) | 6.38 |
| Bay analytics, if partners supply CCTV before the demo (d 5, e 6) | 6.06 |

### 3.4 Short justifications (key drivers per idea)

- **01 Reception & handover act.**
  - a6: complaints cover handover 14, reception wait 23, price/upsell 55 (FS); the BYD Megawatt act dispute is on record [F]. No Royal Motors-specific handover complaints were found (FS).
  - b7: reception and active delivery are the partner's own step [S]. Ravin-type tools are used in Toyota's world abroad (FS, [F]).
  - c7: detection, reading and diff are visual tasks. A tablet checklist covers only the paperwork half.
  - d8: CarDD, VehiDE, 3DRealCar, TRODO and Uzbek plates are all real data; the diff is synthetic only.
  - e9: works on any car and any phone, and on street video.
  - f5: the components are commodity. The moat is the multi-view pipeline, the local data flywheel and the process integration.
  - g8: needs only phones and one edge PC.
  - h7: four models plus glue is tight in 1.5 days.
- **04 MPI.**
  - a6: the top complaint is price/upsell, 55 (FS). Toyota ran two free 10-point campaigns in 2026 (FS). J.D. Power gap [S].
  - c5: technicians already have gauges, so CV adds evidence and standardisation rather than a missing measurement.
  - d6: tread-class sets are CC BY 4.0 [S] and DOT OCR data exists (FS). Depth-in-mm ground truth is unverified.
  - e8: the jury can photograph tyres and glass.
- **13 Warranty QA.**
  - a3: no public Uzbek evidence. "Warranty quality" is a BYD Europe dealer KPI (FS).
  - d6: TRODO, VIN and part-label OCR.
  - Mostly OCR. It is a sleeper: ask partners.
- **05 Tyre check.** a3: tyres are absent from the Uzbek crash-cause list [S]. Good demo. DOT OCR on black-on-black embossing is hard.
- **09 PDI.** a4: no transport-damage evidence found. It is effectively a module of 01.
- **07 Trade-in.**
  - a5: 810k used transactions and no Toyota certified-used programme in Uzbekistan (FS). Royal Motors trade-in not found.
  - c5: price is a non-CV model. Hidden repaint is invisible in photos.
  - d5: listing data has no licence.
- **24 Road damage.**
  - a7: pothole orders and complaints [S]. d8: RDD2022 [S].
  - b3: dealers are indifferent.
  - g4: payer is the city, with slow procurement.
  - f4/N4: crowded and well known.
- **20 Dashcam accident pack.** f7: hard to build. h4: too big for 1.5 days. Likely to overlap with "traffic events" carried over from the elimination task [I].
- **02 Bay analytics.**
  - a7: time, status and reception wait add up to 83 complaint mentions (FS).
  - b6: productivity is a KPI, but Royal Motors' own sites show no waiting complaints, and Qushbegi already has glass walls (FS).
  - c6: DMS job clocks and advisor status messages are non-CV fixes.
  - d3: no labelled bay CCTV (FS). e3: the jury cannot film bay CCTV.
- **06 Body estimate.** Royal Motors BYD hires a "damage assessment" mechanic (FS). Repair-cost labels would be synthetic, which risks an invented metric.
- **27 EV charger blocking.**
  - a5: Tokbor complaints [S]; 101 stations disconnected (FS).
  - c6: OCPP (the charger protocol) already reports charging state. CV is only needed for combustion cars blocking a bay, and idle fees were called illegal (FS).
  - Nice local touch: the green plate field marks EVs (FS).
- **18 KASKO inspection.** a4: KASKO volume unknown. b4: partners are lukewarm. The tech is the same as 01.
- **12 Paint scan.** Ziebart is rolling out (FS), but there is no complaint data and swirl/chip data is thin.
- **03 Proof-of-work.** Customers watch bay cameras and paid items were not done (FS). The idea implies technicians cheat. Mandatory step photos are a non-CV fix. YouTube-to-CCTV domain gap.
- **22 Courier compliance.**
  - a8: freshest evidence [F].
  - b3: dealers are indifferent.
  - d4: no moped reflector or plate data. Square moped plates have been mandatory only since Oct 2025 (FS).
- **15 Warning-light bot.** f3: a general vision-language model (VLM) reads warning lights well, so it fails the replication test.
- **08 Gate ANPR.** f3: ANPR is commodity and built into IP cameras.
- **16 Ducati coach.** a2: tiny market (Telegram ~504 subscribers; short season, FS).
- **19 OSAGO triage.**
  - a6: ~2,700 complaints about payout delays (FS); the Sug'urta Servis app exists [S].
  - g4: payers are a state fund and insurers, both slow.
- **28 EV underbody.** No data (FS) and no live demo.
- **21 / 31 / 30 (ride-hail, bank, listings).** Payers build in-house or show no evidence of pain.
- **29 Car wash.** Owner pain is shown only by a Kazakh tech task [S]. Partners do not care.
- **11 Showroom.** No evidence. A local retail-CV player (TASS Vision) exists (FS).
- **25 / 26 (city violations).** The Huawei-led Safe City (FS) is the incumbent and the city is a slow payer.
- **33 Repaint.** No data, and a paint gauge is the right tool.
- **23 Driver monitoring.**
  - a5: falling asleep caused 4.2% of crashes [S].
  - f3: cheap aftermarket DMS cameras exist [I], and new Toyota/BYD models have driver monitoring built in [I].
- **14 Grey import.** A VIN database lookup solves it without CV.
- **32 / 35 / 17 / 10 (counterfeit, parts ID, moto check, yard).** No data or no evidence.
- **34 Piskent.** Dead: testing ended 2 Sep 2025 [F].

---

## 4. Deep dives

### 4.1 WINNER: "Qabul", AI vehicle reception and handover act (partner product)

**One line.** Every service visit starts and ends with an act that nobody can check. Qabul turns a 60-second phone walkaround plus one dashboard photo into a signed act (plate, mileage, warning lights, panel-level damage map), and at pick-up shows what changed and whether the car can be released.

**Unit and process**
- **Owner:** Royal Motors Aftersales, Service Reception (service consultants / advisors).
- **Sites:**
  - Toyota Tashkent City and Royal Motors Qushbegi, whose express-service zone needs fast reception (FS).
  - BYD Chigatoy, which takes online bookings only 09:00–17:00 (FS), so after-hours drop-offs need an act.
  - Ducati Tashkent City, partially (see risks).
- **Processes changed:**
  1. Vehicle reception (*приёмка*, the acceptance act).
  2. Active delivery (*выдача*).
  3. After-hours drop-off with security.

**Problem and evidence**
- **Disputes turn on the act.**
  - BYD Megawatt (Tashkent) relied on the missing act in a public dispute: the car was "left overnight with security without an acceptance act" (kun.uz, 13 Mar 2026) [F].
  - Tashkent dealer reviews mention handover problems 14 times, reception wait 23, price/upsell 55, repair quality / repeat visits 46, and paid items not done (BYD Chigatoy, 1 Oct 2026) (FS [F]).
  - The Competition Committee logged 350+ car-service complaints in 2025, and the President called servicing of foreign brands insufficiently developed (FS [F]).
- **More dealers, more service competition.** Official dealers of 21 foreign companies now operate, with >100 models type-approved [F]. Toyota Academy Central Asia wants "one standard of service quality at all dealers" (FS [F]).
- **The process exists in Toyota's world.** "Active Reception" walkarounds and recording an accurate odometer reading at intake are standard dealer practice [S]. Ravin sells gate-CCTV "Autoscan" to service centres and shows a Toyota logo among its clients [F]. Toyota Tsusho invested in UVeye (FS [S]).
- **Gap.** No Uzbek deployment was found [S].
- **Honest gap.** Royal Motors' own sites show no waiting/status complaints, and handover complaints there were not specifically found (FS). The pitch therefore has to be "protect and prove your quality, and make it one standard across sites", not "fix your failures".

**User and payer**
- Users: the advisor captures; the customer receives and signs the act; the service manager sees the disputes dashboard.
- Payer: dealer aftersales, on a per-site subscription.

**Solution flow**
1. **Reception.** The advisor opens a web app (no install) or Telegram mini-app. Guided capture: walk around the car for 30–60 s, then take one dashboard photo. On-device checks for blur, exposure and coverage of 8 sectors.
2. **Act generated in ≤1–2 min [I target].** It contains:
   - plate (Uzbek formats; the green EV/hybrid field is flagged), and VIN if visible;
   - odometer;
   - fuel / state of charge;
   - active warning lights with severity;
   - damage list per panel, with crops;
   - optional: tyre DOT age.
   The customer signs on the advisor's phone, and a copy goes to Telegram/SMS. The act also exports to the DMS as PDF/JSON.
3. **Actions at reception:**
   - evidence-backed offers: touch-up / paintless dent repair (PDR) / Ziebart for chips and scratches (the Ziebart rollout is FS [F]); glass;
   - EV/hybrid cars routed to high-voltage-qualified technicians (BYD hires diagnostician-electricians, FS);
   - warning lights pre-filled into the repair order.
4. **Handover.** A second capture is compared with the first:
   - per-panel inventory diff, plus an aligned-crop change check;
   - warning-lights-cleared check;
   - mileage delta.
   Decision: **release**, or **hold** with evidence (e.g. "new mark, rear-left door, 2 views").
5. **Gate mode (stretch).** A fixed camera on the reception lane, or any street video, gives plate + visible damage per passing car. This is also what handles a jury dashcam or street clip.

**Architecture sketch**
```
Phone (web app / Telegram mini-app)
  capture guide + on-device QA (blur, exposure, 8-sector coverage) -> keyframes (12-24) + dashboard photo
Server (edge PC RTX 4060 Ti at the dealer, or cloud GPU)
  1 Vehicle + part segmentation ........ YOLO-seg (3DRealCar parsing maps, car-part sets)
  2 Damage instance segmentation ....... YOLO-seg (CarDD / VehiDE / Roboflow CC BY sets)
  3 Multi-view consensus ............... track each damage in panel coordinates across frames;
                                         reflections slide over the panel when the camera moves, damage does not
                                         -> keep instances seen consistently in >=2 views
  4 Plate detect + OCR ................. grammar-constrained decoding: "01 A 123 BC" / "01 123 ABC" / moto square / taxi;
                                         green region-field colour check -> EV/hybrid flag
  5 Dashboard reader ................... odometer detect+OCR (TRODO), warning-light detector, fuel/SoC
  6 Inventory -> act (PDF uz/ru) -> signature -> Telegram copy -> JSON/CSV to DMS
  7 Handover diff ...................... inventory diff + aligned panel-crop change check (dust/wash-aware thresholds)
  8 Rules .............................. offers, HV routing, hold-if-warning-not-cleared, release/hold decision
```

**Data plan per component** (real vs synthetic; team-built test sets)

| Component | Real data (licence) | Synthetic | Team-built test set | Metric |
|---|---|---|---|---|
| Part segmentation | 3DRealCar parsing maps (Apache-2.0) [F]; car-part sets 3.8k–9k (FS; check licences) | – | Held-out 3DRealCar cars | mIoU |
| Damage segmentation | CarDD 4k / 9k+ instances (licence form) [F]; VehiDE 13,945 (licence not found) [S]; Roboflow damage sets (check each, many CC BY 4.0) | Copy-paste augmentation | CarDD test split + panel-level labels on 30–50 multi-view sequences | mAP50; panel recall; **false alarms per clean car** |
| Multi-view consensus | 3DRealCar reflective / standard / dark subsets of clean cars [F/S] | – | 50–100 clean cars | False alarms per car **before vs after** consensus |
| Plates | Uzbek_Plates 8,434 (CC BY 4.0) [S]; Central Asia plates (Mendeley) [S] | Rendered Uzbek plates incl. green field and moto square | 300 hand-transcribed crops | Exact match %; per-character accuracy; EV-flag accuracy |
| Odometer | TRODO 2,389 (CC BY 4.0) [F]; motorcycle odometer set 5,860 (IEEE DataPort, licence not verified) [S] | – | TRODO held-out split | Exact match %; within ±1% |
| Warning lights | Roboflow sets (CC BY 4.0) [S] | Icon compositing on cluster backgrounds | Held-out split | mAP50; per-class F1 |
| Handover diff | – | CarDD damage masks composited onto 3DRealCar views, consistent across neighbouring views; dusty/washed variants | 200 synthetic change pairs + 50 no-change pairs (**labelled synthetic**) | New-damage precision/recall; false "new" rate |
| Baseline | Open VLM (e.g. Qwen2.5-VL, local) prompted on the same sets [I plan] | – | Same sets | Accuracy, latency, cost/inspection versus our pipeline |
| End to end | – | – | 20 full inspections | Latency; cost/inspection |

**Demo on jury input**
1. A judge scans a QR code and films any car (their own, in the venue car park) for 30–60 s, then photographs its dashboard. The act appears with plate, mileage, warnings and a damage map. The judge signs.
2. A removable mark (painter's tape or a scratch decal) is put on one panel and the car is filmed again. Handover mode flags the change on the right panel.
3. Any street or dashcam clip in gate mode gives plate + visible damage per car.
4. The team states failure cases openly: tiny scratches on dusty cars, black cars at night, dirty plates.

**Engineering answers** (brief: where it runs, speed, cost, night, rain, local cars)
- **Where inference runs:** capture QA on the phone; models on a dealer edge PC with an RTX 4060 Ti (~$925–1,345 in Tashkent, FS) or a cloud GPU.
- **Speed:** target ≤60 s per inspection [I]; measure and report it.
- **Cost:** ~$0.01–0.03 per inspection, from edge-PC amortisation [I].
- **Night and rain:** test on 3DRealCar's dark and reflective subsets plus synthetic rain/low light. Reception lanes are lit and covered [I].
- **Local plates:** dedicated Uzbek plate test set, with green-field EV plates.

**Moat (replication test)**
1. **Pipeline, not one model call.** Multi-view consensus and handover diff, with numbers showing false alarms cut on clean reflective cars and a VLM baseline beaten on the team's own test set.
2. **Local data flywheel.** Every advisor confirmation becomes labelled Uzbek data: dust, local models, green plates.
3. **Integration into a named process.** Act, signature, DMS export, Telegram.
4. **Domain rules.** Act fields, warning severity, HV routing, release/hold policy.

Honest note: a strong team could build a demo-level copy in a weekend. The moat is real only after the pilot.

**Three-month pilot** (one site; BYD Chigatoy or Toyota Qushbegi)
- **Weeks 0–2:** consent line in the work order, staff briefing, install the edge PC.
- **Month 1 — shadow mode:** paper act and Qabul run side by side; measure detections against advisor labels.
- **Month 2:** live act on alternate days (A/B).
- **Month 3:** full use, including the handover diff.
- **KPIs:**
  - reception minutes per car;
  - act completeness;
  - disputes and goodwill cost;
  - approval rate of extra work;
  - review mentions.
- **Pilot budget [I]:** edge PC ~$1.0–1.35k or cloud ~$50/month, plus team time. Offer the pilot free in exchange for the case study and local labels.

**Unit economics [I]**
- **Assumptions, to validate with partners:**
  - 30 cars/day × 26 days ≈ 780 visits per month per site.
  - Price $250–400 per site per month, i.e. ≈$0.3–0.5 per visit. This is below the international anchors of $1.25–4.20 per inspection [S].
- **Value per month:**

| Source of value | Assumption | Value / month |
|---|---|---|
| Advisor time | 3 min saved per car ≈ 39 h, at ~$4/h | ≈$150 |
| Avoided goodwill | 1–2 disputes × $100–200 | $100–400 |
| Extra approved work | 1% of visits convert, $40 gross margin each | ≈$300 |
| **Total** | | **≈$550–850** |

- **Payback:** cost of ~$300/month gives payback within 1–3 months. Every input is an assumption.
- **Market size in Uzbekistan:** >190 official dealerships (FS, 2024) and growing. 200 sites × $300 × 12 ≈ $0.7M/yr [I].
- **Expansion with the same tech:** insurers (KASKO inspection, claims), taxi parks, rentals (§4.4).

**Risks and known failure cases** (adversarial)
1. **Crowded lane.** The jury will see several damage detectors. The first 30 seconds of the pitch must show the act, the diff and the dashboard reading.
2. **Dust and washing.** Many dealers wash cars before handover [I]. A washed car reveals scratches that dust hid, which can produce false "new damage" flags. Counter with dust/wash-aware thresholds, multi-view support and human confirmation.
3. **Reflections and rain on dark paint** remain the top false-positive source, even with consensus filtering.
4. **Small damage.** Recall for scratches and chips under ~1–2 cm on 1080p video will be low. Report it, don't hide it.
5. **Domain gap.** CarDD consists of internet close-ups of damaged cars. Walkaround frames of dusty cars are different. The handover diff is validated only on synthetic pairs.
6. **Ducati.** Plates, TFT odometer and tyres work; damage on motorcycle fairings is not trained (no motorbike damage set was searched for or found).
7. **Copyable parts.** Dashboard reading is VLM-replicable and plate OCR is commodity. Neither should be presented as the innovation.
8. **Unknown local ROI.** Dispute frequency and goodwill cost are unknown, so the business case rests on assumptions.
9. **Build risk.** 4+ models plus UI in ~1.5 days by 3 students. Use the cut list in §8.
10. **CarDD access** may arrive late because of the licence form. Fall back to Roboflow CC BY sets and VehiDE if obtainable.
11. **Liability.** The AI is an evidence assistant and the advisor signs the act. No "AI verdicts".

**Likely partner-judge objections, with answers**

| Objection | Answer |
|---|---|
| "We already do a walkaround / paper act." | Same step, still ~60 s. But the act writes itself, it is searchable, and the handover comparison is automatic. Ask: how long does yours take, and how often is it skipped at night? |
| "False alarms will annoy advisors." | Show the measured false-alarm rate per clean car before and after multi-view consensus, and human confirmation. |
| "Customers will feel accused." | Both sides get the same signed evidence. It protects the customer too (mileage delta, no new damage). |
| "Our distributor has its own tools." | Exports PDF/JSON into any DMS; no lock-in. Ask what Toyota Central Asia / BYD Uzbekistan mandate. |
| "Privacy." | No face processing (background faces are blurred). Plates are covered by the service contract. Data stays on a local server. ZRU-547 requires local storage only for biometric, genetic and telecom data (FS). |
| "Does it work for Ducati?" | Partly. Plates (square), odometer and tyres yes; fairing damage, honestly not yet. |
| "Why AI, not a tablet with photos?" | Photos alone don't find damage, don't read the cluster and don't compare 40 photos at pick-up. Those three steps are where disputes are lost. |

### 4.2 RUNNER-UP: "Show-me inspection", evidence-backed MPI with Telegram approval (partner product)

- **Unit and process:** Royal Motors Aftersales, workshop and advisors. Changes the multipoint / "10-point" inspection and the customer's approval of additional work.
- **Problem and evidence:**
  - Price/upsell is the top complaint category (55) in Tashkent dealer reviews, together with paid items not done (FS [F]).
  - Toyota ran two free 10-point diagnostic campaigns with vouchers in 2026 (FS [F]).
  - Internationally, photo/video documentation roughly doubles approvals (~51% vs ~24%) [S].
  - Vendor claims, e.g. Porsche Centre Langley +$1,994 per RO with video, are unverified [S].
- **Solution:**
  1. The technician follows a guided capture checklist: 4 tyres (tread + sidewall/DOT), windscreen, lights on, body.
  2. CV classifies and measures each item: tread condition class, tyre age from DOT, sidewall cracks, glass chips, body damage.
  3. Each item becomes red/amber/green with an evidence crop and the dealer's price.
  4. The customer approves or declines in Telegram, and approved lines go into the RO.
  5. A distributor-level QA view covers inspection coverage and outlier technicians (oversell/undersell flags), supporting "one standard of service quality".
- **Data:**
  - Tread condition: Roboflow "Tire Tread" (CC BY 4.0) [S]; TyreNet (Mendeley, licence not seen) [S].
  - DOT OCR: TyreAI 4k (FS; I did not verify it).
  - Tread depth in mm: TyreCV (FS; **not found** by my search).
  - Body: shared with the winner.
  - Glass chips, brake discs, wipers: not verified or thin.
- **Demo:** the jury photographs tyres, the DOT code and the windscreen of any car and gets the report with prices. The demo risk is DOT OCR on dirty, embossed sidewalls.
- **Moat:** measurement plus standardised evidence across sites; QA analytics; domain rules (maintenance schedules).
- **Weakest points:**
  1. Technicians already have gauges. CV adds trust and standardisation, not a measurement that didn't exist.
  2. Tread depth in mm from a single photo is unreliable.
  3. Customers may read it as a sales tool.
  4. Approval rates in Uzbekistan are unknown.
- **Kill if:**
  - partners already send photo/video recommendations with high approval;
  - DOT read rate is <60% or tread-class accuracy <80% on the team's test set.
- **Why it is runner-up rather than winner:** weaker CV necessity, thinner data for measurement, and no custody/dispute hook. It is the natural phase 2 of Qabul and reuses the same capture app.

### 4.3 The "obvious partner" alternative: service-bay status and productivity analytics

- **Strengths:**
  - Highest complaint volume: waiting / status / reception, 83 mentions (FS).
  - Customers already check bay cameras themselves (FS).
  - Precedents: SKAIVISION has a Toyota dealer customer; Astana Hub ran a "VisionFlow AI" task (FS).
  - "Workshop productivity" is a BYD dealer KPI (FS).
  - Likely fewer competing teams.
- **Why it ranks 9th:**
  1. No labelled bay CCTV is public (FS); the Kinetics clips are handheld YouTube video.
  2. The jury cannot film bay CCTV on the spot, so the live demo would run on clips the team picked.
  3. Royal Motors' own sites show no waiting complaints, and Qushbegi has glass walls built for exactly this transparency (FS).
  4. Non-CV fixes are credible: DMS job clocking, an advisor's Telegram status updates.
  5. Monitoring staff on camera is sensitive; employers must inform staff (FS).
  6. The pilot needs NVR/IT access.
- **What would flip it:** a Royal Motors judge brings 30–60 min of their own bay CCTV, or says throughput / bay idle time is their #1 KPI. The score then rises to ~6.1.

### 4.4 Startup alternative: "condition passport" for insurers and fleets

- **Same technology as the winner**, sold to:
  - insurers, for remote KASKO pre-inspection and claim photo triage;
  - taxi parks and rental firms, for shift and rental handover diffs.
- **Evidence:**
  - The Uzbek insurance market is growing fast (2025 premiums 13.46 trn UZS, +38%) [S].
  - No Uzbek insurer uses AI photo inspection (FS [F/NF]).
  - Sug'urta Servis already collects accident notices without police [S].
  - Complaints about OSAGO payout delays and low limits were >50% of all complaints (~2,700) (FS).
- **Weak points:**
  - KASKO volume not found.
  - Insurer pilots are slow and regulated.
  - Taxi-park rental practice in Tashkent: not found.
  - Partner judges would be lukewarm, except that faster insurer approvals help dealer body shops [I].
- **Use:** as the business-expansion slide of the winner, not as the main pitch to this jury.

### 4.5 Road-safety alternatives (if the team wants impact over partner fit)

- **Courier two-wheeler compliance (S):**
  - Evidence: the strongest fresh numbers — 23,785 violations, ~4.7 bn UZS fines, 92% for equipment/behaviour, 89 deaths in 9 months [F].
  - Payers: Yandex Eats/Go and Uzum Tezkor; Uzum Tezkor and the Yandex delivery firms are classed as dominant platforms [S].
  - CV: a pre-shift photo check of helmet, headlight, reflectors and plate.
  - Weak points: moped reflector/plate data does not exist; dealers are indifferent; some violations (passengers, behaviour) cannot be checked at shift start; platforms may already run selfie checks (not found).
- **Road-damage mapping (S):**
  - Strong: RDD2022 data [S], a real city pain [S] and a good street-video demo.
  - Weak: a crowded, well-known task; the city is a slow payer; partner judges are indifferent.

---

## 5. Decision

- **Winner: Qabul (reception & handover act), score 6.60.** Reasons:
  1. It is the only top idea where the **live demo on arbitrary jury input** is natural: any car, any phone, plus street video.
  2. **All core components have real, licensed public data.** This includes a commercial-use multi-view car dataset (3DRealCar, Apache-2.0) to measure false alarms and night/reflection robustness.
  3. It changes a **named process the partner already runs** (reception / active delivery), with a **local dispute on record** (BYD Megawatt).
  4. It produces **decisions**: release or hold, routing, offers.
  5. The **pilot is cheap**: phones plus one edge PC.
- **Runner-up: Show-me inspection (6.16).** It targets the top complaint (upsell distrust) with the same capture stack, but CV is less necessary and measurement data is thinner.
- **Ask about, as a possible upgrade:** warranty-claim photo QA. If partners say claims get rejected over photos or mileage evidence, it climbs to ~6.4 and fits as a Qabul module (VIN + odometer + part photos).

### Adversarial summary of the winner's weakest points
- **Crowded lane:** innovation ≈4.6/10 by my heuristic.
- **The pain may be small at Royal Motors itself:** few complaints were found there.
- **A tablet checklist covers part of the value.**
- **The handover diff is proven only on synthetic pairs.**
- **Dust, wash and reflections** are the dominant real-world failure modes.
- **The ROI rests on unknown dispute counts.**
- **Ambitious scope for 1.5 days.**

---

## 6. Kill criteria (evidence that would falsify the winner)

**Before pitching (team's own test sets):**
- **K1.** Damage recall at panel level is <0.6 for damage ≥~3 cm, **or** false alarms are >1.0 per clean car after consensus (3DRealCar clean set).
  - Action: demote damage to "advisor-assisted tagging" and lead with plate + odometer + warnings + diff, or switch to the runner-up.
- **K2.** Plate exact-match is <85% on Uzbek crops, **or** odometer exact-read is <80% on TRODO held-out.
  - Effect: the "act fills itself" claim fails.
- **K3.** End-to-end latency is >3 min per inspection on available hardware.
  - Effect: the reception workflow breaks.

**From partner representatives (ask at the final):**
- **K4.** Reception is already digital with photos **and** condition/mileage disputes are fewer than ~1–2 per site per month **and** reception time is not a bottleneck.
  - Action: pivot the pitch to the runner-up (approvals) or to warranty QA.
- **K5.** Toyota Central Asia / BYD Uzbekistan forbid third-party tools in reception or mandate their own global tool.

**During the pilot:**
- **K6.** After 4–6 weeks of shadow mode, advisors confirm <70% of AI-flagged damage, or capture is skipped on >30% of cars.

---

## 7. Questions for partner representatives

1. How is a car received today at Toyota Tashkent City, Qushbegi, BYD Chigatoy and Ducati: paper act, DMS form or photos? How many minutes does it take?
2. How many condition disputes (scratches, rims, mileage, items) per month, and what is the typical goodwill cost?
3. What share of cars is dropped off outside working hours or left with security? Is an act done then?
4. Which DMS do you use (e.g. 1C Alfa-Auto, BYD DMS, a Toyota global system)? Can it take a PDF or JSON act?
5. Do you wash cars before handover? (This affects the handover diff.)
6. How are extra-work recommendations presented today, and roughly what share is approved?
7. Do Toyota Central Asia or BYD require a specific reception checklist or tool ("Active Reception")?
8. Are warranty claims ever rejected for missing or poor photos, VIN or odometer evidence? (This is the sleeper idea.)
9. Would you allow a fixed camera in the reception lane (gate mode)?
10. Where can pilot data be stored, and does the work order include photo consent?
11. Which site would host a 4-week shadow pilot, and which single KPI would convince you?
12. For Ducati: is motorcycle reception the same process, and is there an April peak?

---

## 8. If the team picks the winner: a ~36-hour build plan

| Hours | Work |
|---|---|
| 0–2 | Send the CarDD licence form immediately. Download the Roboflow damage set(s) (CC BY 4.0), TRODO, Uzbek_Plates, a warning-light set, and a 3DRealCar subset (~50–100 cars; check download size first). Start the README data/licence table. |
| 2–8 | Train YOLO-seg damage, part segmentation, plate detector and warning-light detector. Build odometer detection + OCR with digit-constrained decoding. |
| 8–16 | Pipeline: keyframes, multi-view consensus, inventory, act PDF (uz/ru), FastAPI + web capture page, Telegram bot. |
| 16–22 | Inventory diff, then the crop-level change check on synthetic pairs. Rules: offers, EV flag, release/hold. |
| 22–28 | Evaluation scripts → one metrics table: per-component metrics, false alarms before/after consensus, VLM baseline, latency, cost/inspection. |
| 28–34 | Public deployment. Test on team members' own cars with consent (if filming on site is allowed). Build the deck with a failure-case slide. |
| 34–36 | Rehearse the live jury-car demo, including the tape/decal handover trick. |

**Cut list, if behind (in this order):**
1. Crop-level change check (keep the inventory diff).
2. Warning lights (keep the odometer).
3. Telegram (keep the web app).
4. Part segmentation (fall back to 8-sector viewpoint labels).

**Pitch opening, to stand apart from other damage-detection teams:** "Every dealer dispute starts with one question: what did the car look like when you took it? Today the answer is a paper act; at BYD Megawatt it didn't even exist. Qabul turns 60 seconds of video into a signed act and, at pick-up, proves what changed. Give us your car."

---

## 9. Sources (my own lookups; FS items are cited in the fact sheet)

**Read on the page [F]**
- WIUT Hackathon 2026, CV track: https://hackathon.wiut.uz/tracks/computer-vision/
- BYD Megawatt dispute (kun.uz, 13 Mar 2026): https://kun.uz/ru/news/2026/03/13/3-raza-vyshla-iz-stroya-za-10-dney-v-tashkente-kuplennyy-byd-seal-6-okazalsya-brakovannym
- Moped/courier violations (spot.uz, 6 Oct 2026): https://spot.uz/ru/2026/10/06/couriers
- Piskent testing ended (kun.uz, 2 Sep 2025): https://kun.uz/en/news/2025/09/02/uzbekistan-ends-testing-of-imported-cars-at-piskent-ground
- CarDD: https://cardd-ustc.github.io/
- TRODO (Mendeley, CC BY 4.0): https://data.mendeley.com/datasets/6y8m379mkt/2
- 3DRealCar toolkit README (Apache-2.0, free commercial use): https://github.com/xiaobiaodu/3DRealCar_Toolkit
- Ravin AI: https://www.ravin.ai/

**Search summaries only [S]**
- 3DRealCar paper (size, lighting): https://arxiv.org/html/2406.04875v2
- Uzbek_Plates (Roboflow): https://universe.roboflow.com/suxrobs-workspace/uzbek_plates
- Central Asia plates: https://data.mendeley.com/datasets/4c5z833m6m/2
- Dashboard warning-light sets: https://universe.roboflow.com/cart/carfinale ; https://universe.roboflow.com/abdelrahamn/car_dashboard_warning_lights-ka1bd
- Tyre tread (CC BY 4.0): https://universe.roboflow.com/bams-workspace/tire-tread-82ob2
- TyreNet: https://data.mendeley.com/datasets/32b5vfj6tc
- VehiDE: https://doaj.org/article/6e0eb600059440b390431a42dfe8d138
- RDD2022: https://arxiv.org/abs/2209.08538v1 ; licence per https://datasetninja.com/road-damage-detector
- Price anchors: https://www.damageid.com/?p=2603 ; https://aws.amazon.com/marketplace/pp/prodview-atho6z47gymk2
- Photo/video approval data (secondary): https://www.updatepromise.com/?p=12273 ; https://www.kimoby.com/blog/video-mpi
- Toyota Active Reception: https://www.checkeredflag.com/service/active-reception/ (plus dealer job postings in the same search)
- Pulsar AI: https://digitalbusiness.kz/2026-04-02/shazam-dlya-avtomobilya-kazahstanskiy-startap-nauchil-ii-nahodit-polomki-za-2-sekundi/ ; https://kapital.kz/tehnology/149021/startap-iz-kazahstana-hochet-snizit-dtp-s-pomoshyu-ii-diagnostiki-avto.html
- Insurance market 2025 (spot.uz; exact article not confirmed): https://www.spot.uz/ru/2026/09/29/imkon-sugurta/ ; NAPP open data: https://napp.uz/storage/files/shares/opendata/2025/4Q/
- Sug'urta Servis app: https://apps.apple.com/us/app/-/id6448982242
- OSAGO limits: https://anhor.uz/news/uzbekistane-summa-avtostrahovke/ ; https://podrobno.uz/cat/economic/limity-pokrytiya-po-osago-v-uzbekistane-uvelichat-vdvoe-s-2026-goda/
- Uzbekinvest AI underwriting: https://kun.uz/news/2023/12/26/ozbekinvest-2024-yildan-boshlab-ozbe
- Taxi market: https://anhor.uz/news/taxi-aggregator-market-in-uzbekistan/
- Yandex Go EV tariff: https://kun.uz/en/news/2024/05/01/yandex-go-to-launch-a-taxi-tariff-with-e-cars-in-uzbekistan
- EV charging / Tokbor: https://www.spot.uz/ru/tag/электрозаправки/
- Potholes: https://anhor.uz/news/roads/ ; https://anhor.uz/news/road-repair/
- Crash causes and statistics (kun.uz; exact page for the Q1 2026 cause split not confirmed): https://kun.uz/en/news/2026/07/28/nearly-2300-traffic-accidens-recorded-in-uzbekistan-over-six-months ; https://kun.uz/en/50492709
- Platform labour (Ministry of Labour): https://gov.uz/ru/bv/news/view/191004
- Uzum Tezkor dominant: https://m.kun.uz/en/news/2025/04/02/uzum-tezkor-named-dominant-digital-platform-in-delivery-market
- Astana Hub car-wash task: https://astanahub.com/en/tech_task/intellektualnaia-sistema-avtomaticheskogo-ucheta-i-kontrolia-avtomoek
- UzAuto trade-in: https://uzavtosanoat.uz/ru/news_show/trade_in
- 1C Alfa-Auto: https://a2is.ru/catalog/programmy-dlya-kompanij-po-remontu-avto/1s-alfa-avto

**Not found / not verified:**
- TyreCV dataset
- VehiDE licence
- CO3D licence
- KASKO volumes
- Tashkent taxi-park rental practice
- Royal Motors trade-in
- Any Uzbek AI-inspection deployment
- Primary J.D. Power report
- Official Toyota "Active Reception" standard document
