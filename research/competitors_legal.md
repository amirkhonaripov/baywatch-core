# Service-flow analytics for Royal Motors: competitors, CIS landscape, OEM tools, Uzbek legal constraints, pricing anchors

Research date: 2026-10-10. Author: web-research subagent for the hackathon team.
Product under review: CCTV-based service-flow analytics (plate-recognition arrival, reception, waiting, bay, work/idle, wash, ready, handover), on-site GPU inference, only events leave the site.

## How to read this report

Every claim is tagged:

- **[F]** fact read on the page itself (page was fetched and read).
- **[S]** search summary only (the search result said so; the page was not read or could not be fetched).
- **[I]** my inference.

Limits you should know about:

- The search tool is US-based. Uzbek and Kazakh coverage is thin, and "not found" there is weak evidence of absence. Treat every "not found" as "ask a local".
- Almost all performance numbers are vendor-reported. None is independently verified.
- The legal section is research, not legal advice. Have an Uzbek lawyer confirm the points marked as open before the final pitch.
- FX used for conversions: CBU official rate effective 10.10.2026 = 11,897.90 UZS per USD; 140.18 UZS per RUB [F, cbu.uz]. Conversions are mine [I].

---

## 0. Executive summary

1. **The idea already exists commercially, closely.** SKAIVISION (US) tracks cars through Entry, Lane, Staging, Rack (bay), Wash and Delivery on existing dealership cameras, with alerts for missed greet, vehicle delay, rack delay, rack empty, technician idle and delivery delay [F]. PitCrew by Leverege (US) does the same for tyre/oil-change chains with an edge appliance [F]. EyeQ Monitoring launched an ANPR-based key-to-key cycle-time product at NADA 2026 [F]. Do not pitch this as a world-first.
2. **A Toyota dealer is a named SKAIVISION customer** (Fred Anderson Toyota of Raleigh, deployed January 2024, about 30 cameras, about 40 bays) [F]. Southeast Toyota Distributors (via JM Family) sits on SKAIVISION's board [F]. No Toyota Motor Corp. or BYD programme that supplies such tools was found [S/F, see section 3].
3. **Nearest CIS precedent: Ivideon + AvtoVAZ.** More than 200 LADA dealer centres were connected to Ivideon cloud video in 2025, used for traffic counting, showroom load and service-standard compliance. A strategic agreement for more AI cameras in showrooms and service centres was signed at SPIEF 2026 [F]. Dealers reportedly pay for the equipment, renew a licence quarterly, and video goes to AvtoVAZ servers in Togliatti [F]. This is the cloud-and-OEM-controlled model we differ from.
4. **In Kazakhstan someone is tendering exactly our concept**: an AstanaHub task "VisionFlow AI" for service-bay occupancy, plate reading, ETD prediction and Telegram notifications, bids until 20.04.2026 [F]. Whether it was awarded is not known.
5. **Nothing deployed at Uzbek or Kazakh dealers was found.** Local video-analytics players found (TASS Vision, VEON AdTech/Beeline) are retail or generic, not automotive service [F].
6. **Legal picture changed in 2026.** Law ZRU-547 was amended by Law ZRU-1125 of 26.03.2026: mandatory in-country storage now applies only to biometric data, genetic data, and telecom-subscriber data; other personal data may be processed abroad if the destination is on the Cabinet's equal-protection list (49 countries approved 29 July 2026, including the US, UK, Germany, Russia) or contractual safeguards exist [F]. Registration in the state register is likewise limited to those in-country categories [F]. Biometric processing needs consent [F]. So: **no faces, no biometrics, events only, on-site** keeps us out of almost every regulated corner.
7. **CCTV-specific rules are still thin.** A 2024 legal analysis found no enacted duty to warn visitors about filming [F]. An image-rights bill (consent, warning signs, camera bans in fitting rooms/toilets) passed the Legislative Chamber in October 2025 [S]; whether it became law was not confirmed. Design for it anyway.
8. **Pricing anchors:** Russian cloud video is about 200-875 RUB per camera per month (about $2.4-10) and a face-analytics add-on is 4,700-8,800 RUB (about $55-104) [F via aggregator]. UVeye is quoted at $6,000-7,000 per lane per month [S]. Jetson Orin Nano Super kit rose to $399 in July 2026 [F]. No Uzbek retailer for Jetson was found; RTX GPUs are plentiful on OLX.uz (an RTX 4060 Ti card is about 4-5 million UZS, about $330-470) [F].

---

## 1. Global competitors

### 1.1 Summary table (closest first)

| Vendor | What it measures | Hardware | Pricing | Claimed results (vendor-reported) | Customers | Geography | Closeness to our idea |
|---|---|---|---|---|---|---|---|
| **SKAIVISION** | Greet timing, vehicle dwell, staging waits, rack (bay) occupancy, technician idle, wash and delivery timing, total visit cycle; plate-based arrival matched to customer record; also service-lane vehicle acquisition (trade-in offers) [F] | Existing dealership cameras; site review decides readiness; new/repositioned cameras if needed [F]. Hunter Quick Check Drive cameras can feed it [F]. Average dealer has triggers on about 32 cameras [F] | Not public; demo only [F] | Homepage: +15% express jobs under 60 min, +26-28% service-lane acquisitions, 98% greet rate, +18 CSI (unnamed stores) [F]. Fred Anderson Toyota: ToyotaCare 120-125 to 35-40 min, acquisitions 30-35 to about 105 [F]. Case-study page for the same store: basic service 75-90 to under 60 min, about $1M extra annual revenue [F] (the two pages quote different metrics) | Fred Anderson Toyota of Raleigh [F]; Mercedes-Benz dealer (unnamed) [F]; one search summary of a 2025 talk says 150+ dealerships [S] | US (Newnan, Georgia address) [F]. Backers: JM Family (July 2023, Series A, amount undisclosed) [F], Reynolds & Reynolds (announced Dec 2024 [S]; Wards article of 6 Jan 2025 calls it "a junior investment", amount undisclosed [F]) | **Very close** (same stage model) |
| **PitCrew (Leverege)** | Per-service times, customer wait, queue abandonment, labor minutes, foot traffic, bay utilisation, delayed services with cause codes [F]; searchable video per service [F] | Existing cameras/NVR plus "a small edge appliance" on the camera network; POS integration [F] | Not public [F] | +30 services per store per month, -11% delayed services, -13% service time [F]; one chain +8 services/day/store [F] | Discount Tire "in all their service centers" [F]; Jiffy Lube franchisee ACE rolling out after a pilot [S]; Oilex/Grease Monkey rollout announced 8 July 2026 [F, vendor press section] | Not stated; US-centric [I] | **Close** (bay-level, quick service, edge appliance) |
| **EyeQ Monitoring** | Key-to-key cycle time from arrival to completion, live vehicle location, alerts when vehicles near or pass time thresholds; planned wider BI suite [F] | Video analytics plus licence-plate recognition, "advanced camera technology" [F] | Not public [F] | None quantified [F] | Says it supports six of the top ten US dealer groups (security monitoring business) [F] | US, 42 states for monitoring [F]; launch at NADA 2026, press date 26 Feb 2026 [F] | Close on cycle time; security-company DNA |
| **Bayflow (Autoflow)** | Bay occupancy to the minute, time per work order, technician time on job via body detection, plate-to-work-order pairing, moves between bays [F] | Two wide-angle AI cameras per bay, vendor-installed (about 2 days), Wi-Fi [F] | Not public; cancel-anytime option [F] | 29 min saved per average job; 15%+ technician efficiency; 25%+ bay efficiency; no methodology [F] | None named [F] | Not stated [F] | Close on bay/technician time; independent shops |
| **Detect Auto** | Per-repair real-time analytics, alert if car sits in a stall longer than planned, plate-to-repair-order link, says no facial recognition [S] | Wi-Fi cameras [S] | Not public; waitlist [S] | Conflicting: 17+ hours/month billable time per bay (home page) vs 5+ hours (About page) [S] | Not found | US, early-stage; funding figures conflict ($2.9M vs $748K seed) [S] | Close on bay time; early-stage |
| **UVeye** (adjacent) | Drive-through scan of tyres, underbody, exterior at service-lane entry; feeds repair-order upsell; not flow analytics [F/S] | Floor/arch scanners | Confidential; $6,000-7,000 per lane per month quoted by a UVeye executive, plus $7-12K site prep per a pricing-estimate site [S]; GM promo credit up to $2,500/month [S] | Team Toyota of Princeton: +15% service work in 3 months, $100K alignment work in 90 days (UVeye's own case study) [F] | Toyota stores (Princeton, Claremont, Comox Valley, Cranbrook) [F/S]; GM, Volvo, Hyundai, JLR, Subaru partners [F] | US, Canada (launched 30 July 2026), Europe [S]; more than 1,000 systems globally [F] | Adjacent; a possible partner, not a flow competitor |
| **Hunter Engineering Quick Check Drive** | Hardware: reads plates at lane entry and feeds SKAIVISION [F] | Cameras in the inspection lane | Not stated [F] | None [F] | n/a | US [F], Oct 2024 | Partner channel for SKAIVISION |
| **Ravin AI** | Damage/condition inspection from phone or CCTV imagery; dealer app "Inspect Trade" [S] | Phone or CCTV | Not found | Not found | Used-car dealers, auctions, rental, insurers [S] | UK/Israel/US [S] | Not a service-lane product (no such product found) |
| **Spot AI** (generic) | General AI camera platform; dealer security and some service dashboards [S] | Own cameras/NVR | Quote-based, per camera feed; about $99/camera/month in one third-party comparison [S] | Don Franklin Auto (30 sites, Kentucky): over $1M recovered assets; no service-lane numbers [S] | Dealers, car washes (Glide Xpress about 30 s pay-station to tunnel) [S] | US | Generic; no service-flow model |
| **Lumana, Voxel** (generic) | Generic dwell/occupancy/safety analytics; no dealership or service-lane product found [S] | Existing cameras | Not found | Not found | Not found | US/Israel | Not competitors on this use case |
| **Car-wash analytics** (omniX Labs, bought by EverWash; AVUTEC ANPR; LiveReach) | Volume prediction, membership by plate, idle time at pay station, unattended vehicle alerts [S] | Existing cameras / ANPR | Not found | Not found | Car-wash operators | US/EU | Different sector; ideas for wash stage |

### 1.2 Notes that matter for the pitch

- **SKAIVISION's own pages disagree on Fred Anderson Toyota's numbers** (75-90 to under 60 min on the case-study page; 120-125 to 35-40 min for ToyotaCare on skaivision.com) [F]. If we quote any of these, say "reported by the vendor for one US store" and use the case-study number, which is the more conservative.
- **Where SKAIVISION processes video is not stated** (edge vs cloud). It claims SOC 2 Type II and says data scope depends on configuration [F]. License-plate recognition is used "where appropriate" [F]. This is a gap we can fill: they do not publish an events-only, on-site architecture.
- **PitCrew uses a "small edge appliance"** [F], so "on-site box" alone is not a differentiator against PitCrew. Its pages mention neither ANPR nor customer notification [F].
- **Discount Tire's internal 2022 "Bay Utilization" FAQ** (surfaced by search, not read in full) says bay cameras do not record video; images go to an ML program and only event data such as timestamps reach company systems [S]. This is a useful precedent that "events only" is accepted practice in a large chain.
- **Privacy backlash exists.** Motor1 reported a customer at Precision Toyota of Tucson objecting to an AI camera system around her 4Runner without consent; staff could not say where the data went [S, page returned 502, not read]. No vendor was named in what I could access. Use this as the "why events-only and signage matter" example, with the caveat that the vendor is unconfirmed.
- **A granted US patent exists in this area**: US 12,406,575 (assignee MW POS, Inc.; priority 2024-09-26; granted 2025-09-02) on matching degraded plate reads to stored vehicle profiles at service facilities such as car washes [F, claims not read]. Low relevance for Uzbekistan, but worth one line in a freedom-to-operate note [I].
- **Not found** (searched): Tekion or CDK camera-based service-lane tracking of their own (UVeye integrates with both and with Reynolds) [S]; a Bosch workshop camera product that recognises vehicles (Bosch "Connected Repair" is diagnostics software) [S]; a vendor called "Spiral" in this niche; Volteras (UK EV connected-car software, not cameras) [S].
- **Other regions (search only):** an ANPR-for-workshop-entry deployment at a large Norwegian dealer group, a Honda dealer in Japan using plate-recognition cameras (Vivotek) to improve reception, a Japanese repair-shop project with bay detection linked to plate recognition, and a Chinese "smart 4S store" solution from CUE Group on an Intel video-AI box [S]. None gave numbers I could verify.

---

## 2. CIS and Central Asia

### 2.1 Russia

| Player | What they offer relevant to dealers/service | Evidence |
|---|---|---|
| **Ivideon + AvtoVAZ (LADA)** | Cloud video platform deployed at 200+ LADA dealer centres in 2025 for visitor counting, showroom load, service-standard compliance; SPIEF 2026 agreement to expand AI cameras in showrooms and service centres; AvtoVAZ says system would assess staff compliance with service standards, monitor workload and flag problem situations in client areas; planned age/gender estimation and emotion tracking in showrooms [F, hi-tech.mail.ru, 4 June 2026] | [F] |
| AvtoVAZ dealer conditions | Equipment branded Ivideon bought from AvtoVAZ's official partner "Mobile Solutions" (Мобильные решения); software licence renewed quarterly; video goes to AvtoVAZ servers in Togliatti; headline says dealers pay and some are unhappy; AvtoVAZ says the plan was known in advance; rollout began 2024, launched 2025 [F, ixbt, 7 Oct 2025] | [F] |
| Ivideon (retail tariffs) | Cloud tariffs and face-analytics add-ons, see section 5 | [F via aggregator] |
| **Trassir** | Published dealer case is in Morocco (six branches): Neuro Counter, Face Recognition, AutoTRASSIR Parking, NeuroStation server; no numeric results; date not stated [F]. AutoTRASSIR logs plate, time, direction, lists [S] | [F]/[S] |
| **Macroscop** | Modules for object counting, queue length, staff-activity control, left-object detection [S]; no auto-dealer case found | [S] |
| **Vocord** | ANPR and parking; sources conflict on whether IP/team moved to Huawei in 2019, so check ownership before any partnership [S] | [S] |
| **NtechLab, VisionLabs** | Face and vehicle recognition mostly for city/road/retail; no service-centre cases found [S] | [S] |
| **Smart Engines** | Offline CIS plate recognition network and VIN recognition (0.2 s per code) claimed; auto group BorisKhof uses it for passport reading, not plates [S] | [S] |
| CRM side | Russian CRMs (Bitrix24 apps, Splus, 1C modules) key the work order on plate/VIN; stage-timeout alerts fire on CRM status, not physical presence [S] | [S] |

No ready Russian product combining plate-at-entry, work-order link and bay-idle control was found [S]. The AvtoVAZ/Ivideon scheme is the closest analogue and is OEM-mandated and cloud-based [F].

### 2.2 Kazakhstan

- **VisionFlow AI tender on AstanaHub** [F]: customer "ИП Smart Tech"; computer vision to show in real time whether each service bay is free or occupied, read plates (Nomeroff Net or similar, at least 90% accuracy on Kazakh plates day and evening), detect start/process/finish stages, predict departure time (ETD) by Standard/Express service, queue logic and Web Push/Telegram notifications; YOLOv8/v10 across 6+ zones, RTSP, bay status in the DB within 5 s, Python/FastAPI/PostgreSQL/Redis/React PWA with Telegram Mini App; NVIDIA GPU servers; scalable to 50+ cameras; bids until 20.04.2026; budget "1,500,000" with currency not stated; Kazakhstan residents only; 10 applications at time of reading. This validates demand and shows a Kazakh team may build the same thing. Whether it was awarded is **not found**.
- Aldente AI (camera data platform, NVIDIA Inception 2024) is generic [S].
- No Kazakh dealer deployments of service-flow analytics found.

### 2.3 Uzbekistan

- **TASS Vision** (Tashkent): on-camera AI for offline retail; analysis runs "directly on the device" so no cloud transfer; 2,000+ stores in nine countries including Uzbekistan, Kazakhstan, UAE, Turkey, Saudi Arabia; IT Park Ventures put $100K into a EUR 1.3M round; customers are FMCG, fashion, electronics, pharma; **no auto dealers mentioned** [F]. Useful as a local on-device-privacy precedent.
- **VEON AdTech (Beeline group) video analytics**, announced 15.02.2024: neural analytics on customer cameras, deployed at Beeline Uzbekistan's main sales office (staff performance, visitor numbers); claims 99% face recognition; deployable on customer servers or in cloud; 12-month subscription or one-time licence; no prices; only Beeline named [F]. The most likely incumbent a judge could mention.
- **Safe City**: Huawei is the main technology participant in the national concept [S]; Hikvision talks reported; AGT International (Swiss) installed traffic-violation cameras at 160 Tashkent intersections [S]. These are government systems, not dealer products.
- **ANPR/camera hardware**: Hikvision and Dahua are sold by local integrators (prom.uz, glotr.uz); one Tashkent listing is a Hikvision camera with built-in plate recognition (out of stock) [S].
- **Risk precedent**: the national plate-recognition database was reported exposed online without a password, tracking one car's movements for six months [S, anhor.uz/TechCrunch]. Argue "we store no plate history off-site".
- **Not found:** any Uzbek vendor selling service-lane/bay analytics; any Uzbek or Kazakh dealer using one; any "ANPR linked to service CRM" product in Uzbekistan.

### 2.4 Royal Motors / Toyota context in Uzbekistan (for the pitch)

- Toyota distributor: Summit Motors Central Asia (distributor for Uzbekistan, Tajikistan, Turkmenistan since 2018; March 2025 plan for a regional Toyota training centre in Tashkent) [S].
- A directory lists "Royal Motors" at Zargar street 3/1, Tashkent [S]; another directory lists "Toyota Tashkent City" at Zargarlik 3/1 [S]. Same address suggests Royal Motors runs Toyota Tashkent City [I] (unconfirmed; the team knows the dealer better).
- **Toyota Tsusho has a representative office in Tashkent and Uztelecom as strategic IT partner** (Russian version of an IT Park item; English version said the office was planned) [S]. Toyota Tsusho is an early UVeye investor with sales rights for UVeye's undercarriage system in Japan [S]. No Toyota Tsusho + UVeye distribution for Central Asia found [S].
- BYD Uzbekistan: JV of Uzavtosanoat (60%) and BYD Europe B.V. (40%) in Jizzakh [S]. Who runs the Tashkent BYD showrooms was not confirmed [S].
- Ducati: an official Tashkent 3S centre opened [S]; operator not confirmed.

---

## 3. Do Toyota or BYD supply such tools to dealers?

**Toyota**

- No Toyota Motor North America or Toyota Motor Europe programme supplying AI service-lane cameras was found [S]. Toyota Canada's "DDS" is a digital marketing programme [S]. The Toyota/Lexus dealer tablet app reads VINs with the device camera (access via dealer TIS credentials) [S].
- **UVeye lists Toyota as a partner** and has an offering page "for Toyota dealerships" (case study Team Toyota of Princeton). The page does not say Toyota endorses or funds it [F]. The June 2026 CARFAX release lists "Toyota (via Woven Capital)", i.e. an investment relationship [F]. Toyota stores adopt it individually (Comox Valley and Cranbrook Toyota in Canada, Claremont Toyota at NADA 2026) [S].
- **SKAIVISION has a Toyota-distributor link**: Kai Ramadan of Southeast Toyota Distributors (a JM Family subsidiary) joined SKAIVISION's board in July 2023 [F]. This is an investor/board tie, not a Toyota Motor program [F].
- Other OEMs do certify or subsidise tools: GM gave UVeye subscription credits up to $2,500/month [S]; Hyundai and GM maintain certified service-lane technology vendor lists (no cameras) [S]; **AvtoVAZ mandates Ivideon cameras** [F]. So an OEM mandate is possible, but none was found for Toyota or BYD.
- Not found: any Toyota Tsusho/UVeye programme in Uzbekistan, Kazakhstan, or the CIS [S].

**BYD**

- No BYD dealer-facing AI camera or bay-analytics tool found, in English or Chinese [S]. Job postings show BYD pushing DMS, LMS and TIS to dealers' workshops, and (Canada) tracking workshop productivity and repair efficiency as dealer KPIs [S]. BYD's site cites 2,300+ after-sales outlets and an online AI technician Q&A [S].
- Implication [I]: our comparable numbers (bay utilisation, wait time) could line up with BYD's KPI language, but an OEM tool could still appear later.

**Not searched:** Ducati dealer tooling.

---

## 4. Uzbek legal constraints

Not legal advice. Primary text read: lex.uz page for Law ZRU-547 [F]. Several sections rely on secondary sources and are marked.

### 4.1 Is a plate or a video image "personal data"?

- Definition (Art. 4): information recorded on any medium that relates to a specific individual or allows identifying them [F, lex.uz]. A frame showing an identifiable face fits this [I].
- **A plate on its own**: no Uzbek authority statement found [S/not found]. Once our system links the plate to a customer record or phone number in the dealer's CRM, the plate event is effectively personal data [I]. Plan on that basis.
- **Biometric data** (Art. 26): data describing anatomical and physiological characteristics; processed to identify a person only with the subject's consent, with exceptions in law [F]. A 2023 Ministry of Justice order reportedly lists fingerprints, facial features, eye colour and voice as examples [S]. Face recognition therefore falls here [I].

### 4.2 Data localisation

- Original localisation obligation (Art. 27-1) applied broadly to citizens' personal data; amendments dated 15.01.2021 and 21.04.2021 appear in the law's publication list [F]. Secondary sources date the April 2021 start and note 2021 enforcement against TikTok, Skype and others, and fines raised by ZRU-726 (29.10.2021: 7 BRV for citizens, 50 BRV for officials) [S].
- **Current text (as amended by ZRU-1125, signed 26.03.2026, published 27.03.2026)** [F, lex.uz]:
  - Must be stored in Uzbekistan: biometric data of individuals, genetic data of individuals, and data of individuals using telecom operators' services in Uzbekistan (Art. 27-1 part 2).
  - Everything else may be stored and processed abroad if one of three conditions holds: destination is recognised as providing equal protection; approved standard contractual terms or binding corporate rules; or compliance with approved international standards (part 3). The Cabinet sets the equal-protection country list (part 4).
- **Country list adopted 29 July 2026**: 49 countries including the US, UK, Canada, Japan, South Korea, Singapore, Russia, Germany, France, Italy, Spain, Netherlands, Switzerland, Brazil, Israel [F, kun.uz, 4 Aug 2026]. Breach during cross-border transfer: notify the authority within 24 h and report within 72 h [F].
- Practical meaning: a cloud backend is now **legal for non-biometric data** to listed countries, but any face-based data must stay in Uzbekistan [F/I].

### 4.3 State register ("davlat reestri") and the authority

- Art. 20 as amended 2026: databases that must be stored in Uzbekistan under Art. 27-1 part 2 must be registered in the State Register of Personal Data Databases; registration is by notification; changes must be notified within ten calendar days; exemptions include databases processed under labour law and name-only lists [F, lex.uz].
- [I] A system that holds only vehicle events and no biometrics should fall outside mandatory registration under the current text. A lawyer should confirm, and the dealer's own CRM may already need registration.
- Authority: Art. 8 names the State Personalization Center under the Cabinet of Ministers [F]. An August 2026 report says the Ministry of Internal Affairs' Department for Migration and Personalization proposes changes to the country list [F]. The institutional set-up may have shifted; verify who is the supervising body today. A body called an "Agency for personal data protection" was **not found**.
- 2023 Ministry of Justice sub-laws: standard procedure for processing personal data and one for responsible units, registered 15 Nov 2023 [S].

### 4.4 Consent, retention, access

- Consent may be in any form that confirms receipt; written (including electronic) for special categories such as biometrics; other lawful grounds exist (contract, legal obligation, protection of legitimate interests) (Arts. 18, 21) [F].
- Secondary summaries: consent must name the operator, purposes, data list and term and can be withdrawn; changed purpose needs fresh consent; data to be destroyed within 3 days after goals are met or consent withdrawn; employees may use personal data only per their duties [S, Mondaq/Legal500 summaries]. Verify the retention period against the text.

### 4.5 Employees (technician idle-time metric)

- Labour Code (new Code in force 30.04.2023 [F]) Arts. 175-180 protect employees' personal data; Art. 177 voids waivers of this protection; Art. 180 sets liability [F, article headings on lex.uz]. A law-firm summary says the employer must tell employees why and how data is collected, what is collected, and the consequence of refusing; consent is written; recipients must keep data confidential [F, Mondaq; read]. A registration exemption covers databases processed under labour law [F].
- **No Labour Code rule specific to CCTV monitoring was found** in the visible text [F, not conclusive, full text not read] and secondary sources [S].
- Draft image-rights bill would extend Art. 175 to employees' images and, per one report, exempt employers' filming of work processes and surveillance to prevent crime from the consent requirement [S, spot.uz]. Status unconfirmed.
- [I] Technician idle/work tracking is the most sensitive element. Default to bay-level and team-level metrics; show per-person data only after written notice and internal-rules sign-off.

### 4.6 Customers on private premises

- A February 2024 legal analysis (anhor.uz) states current law has no norm requiring a warning about photo/video recording at the entrance and proposes one; also proposes banning cameras in changing rooms, fitting rooms and toilets, and restricting who can view footage [F].
- **Image-rights bill**: Legislative Chamber first reading 9 Sep 2025 (106 votes for) [F, spot.uz]; passed 7 Oct 2025 and sent to the Senate [S]. Reported content: consent to film or use a person's image with exceptions (including public places, surveillance to prevent crime, employers filming work processes); under-16s need parent consent; warning notice where cameras operate; bans on cameras in fitting rooms and nude areas; covert-recording fines of 10-40 BRV and confiscation [S]. **Final status unknown**; the Civil Code text I could reach did not show an Art. 100-1 [S/not verified].
- [I] Customers who enter a service area with visible signage and who consented in the service contract are the safe path.

### 4.7 Implications: on-site vs cloud

| Question | On-site GPU box, events only (our design) | Cloud video analytics |
|---|---|---|
| Biometric data | None created if we never run face recognition | Common (face counting, emotion) and must then stay in Uzbekistan |
| Localisation | Events can stay on dealer premises or in an Uzbek data centre | Allowed abroad for non-biometric data only to listed countries or under SCCs; legal review per vendor |
| Registration | Likely not triggered [I] | Triggered if biometric; paperwork |
| Breach exposure | Small: no raw video leaving, no plate history exported | Raw video or plate streams leave the site |
| Staff acceptance | Easier: aggregated metrics, no video review by management | Harder |

**Design requirements we should state in the pitch** [I]:

1. No face recognition, no emotion/age/gender, no re-identification of people.
2. Raw video never leaves the dealer's network; the box emits events only (stage, bay id, timestamps, hashed or tokenised plate).
3. Plates: store only what the dealer's CRM already holds; use salted hash off the box; configurable retention (short default) and a delete-on-request path.
4. Signage in Uzbek and Russian at entrances and service floors; clause in the service order or terms.
5. Staff: written notice and a short internal policy before go-live; no cameras in rest rooms or changing areas; individual-level views off by default.
6. Dealer is "owner" and we are "operator" of any personal data; sign a data-processing contract (roles are defined in Art. 4 [F]).
7. Get counsel to confirm: plate = personal data when linked; registration non-applicability; status of the image law; supervising authority; retention period.

---

## 5. Pricing anchors

All prices are anchors, not quotes.

### 5.1 SaaS video analytics (Russia, no Uzbek tariff found)

| Product (source, date) | Price | UZS / USD (mine) |
|---|---|---|
| Ivideon "Online" tariff, per camera/month, annual prepayment (toolfox aggregator, updated 23.07.2026) [F] | 200 RUB | about 28,000 UZS, about $2.4 |
| Ivideon cloud 3 / 7 / 14 / 30-day archive [F] | 350 / 500 / 625 / 875 RUB | about 49,000 / 70,000 / 87,600 / 122,700 UZS; about $4.1 / 5.9 / 7.4 / 10.3 |
| Ivideon face-analytics module, per camera/month [F] | 4,700 (up to 1,000 faces/day), 6,450, 8,800 RUB | about $55, $76, $104 |
| Trassir Cloud, per camera/month, annual billing (toolfox, 23.07.2026) [F] | 85 RUB online, 213 (7 d) to 942 (90 d) archive; free up to 3 cameras; analytics modules not priced | about $1.0-11 |
| Macroscop | Per-camera licence; analytics modules extra; 2021 promo prices only, no current price [S] | not usable |
| VEON AdTech (Beeline Uzbekistan group) | 12-month subscription or one-time licence; no prices [F] | not found |
| Ivideon's own tariff page returned 404; aggregator figures have internal discrepancies (350/500/625/875 vs 375/475/575/825) [F] | | |

### 5.2 Western vertical tools

- UVeye: $6,000-7,000 per lane per month including training and upgrades (executive quote) [S]; GM promo credits up to $2,500/month [S].
- Spot AI: quote-based per camera feed; about $99 per camera per month in one procurement comparison [S].
- SKAIVISION, PitCrew, EyeQ, Bayflow, Detect Auto: **no public pricing** [F].

### 5.3 Edge hardware

- **NVIDIA Jetson prices rose in July 2026**: Orin Nano Super dev kit $249 to $399; Orin NX 16GB module (1,000-unit price) $599 to $999; Orin NX 8GB $399 to $649; AGX Orin dev kit $1,999 to $3,499; NVIDIA gave no official reason; some modules were phased out earlier due to LPDDR4 prices [F, CNX Software, 22 Jul 2026].
- Third-party Orin NX 16GB kits: Yahboom $1,593-1,824 on RobotShop Canada; an Australian reseller's fanless Orin NX 16GB unit at A$2,663 [S]. A Dubai classified listing for an Orin Nano dev kit at AED 1,499 (June 2026) [S].
- **Uzbek retail for Jetson: not found** (no OLX.uz, Uzum, Asaxiy, Texnomart or Mediapark listing surfaced). The UAE looks like a realistic sourcing route [I]. Waveshare's store lists restricted destinations (Russia, Belarus); Uzbekistan was not named there [S]. An NVIDIA-Uzbekistan AI initiative (7 Nov 2025) mentions compute clusters targeted for 2026 subject to export controls [F].
- **RTX-class GPUs and PCs in Tashkent** (OLX.uz private listings, Oct 2026 unless noted; asking prices, many negotiable) [F]:
  - used RTX 4060 Ti card: 4,641,312 UZS (about $390), 2 Oct [F];
  - new MSI RTX 4060 Ti: 3,950,978 UZS (about $332), 28 Jul; new Gigabyte RTX 4060 Ti 8GB: 5,615,640 UZS (about $472), 30 Jul; new RTX 5060 Ti 16GB: 7,000,000 UZS (about $588), 10 Jul [S, search summary of OLX.uz listings, not individually opened];
  - used gaming PC with RTX 4060 Ti: 11,000,000-16,000,000 UZS (about $925-1,345); new i5-13400F + RTX 4060 Ti PC: 12,138,816 UZS (about $1,020) [F, OLX.uz results page read on 10 Oct 2026];
  - Magazine prices (Mediapark, Texnomart, Asaxiy) **not found**.
- Local camera and NVR hardware (prom.uz/OLX.uz, [S]): Hikvision DS-2CD1327G0-L from 380,000 UZS (about $32); 16-channel NVR from 2,944,000 UZS (about $247); 64-channel NVR 5,737,500 UZS (about $482).
- [I] A built RTX 4060-class PC (about $1,000-1,350) is the cheaper and locally buyable edge option; Jetson is cheaper to run but harder to source and now more expensive.

---

## 6. Implications for our pitch

### 6.1 Positioning

1. **Do not say "nobody does this".** Say: "This workflow is proven abroad (SKAIVISION, PitCrew, EyeQ); no one offers it in Uzbekistan, and none of them is built for our legal and operating reality."
2. **Differentiators, in order of strength:**
   - **Compliance by architecture.** On-site inference, events only, no faces or biometrics, hashed plates. Compare: the Ivideon/AvtoVAZ model sends video to OEM servers and plans emotion/age/gender analytics [F]; SKAIVISION does not state where processing runs [F].
   - **Multi-brand group comparison.** Royal Motors spans Toyota, BYD and Ducati; the same bay-time and wait-time definitions across sites and brands give management comparable numbers that no OEM portal provides, because each OEM sees only its brand [I].
   - **Customer notification in local channels** (Telegram/SMS in Uzbek and Russian) triggered by stage changes. None of the vendors above showed customer-facing stage notifications except acquisition texts at SKAIVISION [F/I]. Check the consent wording for outbound messages.
   - **Local fit**: Uzbek plate formats, local hardware and UZS pricing, no per-camera foreign SaaS, supportable in Tashkent.
   - **Cost**: price a pilot against UVeye's $6-7K per lane per month, or against Ivideon's analytics add-ons ($55-104 per camera per month), not against bare CCTV.
3. **Stage coverage:** SKAIVISION already lists Entry, Lane, Staging, Rack, Wash, Delivery [F]. Our edge is the join to the dealer's work-order/CRM and the measured "lost bay time" (bay occupied but nobody working), which only some vendors show (Bayflow, PitCrew, Detect Auto) [F/S].

### 6.2 Answering "we already have X" from a judge

| Judge says | Answer |
|---|---|
| "SKAIVISION/PitCrew already does this." | "Yes, and it is the proof the model works: a Toyota store in Raleigh reports faster service. Those are US products tied to US DMS integrations, English UI and, for SKAIVISION, an undisclosed processing location. We are the on-site, events-only version for Uzbek law and Uzbek systems." |
| "Our CCTV vendor / Hikvision / Ivideon has analytics." | "Generic analytics count people and queues. Service flow needs the plate-to-work-order join and bay-level idle logic. And the Russian dealer-camera model sends video to OEM servers; ours sends only events." |
| "Does Toyota or BYD require something?" | "We found no Toyota or BYD programme that supplies or mandates such tools. UVeye lists Toyota as a partner via investment. We would ask Summit Motors Central Asia and BYD Uzbekistan before rollout." |
| "Is it legal?" | "Personal data is the dealer's, we are the processor. We create no biometric data, so the strict localisation and registration rules do not apply. Signage and a staff notice are part of the product. Image-rights legislation is in progress and we designed for it." |
| "Will technicians accept being watched?" | "Default views are bay-level and team-level; individual views need written notice. The Labour Code protects employee data, so we do not rank people by default." |
| "Why not buy UVeye?" | "UVeye scans cars at the door and is $6-7K per lane per month; it does not measure bay time or waiting. It could be complementary." |

### 6.3 What not to claim

- Any ROI figure as ours. If cited, label as vendor-reported for one store.
- That a plate is not personal data.
- That the 2021 "localisation for everything" rule still applies (it was narrowed in 2026), or that cloud is illegal.
- That the image-rights law is in force.

### 6.4 To verify before the pitch

1. Ask Royal Motors IT: camera brands and count, NVR/VMS, who sees the footage, plate cameras at the gate, existing DMS/CRM, any OEM requirement from Summit Motors/BYD Uzbekistan.
2. Status of the image-rights law (lex.uz, Senate) and the supervising authority for personal data.
3. Whether the VisionFlow AI tender in Kazakhstan was awarded.
4. Whether VEON AdTech or any Uzbek integrator has pitched Royal Motors.
5. Jetson availability via UAE or an NVIDIA distributor; Uzbek retail prices for RTX-based mini PCs.

---

## 7. Gaps ("not found")

- Prices for SKAIVISION, PitCrew, EyeQ, Bayflow, Detect Auto; any Uzbek price per camera for cloud analytics.
- Any service-flow analytics deployment at a dealer in Uzbekistan or Kazakhstan.
- Any Toyota or BYD OEM programme for camera-based service-lane analytics.
- Uzbek retailer price for Jetson Orin; Mediapark/Texnomart/Asaxiy GPU prices.
- Text of the Labour Code CCTV-related provisions (articles 175-180 headings seen, full text not read).
- Final status of the image-rights law; authority name today.
- Independent verification of any vendor claim.

---

## 8. Sources

Competitors and vendors
- SKAIVISION service lane visibility: https://skaivision.lovable.app/solutions/service-lane-visibility
- SKAIVISION case study, Fred Anderson Toyota: https://skaivision.lovable.app/case-studies/fred-anderson-toyota-raleigh
- SKAIVISION FAQ: https://skaivision.lovable.app/faq
- SKAIVISION site: https://www.skaivision.com/
- Wards Auto, Reynolds and SKAIVISION (6 Jan 2025): https://www.wardsauto.com/news/reynolds-and-reynolds-and-skavision-team-for-ai-efficiency/798438/
- Hunter Engineering and SKAIVISION: https://www.motor.com/2024/10/hunter-engineering-announces-integration-with-skaivision/
- JM Family investment in SKAIVISION: https://jmfamily.com/news/skaivision-secures-investment-from-jm-family-enterprises/
- Embedded Vision Summit talk summary: https://www.edge-ai-vision.com/2025/10/skaivision-transforming-automotive-dealerships-with-computer-vision-a-presentation-from-skaivision/
- Motor1, Tucson dealership AI cameras (not read, 502): https://www.motor1.com/news/796504/is-toyota-using-ai-surveillance/
- Leverege PitCrew: https://www.leverege.com/pitcrew ; features: https://www.leverege.com/pitcrew/features ; oil change: https://www.leverege.com/pitcrew/oil-change-service-centers ; NVR blog: https://www.leverege.com/blogpost/bringing-ai-to-your-nvr-how-pitcrew-unlocks-bay-intelligence-from-existing-cameras
- IoT For All, bay intelligence: https://www.iotforall.com/bringing-ai-to-existing-camera-systems-unlocking-real-time-service-bay-intelligence
- Discount Tire Bay Utilization FAQ (surfaced by search): https://kc-dev.discounttire.com/content/dam/dtkc/global/en/store-ops/private/bay-utilization/Bay%20Utilization%20FAQs.pdf
- EyeQ Monitoring cycle time (Digital Dealer, 26 Feb 2026): https://digitaldealer.com/news/eyeq-monitoring-looking-to-bring-ai-precision-to-service-lane-efficiency/169482/
- EyeQ CEO article (18 Mar 2026): https://digitaldealer.com/news/ai-and-video-powered-operational-intelligence-the-next-evolution-of-dealership-performance/169747/
- Bayflow: https://autoflow.com/bayflow/
- Detect Auto: https://www.detectauto.com/about
- UVeye for Toyota dealerships: https://uveye.com/uveye-for-toyota-dealerships/
- UVeye and CARFAX (23 Jun 2026): https://www.repairerdrivennews.com/?p=234879
- UVeye pricing estimate: https://www.thepricer.org/how-much-does-uveye-machine-cost/
- UVeye Canada launch: https://www.autosuccessonline.com/uveye-expands-to-canada/
- Toyota Tsusho and USS/UVeye: https://www.toyota-tsusho.com/english/press/detail/200221_004567.html
- Ravin AI: https://www.ai-online.com/2022/11/ravin-artificial-intelligence-for-vehicle-inspection-comes-to-dealer-floor/
- Spot AI, Don Franklin Auto: https://www.spot.ai/customer-stories/don-franklin-auto
- Spot AI pricing: https://spot.ai/pricing
- Lumana: https://www.lumana.ai/products/visual-intelligence
- Car wash: https://www.carwash.org/car-wash-news/everwash-acquires-omnix-labs ; https://www.parking.net/parking-news/avutec/efficiency-meets-convenience-anpr-for-car-washing
- Patent US12406575: https://patents.google.com/patent/US12406575
- Trassir dealer case: https://trassir.com/cases/smart-dealership-efficiency-with-trassir-video-analytics/

CIS / Central Asia
- LADA/Ivideon (Hi-Tech Mail, 4 Jun 2026): https://hi-tech.mail.ru/news/149005-salony-lada-budut-sledit-za-reakciej-posetitelej-na-avtomobili-s-pomoshyu-ii-kamer/
- AvtoVAZ dealer cameras (ixbt, 7 Oct 2025): https://www.ixbt.com/live/comments/4348848
- VisionFlow AI tender, AstanaHub: https://astanahub.com/ru/tech_task/tz-platforma-avtomatizatsii-servisnykh-zon-visionflow-ai-mvp
- TASS Vision (IT Park): https://www.it-park.uz/en/itpark/news/it-park-ventures-completes-its-first-investment-under-the-1-1-co-investment-mechanism-100-000-in-tass-vision-as-part-of-a-1-3-million-funding-round
- TASS Vision profile (Digital Business): https://digitalbusiness.kz/en/2025-11-03/entrepreneurs-from-tashkent-created-ai-cameras-for-stores-and-raised-1-8m/
- VEON AdTech video analytics (Kun.uz, 15 Feb 2024): https://kun.uz/en/news/2024/02/15/new-perspective-on-business-and-security-veon-adtech-launches-video-analytics-service
- Uzbek national ANPR exposure: https://anhor.uz/news/v-uzbekistane-45/
- Safe City background: https://jamestown.org/?p=88575 ; https://anhor.uz/news/videokameri-v-tashkente-ustanavlivaet-shveycarskaya-agt-international/
- Summit Motors Central Asia (Toyota distributor): https://www.uzdaily.uz/en/uzbekistan-plans-to-open-a-regional-toyota-training-center/ ; https://www.spot.uz/ru/2019/12/02/toyota/
- Toyota Tsusho and IT Park: https://www.it-park.uz/ru/itpark/news/it-park-i-toyota-tsusho-obsudili-perspektivy-raboty-v-uzbekistane
- Royal Motors directory entry: https://president.uz/ru/lists/view/7389 ; Toyota Tashkent City: https://www.yellowpages.uz/teg/oficialnyj-diler-toyota-uzbekistan/tashkent
- BYD Uzbekistan JV: https://gazeta.uz/en/2024/07/10/byd

OEM programmes
- BYD aftersales job listings: https://careers.bydeurope.com/jobs/8191625-service-digitalization-specialist
- BYD China aftersales page: https://www.byd.com/cn/after-sales-service
- Digital Dealer on service-lane programmes: https://digitaldealer.com/news/mykaarma-earns-stellantis-mopar-service-gateway-certification-expanding-connected-service-lane-capabilities/173435/

Uzbek legal
- Law ZRU-547 on lex.uz (current text): https://lex.uz/ru/docs/4396419
- Cross-border list of 49 countries (Kun.uz, 4 Aug 2026): https://kun.uz/en/news/2026/08/04/uzbekistan-approves-49-countries-for-simplified-personal-data-transfers
- Amendment news (Kun.uz, 27 Mar 2026): https://kun.uz/en/news/2026/03/27/uzbekistan-amends-personal-data-law-to-facilitate-global-payment-systems
- Localisation relaxed (Kun.uz, 28 Mar 2026): https://kun.uz/ru/news/2026/03/28/xraneniye-nekotoryx-personalnyx-dannyx-za-predelami-uzbekistana-razresheno
- Legal 500, personal data compliance: https://www.legal500.com/developments/thought-leadership/personal-data-compliance-in-uzbekistan/
- Mondaq, employee privacy: https://www.mondaq.com/privacy-protection/1522174/data-protection-and-employee-privacy-in-uzbekistan
- Labour Code on lex.uz: https://lex.uz/ru/docs/6257288
- Anhor, legal framework for video surveillance (22 Feb 2024): https://anhor.uz/pravovoy-razbor/legislative-framework-video-surveillance-uzbekistan/
- Anhor, biometric data (4 Apr 2024): https://anhor.uz/pravovoy-razbor/biometric-data/
- Spot.uz, filming-consent bill (11 Sep 2025): https://www.spot.uz/ru/2025/09/11/shooting-consent/
- Kun.uz, deputies approve image bill (8 Oct 2025): https://kun.uz/en/news/2025/10/08/deputies-approve-amendments-to-ban-filming-citizens-without-their-consent
- Kun.uz, Justice Ministry draft (17 Mar 2025): https://kun.uz/en/news/2025/03/17/justice-ministry-proposes-ban-on-filming-people-without-consent

Pricing
- CBU exchange rates: https://cbu.uz/ru/arkhiv-kursov-valyut/print/
- Ivideon tariffs (toolfox): https://toolfox.ru/services/s/ivideon
- Trassir Cloud tariffs (toolfox): https://toolfox.ru/services/s/trassir-cloud
- NVIDIA Jetson price increase (CNX Software, 22 Jul 2026): https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101
- OLX.uz RTX 4060 Ti listings: https://www.olx.uz/elektronika/q-rtx-4060-ti/
- NVIDIA and Uzbekistan AI initiative: https://www.it-park.uz/en/itpark/news/the-republic-of-uzbekistan-launches-major-ai-initiative-with-nvidia
- Prom.uz Hikvision: https://www.prom.uz/brand/hikvision/kamery-videonablyudeniya/
