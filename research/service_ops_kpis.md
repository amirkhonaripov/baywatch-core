# Service-flow analytics for Royal Motors: how OEMs score service, workshop time-loss benchmarks, DMS coverage, labour rates, Uzbek capacity pain

Research date: 2026-10-10. Prepared for the hackathon team (jury will include dealer service/sales managers).

**Tag legend.** [F] = fact I read on the page itself (WebFetch). [S] = taken from a search-engine summary only (page not opened or blocked, so treat as less certain). [I] = my inference. Numbers in square brackets after tags, like [12], point to the source list at the end.

**Method caveat.** The search tool is US-based and returns summaries, and many Uzbek/CIS pages are thinly indexed. "Not found" means "I could not find it", not "it does not exist". Many vendor/consultancy numbers below are unverified marketing claims; I say so where it applies.

---

## 0. Who we are pitching to (what is publicly confirmable)

- **Toyota Royal Motors Qushbegi** (Tashkent) officially opened 8 May 2026. Toyota Central Asia describes an "express service zone with panoramic glass" so customers can watch their car being serviced in real time, built to Toyota Motor Corporation standards. Bay count, size and express-service terms: **not stated** [F][1]. Toyota Central Asia's Uzbekistan block shows "4 models, 4 dealers, 5 centers", and its dealer list names Toyota A-Auto, Toyota Royal Motors, Royal Motors Qushbegi, Toyota Otto Motors (Samarkand) and Uzinavto Service Center (service only). No bay counts anywhere [F][1][2].
- **Toyota Tashkent City** (Zargar 3/1, near Tashkent City; directory legal name "Toyota Tashkent City", brand "Oltin Tulpor", domain toyotaroyalmotors.uz) opened 11 Aug 2023; the report says it has "five-star status" (meaning not explained). Service area and bay count: not stated [F][3][4]. Toyota Central Asia's own page calling it "the first official Toyota dealer in Uzbekistan within Royal Motors Group" came from a search summary only [S].
- **BYD Chigatoy** is described on its own site as a new official BYD dealer "part of the Royal Motors group" [F][5]. Its site has an online service-booking form (car data, reason, name/phone, preferred date, one-hour slots 09:00-17:00). **No notifications, no job-progress tracking, no bay count, no prices** are mentioned [F][5][6].
- **Ducati Tashkent**: an official Ducati dealer exists at Zargar 4 (Tashkent City), 3S, service Mon-Fri 9:00-18:00, factory-trained technicians [S][62]. Sources I found do **not** tie it to Royal Motors; treat the group link as something the team knows and I could not verify.
- **BYD Namangan / Samarkand under Royal Motors**: not confirmed. BYD International lists the Samarkand dealer under the operator name "Beni Motors"; BYD's Uzbek contacts page lists a Namangan dealer with no owner shown [S][63].
- **Scale context** (other BYD dealers, for comparison of bay counts): BYD Yunusobod (opened 14 Jul 2023) has a 12-post service centre, 5,100 m2 whole dealership [F][10]; BYD Sergeli (Astana Motors, second branch, Mar 2026) advertises 12 posts, unclear whether per branch or total [F][11]; BYD Autozone has 3 Tashkent service branches (Lunacharsky, Yashnobod, Northern), posts not stated [F][7]. Porsche Tashkent (Apr 2026) has 5 posts in 1,100 m2 incl. one EV-battery post [F][14]. BYD's official distributor says 18 official showrooms/service centres in Uzbekistan (Sept 2024) [F][13]; the bydauto.uz booking dropdown lists 28 dealer locations [F][8].

---

## 1. How OEMs score dealers on service

### 1.1 Toyota

**Headline: no public Toyota dealer scorecard with thresholds exists in anything I could reach.** What is public is the philosophy and certification language:

- **TSM (Toyota Service Management; Toyota India press releases expand it as "Customer Service Marketing")** is a service-development programme "based on the principles of the Toyota Production System", aiming at "maximum efficiency and continuous improvement of all service processes" and "the highest customer satisfaction". Example given: **parts and mechanics for a specific car are selected in advance**. Toyota Center Almaty (Astana Motors group) was the first Kazakh dealer to get the TSM certificate, 2 July 2014, with a Toyota Motor Europe delegation present. The article lists no KPIs or wait-time figures [F][15].
- **Toyota's own diagnosis of lost time (important for our pitch).** Toyota Kirloskar's TSM Advanced / Express Maintenance releases say **"nearly 50%" (2007) / "more than 50%" (2009) of servicing time is spent "in stagnation of the vehicle in between processes"**. The remedy is bay layout redesign so no time is lost searching for equipment, kaizen, and a 1-hour express service (3 people at the bay, booked in advance, same price as normal service). The 2009 release claims servicing time falls "from 17 hours to 1 hour" under EMS and promises a "significant increase in workshop productivity" with no figure. The two releases disagree on when TSM Advanced was introduced (1994 vs 2001) [F][16][17]. These are Indian press releases from 2007/2009, not Central Asia standards [I].
- TSM certification has tiers ("L1" for a Kyiv dealer; "TSM-Kodawari" in Hawaii requires full compliance with all standards); criteria not public [S].
- **Toyota Europe (Ichiban awards)** is judged on an annual customer-satisfaction survey introduced in 2006, covering "welcome to handover" and "handling of appointments to follow-up"; thresholds unpublished [S]. Toyota Australia's 5-Star dealer award used survey results averaged over 12 months [S].
- **Toyota Central Asia** only states generic standards: certified dealers, maintenance "according to international standards", genuine parts [F][1]. No CSI/lead-time targets found.
- A dealer-software vendor claims Toyota's VOC survey contacts customers 3-7 days after a visit and weights **communication quality and scheduling ease**; the same page says no OEM-mandated KPIs are listed and its financial figures are "illustrative, not sourced" [F][19]. Vendor claim, weak.

**What the third-party CSI benchmark (J.D. Power, the standard Toyota is publicly measured against) says about waiting and communication** [F][18] (Vietnam 2018, 29 Jan 2019 release; Toyota ranked #1 with 837):
- Five factors and weights: service quality 24%, service initiation 21%, vehicle pick-up 20%, service advisor 18%, service facility 17%.
- Customers greeted immediately on arrival scored 838 vs 773 for those who had to wait for an advisor.
- Average handover time 22 min (29 in 2017); handovers of 10 min or less scored 846 vs 820.
- **Only 88% were told when the car would be ready (96% in 2017); only 81% said the advisor kept them updated (93% in 2017).**
- 38% waited 3+ days between booking and service (28% in 2017).
- Other J.D. Power items (search summaries): greeting within 2 min matters (Mexico: -30 pts if later), 61% of UK customers had to wait for a greeting (2016), 15+ min handover costs up to 29 pts (Malaysia 2016) [S][64][65].
- [I] Takeaway: the dimensions a manager is scored on (greeting, wait, ready-time promise, status updates, handover time) are exactly the ones camera events could time-stamp.

### 1.2 BYD

**No public BYD dealer service standard with numeric targets found.** Indirect evidence:

- BYD Europe aftersales posting (France): regional manager owns KPIs **"NPS, CCR, ticket resolution time, parts availability, workshop productivity, warranty quality"**, runs dealer performance reviews, drives NPS, defines action plans for underperforming sites, ensures "compliance with BYD service standards", monitors "audit readiness", and rolls out "operational standards (TIS, DMS, CRM, warranty procedures, technical communication)"; also "secure achievement of APV bonus conditions". No numeric targets [F][20].
- Canada posting lists dealer KPIs service retention, CSI, repair efficiency, warranty performance, parts sales, workshop productivity, plus "service capacity planning" and capacity models using "workshop throughput, technician headcount, bay allocation" [S][66].
- BYD Italy interview: "KPI systems" to evaluate service quality (no names/targets); dealers must use BYD's standard diagnostic equipment and complete strict training; each service point must have at least two courtesy cars; goal of a service point within 30 minutes' drive; all dealers are 3S; BYD provides dealers a DMS that "collects data in real time" and BYD works with individual dealers when data is weak; parts: 98% of references centrally available, but 5-6 working days delivery vs a 48-hour goal [F][21].
- BYD Europe's dealer-admission page lists only partner criteria (finances, experience) and process; **no service standards or KPIs** [F][25].
- Systems: BYD Europe/Belgium/Italy postings require DMS, LMS and TIS "in every dealer and service shop"; a "Service Digitalization Specialist" role integrates DMS, SAP, WMS, OMS [S][67].
- **Where BYD service is visibly under strain (relevant to what BYD managers worry about):**
  - Australia: average service-booking wait about **7 days**, target 4-5 days; 95% local parts fill rate; COO admitted after-sales "needed vast improvement" 6-9 months earlier (Sept 2026) [F][22]. In 2023 the distributor said it underestimated demand and its service network could not keep up [F][23].
  - Korea: average repair period for four BYD models **9.8 days vs 5.3 for domestic and 8.2 for imported EVs** (H1 2026, Insurance Development Institute data); of ~20 service hubs only 5 do accident repair; parliamentary audit scheduled (Oct 2026) [F][24].
  - China: 2025 collapse of BYD's Shandong dealer group; complaints about 4S "deflection" [S][68].
- **In Uzbekistan** the BYD Astana Motors network advertises (sponsored article, Apr 2026) day-before reminders by a service manager, online booking, and waiting-room screens that "broadcast the progress of service work" [F][9]. The official BYD Uzbekistan booking form has no time slots and only promises a callback ("we will contact you") [F][8].

---

## 2. Industry benchmarks on workshop time loss

### 2.1 Definitions (the terms are used inconsistently; fix them before quoting)

[F][26] (vendor glossary citing NADA conventions):
- **Efficiency** = sold (flat-rate) hours / clocked hours (speed vs the labour-time guide; >100% is normal).
- **Productivity** = clocked hours / attended hours (share of the paid day clocked onto jobs).
- **Overall labour utilisation** = efficiency x productivity = sold hours / attended hours (worked example 130.8% x 81.3% = 106.3%).
- **Workshop loading** (forward-looking) = booked hours / available technician hours for a future day; common target 80-85% [F][27].
- Other vendors/consultancies define the same words differently (e.g. Autologica: efficiency = invoiced / worked hours; "productivity" = assigned vs available hours) [F][45][46]. Omnetic notes labour utilisation is "especially prone to inconsistent denominators" [F][28]. Russian shop-floor usage: "загрузка поста" = sold normo-hours / available post time; the training-textbook parameters КИСП 0.95 (1 shift) and unevenness factor 1.15 are undated textbook values [S][69].

### 2.2 Typical values

| Metric | Value | Where | Tag |
|---|---|---|---|
| Productivity target | 85-87.5% (15-20% of time is non-labour) | "NADA suggests" via vendor glossary | [F][26] (secondhand) |
| Efficiency, top NADA 20 Group | 125% vs factory manual, 135% vs non-factory | same | [F][26] |
| Efficiency "flag hours / clock hours" | at least 100%, 120% preferred | Digital Dealer 2014 | [F][61] |
| Autologica efficiency alert | below 135% = "low", act | vendor | [F][45] |
| Average US dealer tech output | ~80% (6.4 flat-rate hrs per 8 clock hrs), trainer's estimate | Digital Dealer 2016 | [F][30] (anecdotal) |
| Deloitte ProfitFocus (AU/NZ) performance index (productivity x efficiency) | 95-105% (NZ 2025), 100-105% (AU 2026), best practice = top 30% | summary | [S][70] |
| UK aftermarket workshop utilisation | **fell from 90.4% (2002) to 79.9%**, "idle time of around 20% is now the norm" (ASE survey, article Nov 2010; year of the 79.9% not named, probably 2009) | autobiz.ie | [F][29] |
| Russian dealers, "продуктивность сервиса" (employee working-time use) | **105% in 2021, +9 pp**; some regions near 120%; avg repair order almost 4 normo-hours; ASE study of 10 brands in 11 regions. Formula not given, so >100% implies normo-hours over paid time | auto.ru | [F][31]; interpretation [I] |
| Independent shops, billed hours per tech per 8-h day | **4.2 h** (AIA Canada conference, June 2026) vs ~9 h benchmark; another estimate 5.2 h/day (consultant opinion, July 2026); 4.4 h (2015, AIA Canada) | various | [F][32][33]; [S] for 4.4 |
| Forecasting rule of thumb | loading target 80-85%; remaining 15-20% absorbs walk-ins/carry-overs | vendor | [F][27] |

Caveats: independent-shop and fleet "wrench time" numbers (25-35%; one fleet-vendor split of wrench 30%, parts wait 15%, admin 13%, idle 8%) are not dealership data [S]; I would not quote them to dealers.

### 2.3 What managers believe vs what is measured

No rigorous manager-estimate-vs-time-study comparison found. Closest evidence:
- A powersports (motorcycle) dealer reported **technician efficiency 100% but productivity only 29%**, with service gross profit 44% vs ~70% needed; consultants flagged that technician time was not tracked on repair orders and a time clock was not yet installed. Directly relevant for Ducati [F][34].
- "Workshop utilisation can look high while jobs wait for parts"; denominators vary, so reported figures are not comparable [F][28].
- Vendor: "half of all bad efficiency data is a clocking problem" (forgetting to clock off) [F][26] - a vendor claim, not a study.
- DMS-vendor example: a dealership invoicing 200 of 400 paid technician hours (50%), later 450 invoiced (112.5%) [F][45]; another example 160 of 400 hours [S].
- Carlisle & Co. survey (~9,000 US dealer technicians, 2013/14): 43% of repair orders need additional clarification from the advisor, costing each tech about 30 min/day (about $90k/yr lost for a 12-tech dealer at $60/h); technicians say a third of customers get unrealistic wait times, advisors say they are accurate 83% of the time [S][71].

### 2.4 Common causes of bay/technician idle time

- Parts delays, poor dispatch, empty diary, bay not ready (low productivity); skill gaps, tooling, unrealistic times (low efficiency) [F][26]. Declined work never reaches the diary; overbooking carry-over blocks bays; a single diagnostics-certified technician can run at 140% while overall loading is 70% [F][27].
- Parts delays, slow dispatch, long verbal exchanges between technicians and advisors, waiting for next job, clarifying approvals, labour-guide errors, too few service advisors ("technician downtime is far more expensive than adding administrative support") [F][32].
- Toyota's own list: stagnation of the car between process steps and hunting for equipment [F][16][17].
- Cox Automotive 2021 (US, 529 dealers): 58% cite manufacturer parts delays as a top frustration, 45% finding technicians, 57% not fully staffed; nearly 1 in 4 consumers say service visits take longer than expected [F][35].
- Vendor-only claims: some "1 hour diag" jobs run up to 15 hours of bay time; cars moved in and out of a bay up to 7 times [F][38].
- Russian/CIS: loss of up to 10% of turnover to missed customer reminders, mechanics under 140 normo-hours per month as a sign of inefficiency, cars waiting for parts in the workshop (all unquantified or author estimates, from a consultant promoting courses) [F][57].

### 2.5 Quantified before/after cases (all vendor-published, none independently verified)

- **SKAIVISION at Fred Anderson Toyota, Raleigh** (~250 vehicles/day, ~40 service/parts bays, ~30 existing cameras; monitors technician downtime, greeting time, vehicle time in bay, cycle times; live Jan 2024): basic service time "75-90 min to under 60 min" (the page says customer-reported, not a measured baseline), ~$1M extra annual revenue; **no technician-downtime before/after figures** [F][36]. Reynolds & Reynolds invested in SKAIVISION (Nov 2024); features include cycle-time analysis, critical delay alerts, customer-facing wait-time estimates [F][37]. Obstacles named by the vendor: occlusion, glare, hardware/network issues [S].
- **Autoflow Bayflow** (cameras + plate matching to work orders): "25%+ increase in bay efficiency", "29 minutes saved on the average job", "15%+ technician efficiency"; no method, sample or named customer [F][38].
- **Leverege PitCrew**: average service time -13%, customer wait -18%, +8 invoices per store per day; no methodology [F][39].
- **Toyota Ireland / Infomedia Superservice Triage** (digital vehicle health check at 42 sites): preparation time from up to 90 min to about 15 min [F][40].
- **Regional neighbour (live demand signal):** an Astana Hub tech task (customer Smart Tech, Kazakhstan, proposals until 20 Apr 2026, budget 1.5M, probably tenge) asks for a "VisionFlow AI" MVP: computer vision on RTSP streams to detect free/occupied service zones, plate recognition (>= 90%), detect process stages from video, predict estimated departure time, re-schedule the queue, notify via web push/Telegram, bay status in under 5 s, 50+ cameras; 10 applications at the time of reading [F][41].
- Toyota Ireland and SKAIVISION show OEM/dealer-software incumbents are already moving here [I].

---

## 3. What DMS/software already tracks

### 3.1 CIS / Russian-origin products

- **Альфа-Авто (1C-Rarus; ed. 5 "Проф", ed. 6 "Корп")**: product page lists "планирование загрузки автосервиса и сотрудников", advance repair booking, labour-time norms by model and trim, "начисление выработки по сотрудникам", a **separate paid add-on "Учет рабочего времени"** (48,300 RUB for ed. 5; 51,300 RUB for ed. 6), WhatsApp/Telegram integration. **SMS is not mentioned** on that page [F][42]. A Rarus case says the system records mechanics' work start/end times, breaks and reasons orders are paused; how the times are captured (manual, terminals, barcodes) is **not stated**, and no results or figures are given [F][43]. A 2012 forum review (search summary) said the built-in SMS module did not work on its own and CRM was weak [S].
- **АвтоДилер (autodealer.ru)**: post/slot booking that avoids overlaps; timesheet in the salary module but **no per-job time tracking described**; SMS Center integration for reminders and mailings; Telegram/WhatsApp channels; inspection acts and diagnostic checklists but **no photo-at-reception feature described**; norm-hours for 80+ brands as an add-on [F][44].
- Other small tools: a "Программа для СТО" with a mechanic mobile app for changing order status and attaching photos at reception; Tezarius logs a percentage complete per mechanic, not hours; Bitrix24 car-service solution sends automatic task notifications [S][72].
- **Pattern** [I]: everything above depends on a person tapping start/stop/status. None of the pages describes automatic detection of where a car physically is or why a bay is idle.

### 3.2 Western/OEM DMS

- **Autologica** "Job Clock": technicians log start/end of each operation on a touch screen including non-invoiced time; "Technician Panel" shows in real time who is working or idle and delayed/cancelled tasks [F][45]. Its customers include Toyota dealers in several countries, with shift data sent to Toyota [S].
- **Toyota** chooses DMS per country: Germany (TIS "Topco", provider change 2017), UK (CDK Autoline Drive, announced 2015), Belgium (Incadea, 75 sites) [S][73]. No Toyota-wide DMS and no Toyota Central Asia DMS found. Toyota Ireland uses a digital vehicle-health-check tool integrated with Toyota's customer app [F][40].
- **BYD**: DMS/LMS/TIS rollout to all dealers is a regional-manager duty; the Italian interview describes a DMS collecting data in real time. Nothing on technician clocking or job cards [F][20][21].
- **UzAuto Motors**: SAP ERP/CRM since 2021 (SAP award April 2025); no dealer-level DMS identified [S][74].

### 3.3 Uzbekistan specifically

- **No DMS identified for Royal Motors, BYD Chigatoy, Toyota Tashkent City/Qushbegi or Ducati Tashkent** (not found).
- 1C is used in Uzbek dealer groups: ROODELL LLC (Kia dealer operator) advertised a Tashkent "Аналитик 1С (Лизинг)" vacancy (archived 4 Sep 2026); the description was hidden when I fetched it [F][47]. A search summary claimed an ADM GLOBAL posting named "Альфа-Авто 6 (Автосервис + Автозапчасти)"; I could not confirm that [S]. 1C-Rarus has a Tashkent branch [S].
- Customer-visibility features already present in the market: BYD Astana Motors day-before reminders and waiting-room progress screens [F][9]; Toyota Qushbegi panoramic glass express zone [F][1]; online booking at Chigatoy/BYD Autozone/BYD Uzbekistan (no status tracking) [F][5][7][8].

### 3.4 Is manual clocking reliable?

- No audit-type study found. Evidence is anecdotal/vendor: Powersports case (efficiency 100% vs productivity 29% because time not tracked) [F][34]; vendor claim that half of bad efficiency data is a clocking problem [F][26]; clocking software documentation notes forgotten clock-offs need manager correction [S]; Russian sources describe notebook booking causing double-booked posts and "approximate" repair duration [S][75].
- Pay-by-flat-rate creates incentive to under-clock or mis-clock [I].
- [I] Clocked/booked time answers "who billed what", not "why was the car waiting".

---

## 4. Labour rates in Uzbekistan

**Not found: I could not find a published price per standard labour hour (нормо-час / norma-soat) for any official dealer or independent service station in Tashkent (Toyota, BYD, Chevrolet/UzAuto or other).** Multiple Russian/Uzbek/English searches returned only generic methodology. What exists:

- OEMs do not publish norm-hour rates; the OEM supplies time norms to dealers only and each service sets its own rate (Chevrolet Russia 2010 statement; DRIVE2 2020) [S][76].
- Whole-job ranges in sums, undated per-hour basis (Ustabor, 2025 blog, no dealer/independent split): diagnostics 150,000-500,000; brake-pad replacement 200,000-700,000; engine repair from 2,000,000-10,000,000 sums. The article says dealers are "dearer, but suitable for warranty cars and complex electronics" with no percentage [F][55].
- Unsourced blog (Toyota Base, mixed-date): Corolla 10,000 km service about $150, 20,000 km about $220, Prado first service about $300, RAV4 Hybrid +20%; claims online booking cuts waiting from 2-3 hours to 30-40 minutes. Treat as indicative only [F][54].
- FX for conversion: about 11.8-12.1 thousand UZS per USD, early Oct 2026 [S][77].
- Neighbour market, no rate: Kazakhstan service prices rose y/y by 24.2% in Almaty and 18.9% in Astana (Aug 2025) [F][58].
- BYD Autozone (price list PDF dated 2026-09-08) and Chigatoy pages list only vehicle prices; Chigatoy says service "at a reasonable price" with prices informational [F][5][7]. I did not open the PDF.

**Bays (постов) at Tashkent dealers:** see section 0. BYD Yunusobod 12, BYD Sergeli 12 [F][10][11]; Porsche 5 [F][14]. Toyota Royal Motors sites: not found. Regional references: Toyota Esil (Astana) 760 m2 with 10 posts [S]; BYD sites in Malaysia 4-16 bays, Philippines 6, Nepal 8, Korea 5 [S][78].

**Illustrative arithmetic only** [I]: a 12-bay site working 8 h x 26 days has about 2,500 bay-hours a month (assumption on schedule); each 1 percentage point of utilisation is about 25 bay-hours a month, so a 10-point recovery is about 250 hours. Multiply by the sold-hour rate once the dealer gives it. I deliberately did not assume a rate.

---

## 5. Evidence of waiting/capacity pain at Uzbek dealer service centres

**Strongest Uzbek evidence is structural, not queue data.**

- **President, 16 Sep 2025 (Tarakkiyot zone, Namangan):** servicing of foreign brands is "insufficiently developed"; car ownership and factories grow but service infrastructure "is not keeping pace"; instructed new service centres in the regions and training of qualified specialists. No counts given [F][48].
- **Competition Committee (reported 28 Oct 2025):** over **350 citizen complaints about car service since the start of 2025**, 60+ sellers inspected; **five showrooms had no service centre**; 13 cases of component warranties shorter than the vehicle warranty (e.g. BYD from one dealer: 8 yr/150,000 km vehicle but starter battery 12 months/20,000 km); warranty voided if serviced outside the recommended list; dealers lack a unified warranty-status database [F][49].
- **Jan 2026 consumer-appeal results** (8 cases): vehicle defects (4), delivery delays (2, a Cobalt about 10 months), one warranty-cost case, and a Chery malfunction not repaired - these are sales/warranty complaints, not service-queue data [F][50].
- **Demand shock:** EVs were 46,749 of 65,876 passenger-car imports in Jan-Jul 2026 (about 71%, my calculation) [F][51]; Q1 2026 new-car sales 91,088 (+20.9% y/y), BYD share 8.6% [F][52]. Individual owners' passenger cars about 4.55 million (Apr 2026) and about 2,030 listed service stations (scraped directory) imply roughly 2,240 cars per listed station (my division, rough) [S][79].
- **EV service:** a 2022 interview-based article quotes Megawatt Motors saying sellers show "no appetite" to invest in EV service and infrastructure [S][80]. An academic paper names infrastructure, skill gaps and inconsistent service quality as key challenges, with no numbers [F][53].
- **Technician shortage in Uzbekistan:** no quantified data found. Qualitative: an Uzbek delegation visited the Minsk automechanic college (May 2026) [S]. Russian shortage articles exist but are not applicable.
- **Direct complaints about waiting at Royal Motors sites:** none found. Toyota Royal Motors Qushbegi on Yandex Maps: 4.9 from 20 ratings, 5 reviews, none mention waiting or queues [F][56]. For BYD Astana Motors, reviews from about a year earlier say service is good but "not all parts are always in stock" [S][81].
- **Honest bottom line:** I found **no published, measured evidence of multi-day queues or bay idle time at Uzbek dealer service centres.** The problem is plausible and consistent with regional BYD patterns (Australia 7-day waits, Korea 9.8-day repairs), regulator attention and the EV import surge, but any claim to the jury that Royal Motors "has queues" must come from their own numbers.

---

## 6. Implications for our pitch

1. **Open in Toyota's language, not ours.** Toyota's own TSM material says about half of servicing time is cars "stagnating between processes" and its remedy (pre-assigned parts and mechanics, kaizen, one-hour express service) is about removing waiting between steps [F][15][16][17]. Service-flow analytics measures exactly that stagnation, per car, per step. Qushbegi's glass express zone shows Royal Motors already believes in visible process.
2. **Position as a sensor layer that feeds their DMS, not a DMS replacement.** DMS/booking/post-planning/SMS/clock modules exist (Alfa-Auto, AutoDealer, BYD's DMS/TIS, Toyota's national DMS) [F][42][44][21]. What none of the pages show: automatic detection of car location, an objective bay-occupied timeline, and a reason for idle time. The redundancy risk is highest for "booking", "SMS reminder" and "manual clocking"; lowest for "ground-truth timestamps + idle-reason + cross-site comparison".
3. **Speak to BYD managers in BYD's KPI vocabulary:** NPS, CCR, ticket resolution time, workshop productivity, repair efficiency, parts availability, audit readiness [F][20]. BYD HQ mostly wants dealer-reported data in DMS/TIS [I]; offer an export so the camera numbers can be the audit trail. BYD managers also know service waits are BYD's weak point elsewhere (Australia, Korea) [F][22][24].
4. **Customer-informed feature is differentiated but contested.** J.D. Power shows status updates and ready-time promises are measurable satisfaction drivers (81% updated, 88% told ready time in Vietnam 2018) [F][18], and BYD Astana Motors is already advertising screens plus day-before reminders [F][9]. Our edge: automatic, no advisor effort, tied to real car location [I].
5. **Competitors exist abroad; none seen in CIS.** SKAIVISION (Reynolds-backed), Autoflow Bayflow, PitCrew, and a Kazakh VisionFlow tender [F][36][37][38][39][41]. I found no commercial CV service-flow product in Uzbekistan or Russia (search not exhaustive) [I]. Expect questions on "why not buy SKAIVISION": local plates, Telegram, price, on-prem/privacy, Russian/Uzbek UI.
6. **Be careful with claimed ROI.** Vendor cases (75-90 to under 60 min, 25% bay efficiency, 29 min/job, -13% service time) have no disclosed baseline or method [F][36][38][39]. Safer framing: we will produce the baseline (idle time by reason, waiting by step) in a 1-2 week pilot; the dealer decides which lever to pull. Cite published independent-ish anchors only: UK utilisation fall to 79.9% and idle about 20% [F][29]; US dealer tech output about 80% (anecdotal) [F][30]; NADA-style productivity 85-87.5% and efficiency 125-135% targets via a vendor [F][26]; Russian dealer "service productivity" 105% in 2021 [F][31].
7. **Use the clocking gap as the core argument.** Efficiency 100% vs productivity 29% in a powersports dealer happened because time was not clocked [F][34]; vendors say clocking errors are the largest source of bad efficiency data [F][26]. Camera timestamps give an independent second source [I].
8. **Define terms on the first slide** (efficiency, productivity, utilisation, loading) with the Ventavid/NADA definitions [F][26] so the jury's own numbers can be compared. Dealers often use "загрузка постов" loosely.
9. **Questions to ask the jury (they hold the numbers we could not find):** normo-hour price (sums) and how warranty hours are paid; number of bays at Qushbegi/Tashkent City/Chigatoy; which DMS and whether technician clocking is on; how many cars per day; average booking wait; what Toyota Central Asia/BYD Central Asia scores them on; whether CCTV covers bays and reception; employee-monitoring comfort.
10. **Privacy/labour-relations risk** [I]: technician-level tracking raises employee-consent issues; vendors stress "no facial recognition, event data only" [F][38]. I did not research Uzbek personal-data law; check before promising technician-level views. Offer bay- and car-level metrics first.

### Things I could not find (be honest about these on stage)
- Any numeric Toyota or BYD service KPI target (CSI thresholds, greeting time, lead time, utilisation targets) for CIS/Uzbekistan.
- Labour hour rates at any Tashkent dealer or independent shop.
- Bay counts for Toyota Tashkent City, Qushbegi, BYD Chigatoy, Ducati Tashkent.
- Which DMS Royal Motors uses.
- Quantified technician shortage or service-queue data for Uzbek dealers.
- A study comparing managers' perceived utilisation with measured utilisation.

---

## Source list

**Dealers / Royal Motors / Uzbekistan (fetched [F])**
1. https://www.toyota-centralasia.com/en/
2. https://toyota-centralasia.com/en/dealers
3. https://uznews.uz/ru/posts/67124
4. https://www.yellowpages.uz/teg/oficialnyj-diler-toyota-uzbekistan
5. https://byd-rm.uz/service
6. https://byd-rm.uz/
7. https://bydzone.uz/ru/servis/
8. https://bydauto.uz/ru/service-request
9. https://www.gazeta.uz/ru/2026/04/21/byd/
10. https://www.gazeta.uz/ru/2023/07/18/byd/
11. https://www.spot.uz/ru/2026/03/21/byd-sergeli/
12. https://www.spot.uz/ru/2026/02/03/byd/
13. https://www.uzdaily.uz/ru/dilery-byd-ne-budut-okazyvat-garantiinoe-obsluzhivanie-serykh-avtomobilei/
14. https://www.gazeta.uz/ru/2026/05/01/porsche/

**OEM standards / KPIs**
15. https://kapital.kz/economic/31422/toyota-tsentr-almaty-vnedril-novyye-standarty-servisa.html
16. https://www.toyotabharat.com/news/2007/toyota-kirloskar-motor-introduces-global-benchmark.html
17. https://www.toyotabharat.com/news/2009/toyota-kirloskar-motor-takes-customer-service-to-a-higher-level.html
18. https://www.dev.jdpower.com/business/press-releases/2018-vietnam-customer-service-index-csi-study/
19. https://numa.com/blog/oem-csi-requirements
20. https://careers.bydeurope.com/jobs/7274376-aftersales-area-service-manage-paris-north-region
21. https://www.sicurauto.it/speciali/aftersales-brand-emergenti/byd-aftersales-lattenzione-al-cliente-viene-prima-di-ogni-cosa/
22. https://www.mynrma.com.au/open-road/news/2026/byd-targets-shorter-service-waits
23. https://goauto.com.au/news/byd/rapid-byd-uptake-led-to-customer-service-drama/2023-10-23/92243.html
24. https://www.starnewskorea.com/en/business-life/2026/10/07/2026100710493694762
25. https://www.byd.com/eu/dealer-admittance.html

**Benchmarks / time loss**
26. https://www.ventavid.com/glossary/technician-efficiency
27. https://www.ventavid.com/glossary/workshop-loading
28. https://www.omnetic.com/insights/dealership-kpi-framework
29. https://www.autobiz.ie/home/dealer-workshops-idle-time-rises
30. https://digitaldealer.com/news/aint-lefty-productive-enough/78419/
31. https://auto.ru/mag/article/statase21year/
32. https://www.autoserviceworld.com/canadian-shops-struggle-to-bill-an-eight-hour-day-why/
33. https://www.aapexshow.com/blog/auto-technician-productivity/
34. https://powersportsbusiness.com/features/2011/09/05/track-tech-time-closely-for-service-profit-boost/
35. https://www.coxautoinc.com/insights/2021-cox-automotive-service-industry-study-parts-shortages-staffing-concerns-hold-back-dealership-service-departments/
36. https://skaivision.lovable.app/case-studies/fred-anderson-toyota-raleigh
37. https://www.autoremarketing.com/?p=70219
38. https://autoflow.com/bayflow/
39. https://www.leverege.com/blogpost/uncovering-the-hidden-costs-of-underutilized-service-bays
40. https://www.infomedia.com.au/customers/toyota-ireland/
41. https://astanahub.com/ru/tech_task/tz-platforma-avtomatizatsii-servisnykh-zon-visionflow-ai-mvp

**DMS**
42. https://www.1cbit.kz/1csoft/1s-alfa-avto-avtoservis-avtozapchasti/
43. http://www.retail.ru/rbc/pressreleases/1s-rarus-1s-rarus-avtomatiziroval-uchet-rabochego-vremeni-v-kompanii-remeks-s-pomoshchyu-alfa-avto/
44. https://autodealer.ru/
45. https://blog.autologica.com/en/7-aftersales-indicators-with-autologica-dms-part-ii
46. https://blog.autologica.com/en/steps-to-optimizing-efficiency-and-productivity-in-your-dealerships-service-department/
47. https://hh.ru/vacancy/135937652

**Uzbekistan pain / market / prices**
48. https://www.gazeta.uz/ru/2025/09/16/foreign-cars/
49. https://www.spot.uz/ru/2025/10/28/car-dealers/
50. https://kun.uz/news/2026/01/05/avtosalonlar-va-servisdagi-muammolar-boyicha-murojaatlar-natijalar-elon-qilindi
51. https://www.uzdaily.uz/en/electric-vehicles-lead-uzbekistans-car-imports-in-2026/
52. https://abauzbekistan.uz/en/news/press-releases/uzbekistan-car-market-review-q1-2026-result.htm
53. https://zenodo.org/records/14207897
54. https://toyotabase.vercel.app/articles/t/ta/tashkent-tojota-tsentr.html
55. https://auto.ustabor.uz/ru/blog/remont-avto-v-tashkente-polnoe-rukovodstvo-po-vyboru-avtoservisa
56. https://yandex.uz/maps/org/royal_motors/190369714981/
57. https://teletype.in/@vagminsk/poteri
58. https://finratings.kz/news/6687-gde-deshevle-remontirovat-avto-v-kazakhstane/amp
61. https://digitaldealer.com/news/service-by-the-numbers-seven-benchmarks-of-success/57572/

**Search-summary-only [S] sources (pages blocked or not opened)**
62. https://centrum-air.com/en/blogs/ducati-uzbekistan and https://www.ducati.com/be/fr/dealers/tashkent/tashkent-city-shayxontohur-distric (Ducati Tashkent)
63. https://www.byd.international/city/samarkand ; https://bydauto.uz/ru/contacts (BYD Samarkand/Namangan)
64. https://www.jdpower.com/business/press-releases/jd-power-2016-malaysia-customer-service-index-csi-study (handover 15 min; blocked 403)
65. https://europe.jdpower.com/system/files/legacy/assets/2016096_uk_csi.pdf (UK greeting; redirected)
66. https://ev.careers/jobs/678958516/apply (BYD Canada aftersales KPIs)
67. https://careers.bydeurope.com/jobs/8191625-service-digitalization-specialist
68. https://cleantechnica.com/2025/05/29/big-byd-dealer-network-collapses-what-is-going-on/
69. https://eios.kgsxa.ru/mod/resource/view.php?id=29942 (КИСП textbook)
70. https://www.deloitte.com/content/dam/assets-zone1/au/en/docs/industries/2026-Dealership-Benchmarks-Report-Australia.pdf
71. https://www.businesswire.com/news/home/20140618005854/en/Carlisle-Company-Reveals-Key-Findings-Annual-Survey (403; also https://www.autosuccessonline.com for same figures via summary)
72. https://a2is.ru/catalog/programmy-dlya-kompanij-po-remontu-avto/programma-dlya-sto ; https://a2is.ru/catalog/programmy-dlya-kompanij-po-remontu-avto/tezarius ; https://www.bitrix24.ru/solutions/car-service/
73. https://am-online.com/news/2015/1/6/toyota-and-cdk-global-tie-up-brings-car-dealers-new-management-system/37552 ; https://www.autohaus.de/nachrichten/autohandel/software-fuer-toyota-haendler-tis-bekommt-neuen-eigentuemer-2724352
74. https://uzavtosanoat.uz/en/news_show/uzauto_motors_sap_nominatsiyasida_golib_boldi1744881201 ; https://www.spot.uz/ru/2021/03/03/uzauto-sap/
75. https://habr.com/ru/articles/716478/ (owner anecdote on notebook booking, via search summary)
76. https://www.zr.ru/archive/reis/2010/04/firmiennoie-bliudo ; https://www.drive2.ru/o/b/630814011184651547/
77. https://www.norma.uz/rate_dinamic ; https://top.uz/uz/uslugi/remont-i-tekhnicheskoe-obsluzhivanie-avtotransporta
78. https://autobuzz.my/2023/05/09/byd-opens-3s-centres-in-ara-damansara-glenmarie-cheras/ ; https://www.autoindustriya.com/auto-industry-news/byd-greenfield-mandaluyong-is-officially-open.html ; https://www.b360nepal.com/detail/23085/cimex-inc-inaugurates-2-service-centres-in-kathmandu-valley-expands-byd-service-network-2024-Jul-28-333600 ; https://biz.heraldcorp.com/article/10777717
79. https://rentechdigital.com/smartscraper/business-report-details/list-of-car-service-stations-in-uzbekistan ; https://yuz.uz/en/news/kolichestvo-avtotransportnx-sredstv-prinadlejaix-naseleniyu-uzbekistana-prevsilo-49-mln (the 4.55 million passenger-car and 2,030-station figures come from search-summary text, not a page I opened)
80. https://www.intellinews.com/evs-in-uzbekistan-bring-problems-and-prospects-250931/
81. 2GIS/Yandex review snippets for BYD Astana Motors ("service good, but not all parts always in stock", about a year old), seen only inside a search summary; listing entry point https://2gis.uz/tashkent/search/Byd%20service
