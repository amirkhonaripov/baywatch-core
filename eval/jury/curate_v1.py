"""Labelling record for jury set v1 (how eval/jury/jury.jsonl was produced; kept for review and reproducibility).

    python -m eval.jury.select --out <dir>                  # candidates.jsonl + 1280 px thumbnails
    python eval/jury/curate_v1.py <dir-containing-jury_cand> eval/jury/jury.jsonl

Labels below were assigned by the Lane G agent looking at every image (contact sheets) and at one frame per video;
single annotator, no second opinion. "hard" marks inputs where a reasonable system could fail (toys, drawings,
trains, tiny or partial vehicles, in-car views). Candidate pool: candidates.jsonl from select.py with cid c000-c167.
"""
import json, sys
from pathlib import Path
S = Path(sys.argv[1]); OUT = Path(sys.argv[2])
rows = {json.loads(l)['cid']: json.loads(l) for l in open(S / 'jury_cand/candidates.jsonl')}
T, F = True, False
# cid: (category, scene, acceptable_scenes, vehicle, plate_visible, inspection_target, hard, notes)
L = {
"c000": ("uz_car","street",["street","gate"],T,T,F,F,"Chorsu bazaar, Tashkent: several parked cars, Uzbek plates small but legible; people"),
"c002": ("uz_car","street",["street","closeup"],T,T,T,F,"Lamborghini parked in Tashkent; Uzbek plate legible"),
"c003": ("uz_car","street",["street","closeup"],T,F,T,F,"Chevrolet Cobalt police car; plates blurred by the uploader"),
"c004": ("uz_car","street",["street","closeup"],T,T,T,F,"Daewoo Labo pickup, Tashkent; Uzbek plate legible"),
"c005": ("uz_car","street",["street","closeup"],T,F,T,F,"wrecked Moskvich-412 shell without wheels (heavy damage)"),
"c008": ("uz_car","street",["street","closeup","gate"],T,T,T,F,"Chevrolet Captiva at night, phone photo with motion blur; Uzbek plate partly legible"),
"c009": ("uz_car","street",["street","closeup"],T,T,T,F,"Chevrolet Cobalt, Tashkent; Uzbek plate legible"),
"c016": ("uz_car","street",["street","gate","closeup"],T,F,T,F,"Chevrolet Cobalt rear in front of a yard gate; plate masked by the uploader"),
"c020": ("uz_car","gate",["gate","street"],T,F,F,F,"car park in front of a building, Uzbekistan; plates not legible"),
"c022": ("brand_car","street",["street","closeup"],T,T,T,F,"BYD Song Plus, Almaty; Kazakh plate legible"),
"c025": ("brand_car","street",["street","closeup","unknown"],T,F,T,T,"Toyota Land Cruiser on the Uyuni salt flat with its mirror reflection (unusual scene)"),
"c027": ("brand_car","street",["street","closeup"],T,T,T,F,"BYD Han EV; German plate legible"),
"c028": ("brand_car","street",["street","closeup"],T,T,T,F,"Toyota Camry XV70, China; blue Chinese plate legible"),
"c029": ("brand_car","street",["street","closeup","bay"],T,F,T,F,"Toyota Land Cruiser 300 in a showroom; no plate"),
"c030": ("brand_car","street",["street","closeup"],T,T,T,F,"BYD Song Plus, Uzbekistan; plate legible"),
"c001": ("tricky","unknown",["unknown"],F,F,F,T,"Soviet postage stamp (Tashkent): printed drawing with tiny cars"),
"c038": ("tricky","unknown",["unknown"],F,F,F,T,"dyno power chart with a small motorcycle picture"),
"c068": ("tricky","unknown",["unknown"],F,F,F,T,"1930s Ford advertisement: drawing of a car on a lift plus text"),
"c111": ("document","unknown",["unknown"],F,F,F,F,"scanned newspaper column (document)"),
"c141": ("tricky","unknown",["unknown","closeup","street"],F,F,F,T,"1:18 scale model of a Ferrari 250 GTO on white (toy, not a vehicle)"),
"c151": ("tricky","unknown",["unknown"],F,F,F,T,"toy shop window with many small car models"),
"c153": ("tricky","unknown",["unknown"],F,F,F,T,"pen drawing of an early car on a postcard"),
"c032": ("motorcycle","motorcycle",["motorcycle"],T,F,F,F,"Ducati Panigale V2 outdoors"),
"c033": ("motorcycle","motorcycle",["motorcycle"],T,F,F,F,"Ducati Monster 1200 S"),
"c034": ("motorcycle","motorcycle",["motorcycle"],T,F,F,F,"Ducati Scrambler on a show stand, other bikes behind"),
"c036": ("motorcycle","motorcycle",["motorcycle","street"],T,F,F,F,"scooter parked on a Tashkent street"),
"c041": ("motorcycle","motorcycle",["motorcycle"],T,F,F,F,"Ducati Panigale V4 R at EICMA with a crowd (faces)"),
"c043": ("motorcycle","motorcycle",["motorcycle","street"],T,F,F,F,"Ducati Scrambler in a car park, cars behind"),
"c048": ("street","street",["street"],T,F,F,F,"Almaty street: truck, cars, scooter courier"),
"c050": ("street","street",["street","gate"],T,F,F,F,"Almaty avenue with a row of parked cars"),
"c051": ("tricky","street",["street","unknown"],F,F,F,T,"Bishkek railway from above: locomotive, no road vehicles"),
"c052": ("street","street",["street"],T,F,F,F,"Almaty square, one SUV driving"),
"c054": ("parking_gate","gate",["gate","street","unknown"],T,F,F,T,"aerial view of a large car park, cars very small"),
"c055": ("parking_gate","gate",["gate"],T,F,F,F,"parking-garage entrance with barrier arm and ticket machine, one car parked"),
"c056": ("parking_gate","gate",["gate","unknown","bay"],F,F,F,F,"empty parking-garage deck, no vehicles"),
"c057": ("parking_gate","gate",["gate","street"],T,F,F,T,"barrier-gate operator housing; one car far away"),
"c060": ("parking_gate","gate",["gate"],T,F,F,F,"barrier gate on wet pavement in rain, a Jeep behind"),
"c061": ("parking_gate","gate",["gate","street","bay"],T,F,F,F,"multi-storey car park interior with parked cars"),
"c062": ("parking_gate","gate",["gate"],T,F,F,F,"car-park height barrier with parked cars behind"),
"c063": ("parking_gate","gate",["gate","unknown","street"],F,F,F,F,"red-white barrier on a dam road, no vehicles"),
"c069": ("garage","bay",["bay"],T,F,F,F,"tractor unit raised on column lifts at a trade show"),
"c070": ("garage","closeup",["closeup","bay"],T,F,T,T,"engine bay close-up (only part of a vehicle)"),
"c072": ("garage","bay",["bay"],T,T,F,F,"Saab in a tyre shop, mechanic kneeling at a removed wheel; US plate legible"),
"c073": ("tricky","bay",["bay","unknown"],F,F,F,T,"railway locomotive on lifting jacks in a rail depot (not a road vehicle)"),
"c074": ("garage","bay",["bay","closeup"],T,T,F,F,"person working under a VW Beetle in a home garage (B/W); vanity plate legible"),
"c076": ("garage","bay",["bay"],T,F,F,F,"Unimog on a scissor lift with people around (exhibition)"),
"c080": ("garage","bay",["bay"],T,T,F,F,"service shop: cars on two-post lifts, bonnet open; plate on a parked car"),
"c083": ("garage","closeup",["closeup","bay"],T,F,T,T,"wheel and oil canister in a workshop (only part of a car)"),
"c084": ("garage","bay",["bay"],T,T,F,F,"Ford Mustang raised on a two-post lift; plate legible"),
"c087": ("garage","unknown",["unknown","bay","street","closeup"],F,F,F,F,"roadside tyre repair: people and loose tyres, no vehicle"),
"c088": ("wash","wash",["wash"],T,F,F,F,"Volvo 850 being washed in a self-service bay; plate blurred by the uploader"),
"c089": ("wash","wash",["wash","street","gate"],T,F,F,T,"car-wash building, cars parked far away"),
"c090": ("wash","wash",["wash","gate","street"],T,F,F,F,"self-service car-wash yard with cars in bays"),
"c093": ("wash","wash",["wash","unknown","bay"],F,F,F,F,"empty self-service wash bay, no vehicle"),
"c098": ("wash","wash",["wash"],T,F,F,F,"self-service car wash, cars in two bays"),
"c099": ("wash","wash",["wash","bay"],T,F,F,F,"person washing a car inside a car-wash bay"),
"c101": ("closeup","closeup",["closeup"],T,F,T,T,"chrome spoke-wheel close-up (only part of a car)"),
"c102": ("closeup","closeup",["closeup"],T,F,T,F,"flat tyre on an SUV, close"),
"c103": ("closeup","street",["street","closeup"],T,T,T,F,"VAZ-2102, Tashkent; Uzbek plate legible"),
"c104": ("closeup","closeup",["closeup"],T,F,T,T,"open car door with a scratched, dirty window (only part of a car)"),
"c108": ("closeup","closeup",["closeup"],T,F,T,T,"flat tyre close-up, little of the car visible"),
"c109": ("closeup","closeup",["closeup","street"],T,T,T,F,"damaged Mercedes front: New York plate and an Uzbek-style plate, both legible"),
"c112": ("closeup","closeup",["closeup","street"],T,F,T,F,"black car with a flat tyre, rear quarter"),
"c113": ("closeup","street",["street","closeup"],T,T,T,F,"VAZ-2101, Tashkent; Uzbek plate legible"),
"c116": ("closeup","street",["street","closeup"],T,T,T,F,"Moskvich-412 with an old-format Uzbek plate (2001, low resolution)"),
"c018": ("night_rain","street",["street","gate","unknown"],T,F,F,T,"night square in Uzbekistan, one distant car"),
"c120": ("night_rain","street",["street"],T,F,F,F,"night street in Amsterdam: cyclists, cars, tram"),
"c121": ("night_rain","street",["street"],T,F,F,F,"cars with headlights in heavy rain at night"),
"c122": ("night_rain","gate",["gate","unknown","street"],F,F,F,F,"empty parking deck at night, no vehicles"),
"c128": ("night_rain","gate",["gate","street"],T,F,F,F,"car park at night seen from above, many cars"),
"c129": ("night_rain","street",["street"],T,F,F,F,"night traffic on a wide avenue"),
"c131": ("night_rain","unknown",["unknown","gate","street"],F,F,F,F,"foggy car park at night, lamps only, no vehicle visible"),
"c118": ("non_car","unknown",["unknown"],F,F,F,F,"observatory under the Milky Way at night"),
"c134": ("non_car","unknown",["unknown"],F,F,F,F,"man cooking wedding plov in a kazan (Uzbekistan), dark"),
"c135": ("non_car","unknown",["unknown"],F,F,F,F,"portrait of a woman (face)"),
"c137": ("non_car","unknown",["unknown"],F,F,F,F,"cat"),
"c138": ("non_car","unknown",["unknown"],F,F,F,F,"Registan square, Samarkand; small people"),
"c140": ("non_car","unknown",["unknown","street"],F,F,F,F,"bicycle parked by a graffiti door (bicycle is not a vehicle class)"),
"c142": ("non_car","unknown",["unknown"],F,F,F,F,"souvenir stall at Chorsu bazaar"),
"c144": ("non_car","unknown",["unknown"],F,F,F,F,"plov being served (Besh Qozon), people"),
"c149": ("non_car","unknown",["unknown"],F,F,F,F,"passport office queue (indoor, people)"),
"c136": ("document","unknown",["unknown"],F,F,F,F,"identity card specimen, front (document)"),
}
# videos: (category, scene, acceptable, vehicle, plate_visible, inspection, hard, notes, frame_t_s)
V = {
"c154": ("video","street",["street"],T,F,F,F,"old-town street in Ljubljana with parked vans and passing traffic (80 s; first 60 s used)",30.0),
"c155": ("video","wash",["wash","unknown"],F,F,F,T,"timelapse filmed from inside a car going through an automatic wash; no vehicle visible from outside",8.0),
"c157": ("video","gate",["gate","street"],T,F,F,F,"supermarket car park behind a wall, low resolution (633 px)",25.0),
"c158": ("video","unknown",["unknown"],F,F,F,F,"cat playing with a lizard",7.0),
"c163": ("video","gate",["gate","street"],T,F,F,F,"busy car park timelapse from a fixed camera, people walking",15.0),
"c165": ("video","street",["street"],T,F,F,T,"night traffic jam in heavy rain, Mymensingh: auto-rickshaws, dark",11.0),
"c167": ("video","motorcycle",["motorcycle","street"],T,F,F,F,"teenager learning to ride a motorcycle",8.0),
}
out = []
n = 0
def base(r, cid, kind, fid, fname):
    return {"id": fid, "kind": kind, "cid": cid, "title": r["title"], "page": r["page"],
            "download_url": r["thumb_url"] if kind == "image" else r["url"], "file": fname,
            "width": r["width"], "height": r["height"], "licence": r["licence"], "licence_url": r["licence_url"],
            "author": r["author"], "credit": r.get("credit", ""), "commons_group": r["group"]}
for cid, (cat, sc, acc, veh, pl, insp, hard, notes) in L.items():
    n += 1
    r = rows[cid]
    fid = f"jury_{n:03d}"
    ext = ".png" if r["mime"] == "image/png" else ".jpg"
    it = base(r, cid, "image", fid, fid + ext)
    it.update({"category": cat, "expected": {"scene": sc, "acceptable_scenes": acc, "vehicle": veh, "plate_visible": pl,
               "inspection_target": insp, "hard": hard, "notes": notes}})
    out.append(it)
for cid, (cat, sc, acc, veh, pl, insp, hard, notes, t) in V.items():
    n += 1
    r = rows[cid]
    vid = f"jury_{n:03d}"
    it = base(r, cid, "video", vid, vid + ".webm")
    it.update({"duration_s": float(r["duration_s"]), "category": cat,
               "expected": {"scene": sc, "acceptable_scenes": acc, "vehicle": veh, "plate_visible": pl,
                            "inspection_target": insp, "hard": hard, "notes": notes}})
    out.append(it)
    n += 1
    fr = {**{k: v for k, v in it.items() if k not in ("duration_s",)}, "id": f"jury_{n:03d}", "kind": "video_frame",
          "file": f"jury_{n:03d}.jpg", "frame_of": vid, "frame_t_s": t, "download_url": None, "category": "video_frame"}
    fr["expected"] = {**it["expected"], "notes": f"frame at {t:g}s of {vid}: " + notes}
    out.append(fr)
OUT.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
print(len(out), "items", sum(o["kind"]=="image" for o in out), "images")
