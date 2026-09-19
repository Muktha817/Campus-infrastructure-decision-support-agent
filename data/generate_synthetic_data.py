"""
Generates 250+ realistic synthetic historical maintenance records for campus/facility equipment.
Covers:
- HVAC & Chillers
- Backup Diesel Generators
- Elevators & Escalators
- Water Supply & Pumps
- Electrical Switchgear & Transformers
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

EQUIPMENT_TAXONOMY = {
    "HVAC": {
        "equipment_ids": [f"HVAC-CHILLER-{i:02d}" for i in range(1, 8)] +
                         [f"HVAC-AHU-{i:02d}" for i in range(1, 12)] +
                         [f"HVAC-VRF-{i:02d}" for i in range(1, 10)],
        "locations": ["Science Block - Roof", "Main Auditorium - Plant Room", "Engineering Annex - Floor 3",
                      "Student Center - East Wing", "Central Library - Basement Mech Room", "Administration - Attic"],
        "issues": [
            {
                "reported_issue": "Chiller unit cycling rapidly and blowing warm air",
                "symptoms": "High compressor discharge temperature, low suction pressure, ice buildup on expansion valve",
                "root_cause": "Refrigerant R-410A leak at flare nut fitting causing subcooling loss and short cycling",
                "fix_action": "Recover remaining refrigerant, braze leaky flare joint, nitrogen pressure test at 350 PSI, vacuum down to 400 microns, recharge 8.5 kg R-410A.",
                "time_hrs": 5.5,
                "cost_usd": 780,
                "urgency": "High",
                "notes": "Common during spring startup when thermal expansion loosens fittings."
            },
            {
                "reported_issue": "Air Handling Unit producing loud screeching noise and vibrating ductwork",
                "symptoms": "High decibel belt squeal, rubber dust inside blower compartment, airflow reduced by 40%",
                "root_cause": "Worn serpentine drive belt and seized idler pulley bearing",
                "fix_action": "Locked out power, replaced dual B-section V-belts and idler pulley assembly, laser-aligned sheaves, adjusted belt tension to 120 lbs.",
                "time_hrs": 3.0,
                "cost_usd": 320,
                "urgency": "Medium",
                "notes": "Belts were 6 months past scheduled maintenance replacement window."
            },
            {
                "reported_issue": "Server room cooling failure - ambient temperature reached 29°C",
                "symptoms": "CRAC unit displays high head pressure alarm (E04), condenser fan motor not spinning",
                "root_cause": "Faulty condenser fan run capacitor blown due to lightning power spike",
                "fix_action": "Discharged and replaced 45uF 440VAC dual run capacitor, verified fan motor winding resistance (5.2 ohms), cleared alarm codes.",
                "time_hrs": 2.0,
                "cost_usd": 150,
                "urgency": "Critical",
                "notes": "Server racks nearly throttled. Recommended surge suppressor on CRAC feed."
            },
            {
                "reported_issue": "Water leaking through ceiling tiles beneath 2nd floor air handler",
                "symptoms": "Condensate pan overflowing, slime accumulation in drain pan, float switch failed to trip",
                "root_cause": "Algae and microbial slime blockage in primary 3/4-inch PVC condensate trap and seized float switch",
                "fix_action": "Cleared condensate line with CO2 gun, washed drain pan with antimicrobial tablets, replaced float switch with solid-state optical sensor.",
                "time_hrs": 2.5,
                "cost_usd": 180,
                "urgency": "High",
                "notes": "Ceiling drywall required 2 replacement tiles. Biocide tablets added to PM checklist."
            },
            {
                "reported_issue": "Lecture hall freezing cold despite thermostat set to 24°C",
                "symptoms": "VAV box damper actuator stuck 100% open, zone temperature 17°C, room occupants complaining",
                "root_cause": "Stripped nylon gears in 24V floating actuator motor on VAV terminal 14",
                "fix_action": "Replaced Siemens GDE131.1P actuator with metal-gear Belimo LMB24-SR, recalibrated stroke limits via BMS.",
                "time_hrs": 1.5,
                "cost_usd": 260,
                "urgency": "Low",
                "notes": "Plastic gear wear after 7 years continuous modulates."
            },
            {
                "reported_issue": "Chemical smell and burning odor from supply air grilles",
                "symptoms": "Faint white smoke trace, thermal overload trip on blower motor starter, burning insulation smell",
                "root_cause": "Blower motor phase imbalance leading to stator winding insulation breakdown",
                "fix_action": "Replaced 5 HP 3-phase blower motor, checked MCC bucket contactor, retorqued terminal lugs.",
                "time_hrs": 6.0,
                "cost_usd": 1250,
                "urgency": "Critical",
                "notes": "Building evacuated for 40 minutes due to smell."
            },
            {
                "reported_issue": "Chilled water flow error on central BMS console",
                "symptoms": "Flow meter reading 0 GPM despite secondary pump running, differential pressure sensor pegged high",
                "root_cause": "Clogged Y-strainer mesh screen on chiller evaporator inlet with rust scale",
                "fix_action": "Isolated isolation valves, removed strainer plug, cleaned heavy rust flakes from 20-mesh basket, reinstalled with new gasket.",
                "time_hrs": 3.0,
                "cost_usd": 210,
                "urgency": "High",
                "notes": "Water treatment chemical levels in closed loop need testing."
            }
        ]
    },
    "Diesel Generator": {
        "equipment_ids": [f"GEN-CUMMINS-{i:02d}" for i in range(1, 6)] +
                         [f"GEN-CAT-{i:02d}" for i in range(1, 5)] +
                         [f"GEN-KOHLER-{i:02d}" for i in range(1, 4)],
        "locations": ["Powerhouse Substation Yard", "Hospital Wing - Emergency Generator Bunker",
                      "IT Data Center Compound", "North Campus Utility Compound"],
        "issues": [
            {
                "reported_issue": "Generator cranked slowly and failed weekly automatic load test",
                "symptoms": "Battery terminal voltage drops to 9.2V during crank cycle, overcrank fault indicator lit",
                "root_cause": "Sulfated 24V lead-acid starter battery bank and malfunctioning trickle battery charger",
                "fix_action": "Replaced dual Group 8D 12V batteries, replaced SENS float charger with smart temperature-compensating charger, cleaned terminal posts.",
                "time_hrs": 2.5,
                "cost_usd": 950,
                "urgency": "Critical",
                "notes": "Emergency backup was compromised. Weekly inspection check for float voltage instituted."
            },
            {
                "reported_issue": "Generator engine shuts down after 15 minutes of running on high coolant temp",
                "symptoms": "Coolant temp gauge reaches 104°C, radiator upper hose cold, lower hose boiling hot",
                "root_cause": "Stuck closed mechanical thermostat and air bubble trapped in engine cylinder head",
                "fix_action": "Drained 50/50 ethylene glycol coolant, replaced dual 82°C thermostats and gaskets, refilled with vacuum bleeder tool.",
                "time_hrs": 4.0,
                "cost_usd": 480,
                "urgency": "High",
                "notes": "Thermostats showed calcium scale buildup. Coolant flush performed."
            },
            {
                "reported_issue": "Engine sputters, hunts for RPM, and emits heavy black smoke under 50% load",
                "symptoms": "Frequency fluctuation between 48Hz and 53Hz, primary fuel filter vacuum gauge in red zone",
                "root_cause": "Microbial growth and water sludge blocking primary Racor fuel/water separator",
                "fix_action": "Replaced 30-micron primary and 2-micron secondary fuel filters, drained 2 gallons water from day tank bottom, dosed fuel tank with biocidal additive.",
                "time_hrs": 3.5,
                "cost_usd": 420,
                "urgency": "High",
                "notes": "Fuel sitting stagnant for 11 months without fuel polishing."
            },
            {
                "reported_issue": "Automatic Transfer Switch (ATS) failed to switch back to utility power after grid restored",
                "symptoms": "Building remained on generator power, ATS control panel showed 'Source 1 Voltage Unacceptable' despite healthy 415V utility feed",
                "root_cause": "Burnt sensing fuse on utility voltage monitor phase B inside ATS enclosure",
                "fix_action": "De-energized ATS bypass, replaced 600V 2A Class CC fast-acting sensing fuse, checked utility phase balance, tested auto-retransfer sequence.",
                "time_hrs": 2.0,
                "cost_usd": 120,
                "urgency": "Critical",
                "notes": "Generator ran extra 4 hours consuming 180 liters of diesel unnecessarily."
            },
            {
                "reported_issue": "Generator block heater breaker trips repeatedly in main electrical panel",
                "symptoms": "Block temperature drops to 12°C in winter, cold start cranking time extended to 9 seconds",
                "root_cause": "Grounded heating element inside Kim Hotstart 2500W engine pre-heater",
                "fix_action": "Isolated heater circuit, drained partial coolant, replaced Kim Hotstart element, tested insulation resistance (infinite megohms).",
                "time_hrs": 2.5,
                "cost_usd": 380,
                "urgency": "Medium",
                "notes": "Cold start capability compromised in sub-zero winter temperatures."
            },
            {
                "reported_issue": "Low oil pressure safety trip during monthly full load bank test",
                "symptoms": "Oil pressure dropped below 22 PSI at 1500 RPM, engine halted immediately",
                "root_cause": "Faulty oil pressure sender transducer resistance drift and partially clogged oil bypass valve",
                "fix_action": "Verified real pressure with mechanical master gauge (found 46 PSI), replaced VDO 0-10 bar electronic oil sender and engine oil filter.",
                "time_hrs": 3.0,
                "cost_usd": 290,
                "urgency": "High",
                "notes": "False safety trip prevented emergency readiness."
            }
        ]
    },
    "Elevator": {
        "equipment_ids": [f"ELEV-OTIS-{i:02d}" for i in range(1, 8)] +
                         [f"ELEV-KONE-{i:02d}" for i in range(1, 6)] +
                         [f"ELEV-SCHINDLER-{i:02d}" for i in range(1, 5)],
        "locations": ["Science Tower - Bank A", "Student Union - South Passenger", "Library - Main Central Lift",
                      "Medical Sciences Building - Freight Elevator", "Residence Hall C - Passenger 1"],
        "issues": [
            {
                "reported_issue": "Elevator stuck between 3rd and 4th floors with 4 passengers trapped",
                "symptoms": "Safety circuit tripped, car stopped abruptly, door lock circuit open fault code (F021)",
                "root_cause": "Debris lodged in 3rd floor landing door interlock roller clutch mechanism, breaking safety circuit",
                "fix_action": "Manually lowered car to 3rd floor sill, released passengers safely, cleared debris from sill track, re-aligned door interlock pick-up rollers, adjusted gap to 6mm.",
                "time_hrs": 2.5,
                "cost_usd": 450,
                "urgency": "Critical",
                "notes": "Emergency rescue conducted within 18 minutes. No injuries."
            },
            {
                "reported_issue": "Car landing with a violent shudder and misleveling by 2 inches below floor level",
                "symptoms": "Trip hazard at 1st floor entrance, brake coil engaging prematurely before full stop",
                "root_cause": "Optical leveling vane sensor dusted over with carbon soot from guide shoe wear",
                "fix_action": "Cleaned hoistway optical leveling sensors with isopropyl alcohol, replaced worn gibs on car guide shoes, recalibrated floor table in controller.",
                "time_hrs": 3.0,
                "cost_usd": 310,
                "urgency": "High",
                "notes": "Trip and fall hazard reported by students. Required immediate barricading."
            },
            {
                "reported_issue": "Elevator doors repeatedly reopen and close without car departing",
                "symptoms": "Door nudging mode activates after 20 seconds, door operator motor warm to touch",
                "root_cause": "Misaligned infrared door light curtain sensors and damaged flexible traveling cable leads",
                "fix_action": "Replaced 2D infrared detector curtain (Formula Systems FCU), re-routed flexible sensor harness away from pinch point, adjusted closing force to 135N.",
                "time_hrs": 3.0,
                "cost_usd": 680,
                "urgency": "Medium",
                "notes": "Light curtain cables had internal fatigue fracture."
            },
            {
                "reported_issue": "Elevator cabin emergency intercom telephone silent when button pressed",
                "symptoms": "No dial tone, alarm bell rings locally in hoistway but central security desk receives no call",
                "root_cause": "Broken conductor wire in traveling cable pair 18-19 due to mechanical fatigue at hanger bracket",
                "fix_action": "Tested traveling cable cores with multimeter, switched emergency autodialer connection to spare shielded twisted pair #24, verified 2-way audio to dispatch.",
                "time_hrs": 2.0,
                "cost_usd": 180,
                "urgency": "Critical",
                "notes": "Mandatory safety code violation (ASME A17.1). Elevator shut down until repaired."
            },
            {
                "reported_issue": "Hydraulic freight elevator descending slowly under static load over weekend",
                "symptoms": "Car creeps down 14 inches over 24 hours, oil reservoir level slightly low, oil odor in pit",
                "root_cause": "Worn polyurethane cylinder piston packings leaking hydraulic fluid into scavenge line",
                "fix_action": "Pumped down hydraulic ram, dismantled cylinder gland head, installed new chevron packing seals and wiper ring, refilled ISO 46 hydraulic fluid.",
                "time_hrs": 7.0,
                "cost_usd": 1650,
                "urgency": "High",
                "notes": "Freight elevator out of service for 1 full working day."
            }
        ]
    },
    "Water Supply & Pumps": {
        "equipment_ids": [f"PUMP-BOOSTER-{i:02d}" for i in range(1, 8)] +
                         [f"PUMP-SUMP-{i:02d}" for i in range(1, 6)] +
                         [f"PUMP-CIRC-{i:02d}" for i in range(1, 6)],
        "locations": ["Central Water Works - Pump Room", "Hostel Block A - Basement Sump", "Sports Complex - Pool Filter Plant",
                      "Campus Dining Hall - Grease Interceptor Pit", "Science Complex - Reverse Osmosis Room"],
        "issues": [
            {
                "reported_issue": "Upper floors (4-7) experiencing zero water pressure during morning peak hours",
                "symptoms": "Booster pump #2 trips on drive fault VFD-OC (Overcurrent), pump casing burning hot",
                "root_cause": "Cavitation erosion and foreign object (cloth rag) jammed in impeller eye of centrifugal booster",
                "fix_action": "Locked out pump, decoupled pump head, cleared foreign debris from bronze impeller, replaced mechanical carbon-ceramic seal, purged suction header.",
                "time_hrs": 4.5,
                "cost_usd": 540,
                "urgency": "High",
                "notes": "Overnight water tank inlet screen had torn, allowing debris into header."
            },
            {
                "reported_issue": "Basement boiler room flooded with 3 inches of standing water",
                "symptoms": "Submersible sump pump not activating, water level above high-water alarm probe",
                "root_cause": "Tethered mercury float switch tether tangled around discharge pipe, preventing buoyant tilt",
                "fix_action": "Manually activated pump, untangled and secured float switch bracket with rigid pipe guide, verified automatic cycle and audible high water beacon.",
                "time_hrs": 1.5,
                "cost_usd": 90,
                "urgency": "Critical",
                "notes": "Emergency portable trash pump used to dewater area first. Electrical switchgear at risk."
            },
            {
                "reported_issue": "Excessive water hammer banging noise in dormitory plumbing pipes",
                "symptoms": "Loud clanking noise whenever flush valves close, pressure spikes to 110 PSI",
                "root_cause": "Ruptured internal butyl rubber bladder in hydropneumatic expansion/surge tank",
                "fix_action": "Depressurized water system, tested tank Schrader valve (liquid sprayed out), replaced 100-liter Amtrol expansion tank, pre-charged with nitrogen to 55 PSI.",
                "time_hrs": 3.5,
                "cost_usd": 720,
                "urgency": "Medium",
                "notes": "Continued water hammer would have cracked soldered copper elbows."
            },
            {
                "reported_issue": "Continuous puddle forming beneath domestic hot water circulating pump",
                "symptoms": "Steady water drip of 15 drops/min from pump shaft gland, white mineral scale buildup",
                "root_cause": "Degraded tungsten carbide mechanical shaft seal due to thermal cycling and hard water scale",
                "fix_action": "Isolated hot water loop, pulled pump cartridge, replaced mechanical shaft seal kit and O-rings, re-torqued flange bolts to 35 ft-lbs.",
                "time_hrs": 2.5,
                "cost_usd": 240,
                "urgency": "Low",
                "notes": "Scheduled routine replacement."
            }
        ]
    },
    "Electrical Switchgear": {
        "equipment_ids": [f"SWG-MAIN-415V-{i:02d}" for i in range(1, 6)] +
                         [f"XFRM-OIL-11KV-{i:02d}" for i in range(1, 4)] +
                         [f"MCC-PANEL-{i:02d}" for i in range(1, 6)],
        "locations": ["Main 11kV Substation - Room 1", "Academic Complex - Main Distribution Board",
                      "Athletic Center - Electrical Vault", "Research Labs - Clean Power Distribution Unit"],
        "issues": [
            {
                "reported_issue": "Main 800A air circuit breaker (ACB) tripped, cutting power to Academic Block B",
                "symptoms": "Microprocessor trip unit shows Ground Fault (GF) flag, no visible arc flash or burning odor",
                "root_cause": "Moisture ingress and rodent gnawing on subterranean feeder cable insulation causing intermittent leakage to conduit",
                "fix_action": "Megger-tested feeder cables (found phase A to ground at 0.15 Megohms), pulled new 185 sq mm XLPE cable through duct, sealed conduit ends with duct seal.",
                "time_hrs": 8.0,
                "cost_usd": 2400,
                "urgency": "Critical",
                "notes": "Temporary diesel generator deployed during 8-hour outage."
            },
            {
                "reported_issue": "Substation infrared thermography scan detected extreme thermal hot spot on Main Busbar",
                "symptoms": "Phase L2 busbar joint temperature measured at 112°C while L1 and L3 were at 44°C",
                "root_cause": "Loose Belleville spring washers and contact oxidation at busbar copper splice joint",
                "fix_action": "Scheduled weekend planned outage, unbolted copper busbars, sanded oxidation with Scotch-Brite, applied conductive grease, torqued grade 8.8 bolts to 65 Nm with calibrated wrench.",
                "time_hrs": 4.0,
                "cost_usd": 350,
                "urgency": "High",
                "notes": "Prevented catastrophic busbar arc fault flashover."
            },
            {
                "reported_issue": "Transformer oil temperature alarm triggered on 11kV/415V 1000kVA unit",
                "symptoms": "Winding temperature gauge at 92°C, oil sight glass shows dark amber color, silica gel in breather turned pink",
                "root_cause": "Saturated silica gel breather allowing ambient moisture into oil, and heavy dust clogging radiator cooling fins",
                "fix_action": "Power-washed radiator cooling fins with dry compressed air, replaced 5 kg blue silica gel desiccant charge, took oil sample for DGA (Dissolved Gas Analysis).",
                "time_hrs": 3.0,
                "cost_usd": 280,
                "urgency": "High",
                "notes": "DGA lab report returned dielectric breakdown at acceptable 48 kV."
            },
            {
                "reported_issue": "Capacitor bank automatic power factor controller clicking continuously without correcting PF",
                "symptoms": "Power factor penalty charges incurred (PF dropped to 0.76), swollen capacitor cans in Step 3 and 4",
                "root_cause": "Dielectric puncture and expansion of internal metallized polypropylene film in 25 kVAR capacitor steps",
                "fix_action": "Discharged bank safely, replaced two swollen 25 kVAR 440V heavy-duty capacitors, replaced fused discharge resistors, verified PF restored to 0.98 lag.",
                "time_hrs": 3.0,
                "cost_usd": 620,
                "urgency": "Medium",
                "notes": "Avoided monthly utility low power factor penalty of $450."
            }
        ]
    }
}

def generate_records(target_count=265):
    records = []
    base_date = datetime.now() - timedelta(days=730)
    record_id = 1

    equipment_keys = list(EQUIPMENT_TAXONOMY.keys())

    while len(records) < target_count:
        eq_type = random.choice(equipment_keys)
        eq_data = EQUIPMENT_TAXONOMY[eq_type]
        issue_template = random.choice(eq_data["issues"])
        eq_id = random.choice(eq_data["equipment_ids"])
        location = random.choice(eq_data["locations"])

        time_variance = round(max(0.5, issue_template["time_hrs"] + random.uniform(-0.8, 1.2)), 1)
        cost_variance = int(max(50, issue_template["cost_usd"] * random.uniform(0.85, 1.25)))
        random_days = random.randint(0, 720)
        log_date = (base_date + timedelta(days=random_days)).strftime("%Y-%m-%d %H:%M")

        tech_initials = random.choice([
            "Tech #14 (J. Miller)", "Tech #08 (K. Patel)", "Tech #22 (D. Zhang)",
            "Tech #03 (M. Al-Mansoor)", "Tech #19 (S. Jenkins)"
        ])

        record = {
            "id": f"MNT-{record_id:04d}",
            "equipment_type": eq_type,
            "equipment_id": eq_id,
            "location": location,
            "reported_issue": issue_template["reported_issue"],
            "symptoms": issue_template["symptoms"],
            "root_cause": issue_template["root_cause"],
            "fix_action": issue_template["fix_action"],
            "resolution_time_hrs": time_variance,
            "cost_estimate_usd": cost_variance,
            "urgency": issue_template["urgency"],
            "date_logged": log_date,
            "technician": tech_initials,
            "technician_notes": issue_template["notes"],
            "verified_by_supervisor": True
        }
        records.append(record)
        record_id += 1

    return records

def main():
    output_dir = Path(__file__).resolve().parent
    records = generate_records(265)

    json_path = output_dir / "maintenance_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {len(records)} maintenance records -> {json_path}")

    counts = {}
    for r in records:
        counts[r["equipment_type"]] = counts.get(r["equipment_type"], 0) + 1
    print("Record breakdown by equipment type:", counts)

if __name__ == "__main__":
    main()
