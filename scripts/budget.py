"""Mass and power budget of the stock-counting quad and its endurance on the mission.

    python scripts/budget.py   # writes results/budget.csv and results/budget.json
"""
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim", "vehicle"))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone", "stock_drone"))
import design as d  # noqa: E402
import trajectory  # noqa: E402
import aisle_spec as spec  # noqa: E402

# group, item, choice, qty, unit mass g, unit power W (average in flight), source
ITEMS = [
    ("propulsion", "motor", "T-Motor 2807 1300 KV", 4, 52.6, None, "datasheet"),
    ("propulsion", "propeller", "HQProp 7x4x3", 4, 10.3, None, "datasheet"),
    ("propulsion", "ESC", "Holybro Tekko32 F4 4-in-1 50 A", 1, 13.8, None, "datasheet"),
    ("structure", "frame", "Charpi stacked carbon plates, 283 mm diag.", 1, 150.0, None, "estimate"),
    ("structure", "prop guards", "4 ring guards, carbon/PA12", 4, 20.0, None, "estimate"),
    ("avionics", "flight controller", "Holybro Pixhawk 6C Mini", 1, 39.2, 1.0, "datasheet"),
    ("avionics", "companion computer", "Jetson Orin NX 16 GB + carrier + fan", 1, 150.0, 15.0, "module 28 g + estimate"),
    ("avionics", "Wi-Fi", "Intel AX210 + 2 antennas", 1, 10.0, 2.0, "estimate"),
    ("avionics", "power module and BEC", "Holybro PM02 + 12 V/5 V BEC", 1, 30.0, 1.0, "estimate"),
    ("avionics", "wiring and mounts", "connectors, harness, standoffs", 1, 40.0, None, "estimate"),
    ("sensing", "VIO camera board", "Luxonis OAK-FFC-4P", 1, 25.0, 3.0, "estimate"),
    ("sensing", "VIO fisheye camera", "OV9782 + 220 deg M12 lens", 4, 10.0, 0.16, "datasheet + lens [unverified]"),
    ("sensing", "barcode camera", "IMX477 12 MP + 12 mm M12 lens", 2, 25.0, 1.2, "estimate"),
    ("sensing", "LED light bar", "6 W white LED, strobed with exposure", 2, 20.0, 3.0, "estimate (50 % duty)"),
    ("sensing", "ToF rangefinder", "Benewake TFmini-S (left, right, down, up)", 4, 5.0, 0.6, "datasheet"),
    ("energy", "battery", "6S2P Molicel P45B 21700, 9.0 Ah, 194 Wh", 1, 880.0, None, "12 x 70 g cells + 40 g BMS/wrap"),
]
CONTINGENCY = 0.08
BATTERY_WH, USABLE = 194.4, 0.85  # Wh (21.6 V x 9.0 Ah); 85 % usable, down to 3.2 V per cell
ETA_MOTOR_ESC = 0.80


def main():
    rows, mass, p_av = [], 0.0, 0.0
    for group, item, choice, qty, m, p, src in ITEMS:
        rows.append({"group": group, "item": item, "choice": choice, "qty": qty, "mass_g": round(qty * m, 1),
                     "power_W": round(qty * p, 2) if p else "", "source": src})
        mass += qty * m
        p_av += qty * p if p else 0.0
    m_total = mass * (1 + CONTINGENCY) / 1000.0
    w_h = d.hover_omega(m_total)
    p_shaft = 4 * d.K_Q * w_h ** 3
    p_hover = p_shaft / ETA_MOTOR_ESC
    tr = trajectory.serpentine(0.5, spec.LENGTH - 0.5, [spec.label_z(k) - 0.06 for k in range(spec.LEVELS)])
    t_mission = tr.duration + 15 + 10  # s, plus take-off and landing
    p_total = 1.10 * p_hover + p_av  # 10 % for manoeuvres and HVAC gusts
    endurance = BATTERY_WH * USABLE / p_total * 3600
    energy_mission = p_total * t_mission / 3600
    out = {
        "dry_mass_g": round(mass - 880.0, 1),
        "sum_mass_g": round(mass, 1),
        "mtow_kg_with_contingency": round(m_total, 3),
        "thrust_to_weight": round(4 * d.thrust_max() / (m_total * d.G), 2),
        "hover_rotor_speed_rpm": round(w_h * 60 / (2 * math.pi)),
        "hover_shaft_power_W": round(p_shaft, 1),
        "hover_electric_power_W": round(p_hover, 1),
        "avionics_power_W": round(p_av, 1),
        "total_power_W": round(p_total, 1),
        "battery_Wh": BATTERY_WH,
        "endurance_min": round(endurance / 60, 1),
        "mission_time_min": round(t_mission / 60, 2),
        "mission_energy_Wh": round(energy_mission, 1),
        "margin": round(endurance / t_mission, 2),
        "battery_left_after_mission_pct": round(100 * (1 - energy_mission / (BATTERY_WH * USABLE)), 1),
        "path_m": round(tr.length(), 1),
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "budget.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(ROOT, "results", "budget.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(f"{r['group']:11s} {r['item']:22s} {r['qty']} x  {r['mass_g']:7.1f} g  {str(r['power_W']):>5s} W  {r['choice']}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
