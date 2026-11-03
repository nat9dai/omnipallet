"""Single source of truth for aisle geometry, pallet layout and labels (x along the aisle, y left, z up, m)."""
import dataclasses
import random

LENGTH = 80.0  # m
WIDTH = 2.5  # m, clear width between pallet faces at y = +-WIDTH/2
HEIGHT = 8.0  # m, rack height
LEVELS = 5
LEVEL_H = HEIGHT / LEVELS

BAY_PITCH = 2.7  # m, upright centre to upright centre (3 Euro pallets per bay)
N_BAYS = 29
RACK_X0 = (LENGTH - N_BAYS * BAY_PITCH) / 2  # first upright centre, m
PALLETS_PER_BAY = 3
RACK_DEPTH = 1.1  # m, upright frame depth
UPRIGHT_W, UPRIGHT_D = 0.09, 0.07  # m, upright section (along x, along y)
BEAM_H, BEAM_D = 0.12, 0.05  # m, box beam section
FACE_GAP = 0.05  # m, pallet overhang past the upright front face

PALLET_W, PALLET_D, PALLET_H = 0.8, 1.2, 0.144  # m, Euro pallet, narrow side faces the aisle
LOAD_H = (0.9, 1.3)  # m, load height range; fits under the 1.6 m level pitch

LABEL_W, LABEL_H = 0.105, 0.074  # m, label on the load face
LABEL_Z_ABOVE_PALLET = 0.50  # m, label centre above the pallet base (GS1: 400 to 800 mm)
X_DIM_MM = 0.5  # GS1-128 module width (GS1 allows 0.495 to 1.016 mm)
OCCUPANCY = 0.85  # share of pallet slots that hold a pallet
SEED = 7


def level_base_z(level):
    """Height of the pallet base on a level: floor for level 0, top of the beam above it otherwise."""
    return 0.0 if level == 0 else level * LEVEL_H + BEAM_H / 2


def label_z(level):
    return level_base_z(level) + LABEL_Z_ABOVE_PALLET


def slot_x(bay, slot):
    """x of a pallet centre: three pallets spread evenly between the uprights of a bay."""
    clear = BAY_PITCH - UPRIGHT_W
    gap = (clear - PALLETS_PER_BAY * PALLET_W) / (PALLETS_PER_BAY + 1)
    x_left = RACK_X0 + bay * BAY_PITCH + UPRIGHT_W / 2
    return x_left + gap * (slot + 1) + PALLET_W * (slot + 0.5)


@dataclasses.dataclass
class Pallet:
    side: str  # "L" (y > 0) or "R" (y < 0) when flying in +x
    bay: int
    level: int
    slot: int
    sku: str
    sscc: str  # 18 digits with the GS1 check digit
    qty: int
    load: str  # load type for the visual: "boxes", "drums", "bags", "crates"
    load_h: float

    @property
    def location(self):
        return f"A01-{self.side}-B{self.bay + 1:02d}-L{self.level + 1}-P{self.slot + 1}"

    @property
    def face_y(self):
        return WIDTH / 2 if self.side == "L" else -WIDTH / 2

    @property
    def x(self):
        return slot_x(self.bay, self.slot)

    @property
    def label_pose(self):
        """Label centre (x, y, z) on the load face, facing the aisle centre."""
        return self.x, self.face_y, label_z(self.level)

    @property
    def gs1(self):
        return "00" + self.sscc


def gs1_check_digit(digits17):
    s = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits17)))
    return str((10 - s % 10) % 10)


SKUS = [("LUB-5W30-208L", "drums"), ("LUB-15W40-208L", "drums"), ("GRS-EP2-180KG", "drums"),
        ("FLT-OIL-24PK", "boxes"), ("FLT-AIR-12PK", "boxes"), ("BRK-PAD-40PK", "boxes"),
        ("CLN-DEG-20L", "crates"), ("ADD-FUEL-12PK", "boxes"), ("ABS-GRAN-25KG", "bags"),
        ("CPL-HOSE-10PK", "crates"), ("BLT-V-50PK", "boxes"), ("CLT-COOL-20L", "crates")]


def layout(seed=SEED, occupancy=OCCUPANCY):
    """All occupied pallet slots with random SKU, serial and load; same seed gives the same aisle."""
    rng = random.Random(seed)
    pallets = []
    serial = 1000
    for side in ("L", "R"):
        for bay in range(N_BAYS):
            for level in range(LEVELS):
                for slot in range(PALLETS_PER_BAY):
                    if rng.random() > occupancy:
                        continue
                    sku, load = rng.choice(SKUS)
                    serial += rng.randint(1, 9)
                    body = "3" + "761234" + f"{serial:010d}"  # extension, GS1 company prefix, serial
                    sscc = body + gs1_check_digit(body)
                    pallets.append(Pallet(side, bay, level, slot, sku, sscc, rng.choice([8, 12, 24, 40, 48]),
                                          load, round(rng.uniform(*LOAD_H), 3)))
    return pallets


if __name__ == "__main__":
    p = layout()
    print(f"{len(p)} pallets of {2 * N_BAYS * LEVELS * PALLETS_PER_BAY} slots; first: {p[0].location} {p[0].gs1}")
    print("label heights:", [round(label_z(k), 3) for k in range(LEVELS)])
