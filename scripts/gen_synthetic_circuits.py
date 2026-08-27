"""Generate synthetic circuit schematics with perfect GT labels.
Uses matplotlib for rendering — no external EDA tools needed.
Produces training-ready (image, GT) pairs in JSONL format.

Design: component library → random topology → grid placement →
orthogonal wire routing → SVG rendering → PNG export + JSONL.
"""
import os, sys, json, time, random, math, argparse
import numpy as np
from collections import defaultdict

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrow, Arc
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ============================================================
# Component Symbol Library
# Each symbol: list of (element_type, params_dict) for drawing
# Plus: pin_positions dict {pin_name: (x, y)} in normalized coords
# ============================================================

class ComponentDef:
    """Definition of a component type."""
    def __init__(self, name, category, draw_func, pins, width=60, height=40):
        self.name = name
        self.category = category  # 'passive', 'semiconductor', 'ic', 'power', 'connector'
        self.draw_func = draw_func  # (ax, x, y, rotation, scale) -> None
        self.pins = pins  # {pin_name: (dx, dy)} in raw coords
        self.width = width
        self.height = height

    def get_pin_positions(self, x, y, rotation=0):
        """Get world-coordinate pin positions after rotation."""
        result = {}
        for name, (dx, dy) in self.pins.items():
            if rotation == 0:
                rx, ry = x + dx, y + dy
            elif rotation == 90:
                rx, ry = x - dy, y + dx
            elif rotation == 180:
                rx, ry = x - dx, y - dy
            elif rotation == 270:
                rx, ry = x + dy, y - dx
            else:
                rad = math.radians(rotation)
                rx = x + dx * math.cos(rad) - dy * math.sin(rad)
                ry = y + dx * math.sin(rad) + dy * math.cos(rad)
            result[name] = (rx, ry)
        return result


# ---- Drawing helpers ----

def _draw_resistor_us(ax, x, y, rot=0, scale=1.0):
    """US-style zigzag resistor."""
    w, h = 50 * scale, 20 * scale
    zigzag = [
        (x - w/2, y), (x - w/3, y - h/2), (x - w/6, y + h/2),
        (x, y - h/2), (x + w/6, y + h/2), (x + w/3, y - h/2),
        (x + w/2, y)
    ]
    ax.plot([p[0] for p in zigzag], [p[1] for p in zigzag], 'k-', linewidth=1.5)

def _draw_resistor_iec(ax, x, y, rot=0, scale=1.0):
    """IEC-style rectangular resistor."""
    w, h = 40 * scale, 20 * scale
    rect = Rectangle((x - w/2, y - h/2), w, h, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)

def _draw_capacitor(ax, x, y, rot=0, scale=1.0):
    """Two parallel plates."""
    gap = 12 * scale
    lh = 14 * scale  # half plate height
    ax.plot([x - gap, x - gap], [y - lh, y + lh], 'k-', linewidth=2)
    ax.plot([x + gap, x + gap], [y - lh, y + lh], 'k-', linewidth=2)

def _draw_polarized_cap(ax, x, y, rot=0, scale=1.0):
    """Polarized capacitor with + marker."""
    gap = 12 * scale
    lh = 14 * scale
    ax.plot([x - gap, x - gap], [y - lh, y + lh], 'k-', linewidth=1.5)
    ax.plot([x + gap, x + gap], [y - lh, y + lh], 'k-', linewidth=2)
    # curved plate on the right indicates cathode
    arc = Arc((x + gap, y), 8*scale, 10*scale, angle=0, theta1=-90, theta2=90, color='black', linewidth=1)
    ax.add_patch(arc)
    ax.text(x + gap + 12*scale, y + 4*scale, '+', fontsize=7, ha='center', va='center')

def _draw_inductor(ax, x, y, rot=0, scale=1.0):
    """Series of arcs for inductor."""
    w = 50 * scale
    n_loops = 4
    loop_w = w / n_loops
    amp = 10 * scale
    xs = np.linspace(x - w/2, x + w/2, n_loops * 8 + 1)
    # Actually simpler: draw arcs
    for i in range(n_loops):
        cx = x - w/2 + loop_w * (i + 0.5)
        arc = Arc((cx, y), loop_w, amp * 2, angle=0, theta1=180, theta2=360, color='black', linewidth=1.5)
        ax.add_patch(arc)

def _draw_diode(ax, x, y, rot=0, scale=1.0):
    """Diode: triangle + bar."""
    w, h = 30 * scale, 16 * scale
    tri = Polygon([(x - w/2, y), (x + w/2, y - h), (x + w/2, y + h)], fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(tri)
    ax.plot([x + w/2, x + w/2], [y - h, y + h], 'k-', linewidth=2)

def _draw_led(ax, x, y, rot=0, scale=1.0):
    """LED: diode + arrows."""
    _draw_diode(ax, x, y, rot, scale)
    w, h = 30 * scale, 16 * scale
    # arrows
    ax.annotate('', xy=(x + w/2 + 10*scale, y - h - 4*scale), xytext=(x + w/2 - 2*scale, y - h - 4*scale),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.annotate('', xy=(x + w/2 + 10*scale, y + h + 4*scale), xytext=(x + w/2 - 2*scale, y + h + 4*scale),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

def _draw_zener(ax, x, y, rot=0, scale=1.0):
    """Zener diode: diode with bent bar."""
    w, h = 30 * scale, 16 * scale
    tri = Polygon([(x - w/2, y), (x + w/2, y - h), (x + w/2, y + h)], fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(tri)
    # Zener has a bent bar
    ax.plot([x + w/2, x + w/2 + 6*scale], [y - h, y - h - 3*scale], 'k-', linewidth=2)

def _draw_bjt_npn(ax, x, y, rot=0, scale=1.0):
    """NPN BJT: circle with C/B/E."""
    r = 22 * scale
    circle = Circle((x, y), r, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    # Collector top, Base left, Emitter bottom
    ax.plot([x, x], [y + r, y + r + 10*scale], 'k-', linewidth=1.5)  # C
    ax.plot([x - r, x - r - 10*scale], [y, y], 'k-', linewidth=1.5)  # B
    ax.plot([x, x], [y - r, y - r - 10*scale], 'k-', linewidth=1.5)  # E
    # Arrow on emitter
    ax.annotate('', xy=(x + 6*scale, y - r - 4*scale), xytext=(x - 6*scale, y - r - 4*scale),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

def _draw_mosfet_n(ax, x, y, rot=0, scale=1.0):
    """N-channel MOSFET."""
    r = 22 * scale
    circle = Circle((x, y), r, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    # D top, G left (with gap), S bottom
    ax.plot([x, x], [y + r, y + r + 10*scale], 'k-', linewidth=1.5)  # D
    ax.plot([x - r - 6*scale, x - r], [y, y], 'k-', linewidth=1.5)  # G (with gap)
    ax.plot([x, x], [y - r, y - r - 10*scale], 'k-', linewidth=1.5)  # S
    # Arrow on source
    ax.annotate('', xy=(x + 6*scale, y - r - 4*scale), xytext=(x - 6*scale, y - r - 4*scale),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

def _draw_opamp(ax, x, y, rot=0, scale=1.0):
    """Op-amp: triangle."""
    w, h = 40 * scale, 30 * scale
    tri = Polygon([(x - w/2, y - h), (x - w/2, y + h), (x + w/2, y)], fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(tri)
    # + and - inputs
    ax.text(x - w/2 + 8*scale, y - h/3, '+', fontsize=6, ha='center', va='center')
    ax.text(x - w/2 + 8*scale, y + h/3, '-', fontsize=6, ha='center', va='center')

def _draw_ic(ax, x, y, rot=0, scale=1.0):
    """IC: rectangle with pin stubs."""
    w, h = 60 * scale, 40 * scale
    rect = Rectangle((x - w/2, y - h/2), w, h, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    # Pin stubs on left and right
    for i in range(4):
        py = y - h/2 + h * (i + 0.5) / 4
        ax.plot([x - w/2 - 8*scale, x - w/2], [py, py], 'k-', linewidth=1)
        ax.plot([x + w/2, x + w/2 + 8*scale], [py, py], 'k-', linewidth=1)

def _draw_connector(ax, x, y, rot=0, scale=1.0):
    """Connector/header: rectangle with pins."""
    w, h = 50 * scale, 30 * scale
    # Open side on one end
    rect = Rectangle((x - w/2, y - h/2), w, h, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    # Pin stubs on bottom
    for i in range(3):
        px = x - w/2 + w * (i + 0.25) / 2
        ax.plot([px, px], [y + h/2, y + h/2 + 8*scale], 'k-', linewidth=1)

def _draw_power_flag(ax, x, y, rot=0, scale=1.0):
    """Power flag: arrow pointing up."""
    arrow = FancyArrow(x, y - 10*scale, 0, 20*scale, width=8*scale, head_width=16*scale,
                       head_length=8*scale, fill='black', edgecolor='black', linewidth=1)
    ax.add_patch(arrow)

def _draw_gnd(ax, x, y, rot=0, scale=1.0):
    """Ground symbol: three horizontal lines."""
    lw = [30*scale, 20*scale, 10*scale]
    for i, w in enumerate(lw):
        ly = y - i * 6 * scale
        ax.plot([x - w/2, x + w/2], [ly, ly], 'k-', linewidth=1.5)

def _draw_vcc(ax, x, y, rot=0, scale=1.0):
    """VCC symbol: horizontal line with dot."""
    ax.plot([x - 10*scale, x + 10*scale], [y, y], 'k-', linewidth=1.5)
    ax.plot([x], [y], 'ko', markersize=3)

def _draw_crystal(ax, x, y, rot=0, scale=1.0):
    """Crystal: rectangle with ground plates."""
    w, h = 30 * scale, 16 * scale
    rect = Rectangle((x - w/2, y - h/2), w, h, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    # Ground plate on each side
    ax.plot([x - w/2 - 8*scale, x - w/2 - 8*scale], [y - h/2 - 4*scale, y + h/2 + 4*scale], 'k-', linewidth=0.8)
    ax.plot([x + w/2 + 8*scale, x + w/2 + 8*scale], [y - h/2 - 4*scale, y + h/2 + 4*scale], 'k-', linewidth=0.8)

def _draw_fuse(ax, x, y, rot=0, scale=1.0):
    """Fuse: rectangle with wire through it."""
    w, h = 30 * scale, 14 * scale
    rect = Rectangle((x - w/2, y - h/2), w, h, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.plot([x - w/2 - 6*scale, x + w/2 + 6*scale], [y, y], 'k-', linewidth=1)

def _draw_var_resistor(ax, x, y, rot=0, scale=1.0):
    """Variable resistor (potentiometer): resistor + arrow."""
    _draw_resistor_iec(ax, x, y, rot, scale)
    w = 40 * scale
    ax.annotate('', xy=(x + w/2 + 8*scale, y + 8*scale), xytext=(x + w/4, y + 12*scale),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

def _draw_test_point(ax, x, y, rot=0, scale=1.0):
    """Test point: circle with pin."""
    r = 8 * scale
    circle = Circle((x, y), r, fill='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.plot([x, x], [y + r, y + r + 10*scale], 'k-', linewidth=1.2)


# ============================================================
# Component Definitions
# ============================================================

COMPONENTS = {
    'R': ComponentDef('Resistor', 'passive', _draw_resistor_iec, {'1': (-25, 0), '2': (25, 0)}, 60, 20),
    'R_US': ComponentDef('Resistor (US)', 'passive', _draw_resistor_us, {'1': (-25, 0), '2': (25, 0)}, 60, 20),
    'C': ComponentDef('Capacitor', 'passive', _draw_capacitor, {'1': (-16, 0), '2': (16, 0)}, 40, 30),
    'C_POL': ComponentDef('Polarized Cap', 'passive', _draw_polarized_cap, {'1': (-16, 0), '2': (16, 0)}, 40, 30),
    'L': ComponentDef('Inductor', 'passive', _draw_inductor, {'1': (-28, 0), '2': (28, 0)}, 60, 20),
    'D': ComponentDef('Diode', 'semiconductor', _draw_diode, {'A': (-18, 0), 'K': (18, 0)}, 40, 30),
    'LED': ComponentDef('LED', 'semiconductor', _draw_led, {'A': (-18, 0), 'K': (18, 0)}, 40, 30),
    'ZD': ComponentDef('Zener', 'semiconductor', _draw_zener, {'A': (-18, 0), 'K': (18, 0)}, 40, 30),
    'Q_NPN': ComponentDef('NPN BJT', 'semiconductor', _draw_bjt_npn, {'C': (0, 25), 'B': (-25, 0), 'E': (0, -25)}, 50, 50),
    'Q_PMOS': ComponentDef('P-ch MOSFET', 'semiconductor', _draw_mosfet_n, {'D': (0, 25), 'G': (-25, 0), 'S': (0, -25)}, 50, 50),
    'U': ComponentDef('Op-Amp', 'ic', _draw_opamp, {'IN+': (-22, -8), 'IN-': (-22, 8), 'OUT': (22, 0)}, 48, 36),
    'IC': ComponentDef('IC', 'ic', _draw_ic, {f'P{i}': (-34, -16 + i*8) for i in range(1,5)} | {f'P{i+4}': (34, -16 + i*8) for i in range(1,5)}, 68, 48),
    'J': ComponentDef('Connector', 'connector', _draw_connector, {f'{i}': (-18 + i*12, 18) for i in range(1,4)}, 56, 36),
    'Y': ComponentDef('Crystal', 'passive', _draw_crystal, {'1': (-20, 0), '2': (20, 0)}, 48, 20),
    'F': ComponentDef('Fuse', 'passive', _draw_fuse, {'1': (-20, 0), '2': (20, 0)}, 46, 18),
    'RV': ComponentDef('Var Resistor', 'passive', _draw_var_resistor, {'1': (-25, 0), '2': (25, 0), 'W': (5, 15)}, 60, 30),
    'TP': ComponentDef('Test Point', 'connector', _draw_test_point, {'1': (0, 14)}, 20, 30),
    'PWR': ComponentDef('Power Flag', 'power', _draw_power_flag, {'1': (0, 10)}, 24, 26),
    'GND': ComponentDef('Ground', 'power', _draw_gnd, {'1': (0, 0)}, 30, 22),
    'VCC': ComponentDef('VCC', 'power', _draw_vcc, {'1': (0, 0)}, 24, 10),
}

# ============================================================
# Circuit Topology Generator
# ============================================================

class CircuitTopology:
    """Random circuit topology: list of components + nets."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.components = []  # [(refdes, comp_type, value_str, x, y)]
        self.nets = []  # [[(refdes, pin_name), ...]]
        self.ref_counters = defaultdict(int)

    def _next_refdes(self, comp_type):
        # Map component types to proper refdes prefixes
        refdes_map = {
            'R': 'R', 'R_US': 'R', 'RV': 'RV',
            'C': 'C', 'C_POL': 'C',
            'L': 'L',
            'D': 'D', 'LED': 'LED', 'ZD': 'D',
            'Q_NPN': 'Q', 'Q_PMOS': 'Q',
            'U': 'U', 'IC': 'U',
            'J': 'J',
            'Y': 'Y',
            'F': 'F',
            'TP': 'TP',
            'PWR': 'PWR',  # Power flag — no number
            'GND': 'GND',  # Ground — no number
            'VCC': 'VCC',  # VCC — no number
        }
        prefix = refdes_map.get(comp_type, comp_type[0])

        # Power symbols don't get numbered
        if comp_type in ('PWR', 'GND', 'VCC'):
            # Return just the symbol name, no counter
            # But we need uniqueness for nets — use value string instead
            return prefix  # Will be overridden in _add_component

        self.ref_counters[prefix] += 1
        return f"{prefix}{self.ref_counters[prefix]}"

    def generate(self, n_sections=None):
        """Generate a random circuit topology."""
        if n_sections is None:
            n_sections = self.rng.randint(2, 5)

        self.components = []
        self.nets = []
        self.ref_counters = defaultdict(int)

        # Section templates
        sections = [
            self._section_power_input,
            self._section_mcu_basic,
            self._section_analog,
            self._section_output,
            self._section_passive_network,
            self._section_voltage_regulator,
            self._section_filter,
            self._section_interface,
        ]
        chosen = self.rng.sample(sections, min(n_sections, len(sections)))

        for section_func in chosen:
            section_func()

        # Add some random interconnections between sections
        self._add_cross_connections()

        return self

    def _add_component(self, comp_type, value_str=None):
        comp = COMPONENTS[comp_type]
        refdes = self._next_refdes(comp_type)
        if value_str is None:
            value_str = self._random_value(comp_type)
        self.components.append((refdes, comp_type, value_str))
        return refdes

    def _random_value(self, comp_type):
        if comp_type in ('R', 'R_US', 'RV'):
            vals = ['10', '22', '47', '100', '220', '470', '1k', '2.2k', '4.7k', '10k', '22k', '47k', '100k', '220k', '470k', '1M']
            return self.rng.choice(vals)
        elif comp_type in ('C', 'C_POL'):
            vals = ['10pF', '22pF', '100pF', '1nF', '10nF', '22nF', '100nF', '220nF', '1uF', '10uF', '22uF', '47uF', '100uF', '220uF', '470uF']
            return self.rng.choice(vals)
        elif comp_type in ('L',):
            vals = ['1uH', '10uH', '22uH', '47uH', '100uH', '220uH', '470uH', '1mH']
            return self.rng.choice(vals)
        elif comp_type == 'D':
            return self.rng.choice(['1N4148', '1N4007', '1N5819', 'BAT54S', 'SS14', 'MBR0520'])
        elif comp_type == 'LED':
            return self.rng.choice(['RED', 'GREEN', 'BLUE', 'YELLOW', 'WHITE', 'IR', 'RGB', 'ORANGE'])
        elif comp_type == 'ZD':
            return self.rng.choice(['3.3V', '5.1V', '3.6V', '6.2V', '12V', '15V', 'BZX84C3V3', 'MMSZ5231B'])
        elif comp_type in ('Q_NPN',):
            return self.rng.choice(['2N2222', '2N3904', 'BC547', 'S8050', 'MMBT3904'])
        elif comp_type in ('Q_PMOS',):
            return self.rng.choice(['IRF4905', 'AO3401', 'SI2301', 'FDN304P'])
        elif comp_type in ('U',):
            return self.rng.choice(['LM358', 'LMV321', 'TL072', 'MCP6001', 'OPA333'])
        elif comp_type in ('IC',):
            return self.rng.choice(['STM32F103', 'ATmega328P', 'ESP32', 'ATtiny85', 'RP2040', 'CH340G', 'MAX232'])
        elif comp_type in ('J',):
            return self.rng.choice(['CONN_2P', 'CONN_3P', 'HEADER_4P', 'USB-C', 'DB9', 'RJ45', 'TERM_BLK'])
        elif comp_type in ('Y',):
            return self.rng.choice(['8MHz', '12MHz', '16MHz', '25MHz', '32.768kHz'])
        elif comp_type in ('F',):
            return self.rng.choice(['500mA', '1A', '2A', '3A', '5A'])
        elif comp_type in ('PWR',):
            return self.rng.choice(['+5V', '+3.3V', '+12V', 'VCC', 'VDD', '-12V'])
        elif comp_type in ('GND',):
            return 'GND'
        elif comp_type in ('VCC',):
            return self.rng.choice(['+5V', '+3.3V', 'VCC'])
        elif comp_type in ('TP',):
            return 'TP'
        return '?'

    def _create_net(self, *connections):
        """Create a net connecting multiple (refdes, pin) pairs."""
        self.nets.append(list(connections))

    # ---- Section generators ----

    def _section_power_input(self):
        """Power input: connector → fuse → diode (protection) → capacitors → VCC + GND."""
        j1 = self._add_component('J', 'DC_IN')
        f1 = self._add_component('F', '500mA')
        d1 = self._add_component('D', '1N4007')
        c1 = self._add_component('C', '100uF')
        c2 = self._add_component('C', '100nF')
        pwr = self._add_component('PWR', '+5V')
        gnd1 = self._add_component('GND')
        gnd2 = self._add_component('GND')

        self._create_net((j1, '1'), (f1, '1'))
        self._create_net((f1, '2'), (d1, 'A'))
        self._create_net((d1, 'K'), (c1, '1'), (c2, '1'), (pwr, '1'))
        self._create_net((j1, '2'), (gnd1, '1'))
        self._create_net((c1, '2'), (c2, '2'), (gnd2, '1'))

    def _section_mcu_basic(self):
        """MCU: IC + decoupling caps + crystal + reset."""
        ic = self._add_component('IC', 'ATmega328P')
        c_d1 = self._add_component('C', '100nF')
        c_d2 = self._add_component('C', '100nF')
        c_d3 = self._add_component('C', '10uF')
        y1 = self._add_component('Y', '16MHz')
        c_x1 = self._add_component('C', '22pF')
        c_x2 = self._add_component('C', '22pF')
        r_pu = self._add_component('R', '10k')
        gnd_ic = self._add_component('GND')

        self._create_net((ic, 'P1'), (c_d1, '1'))
        self._create_net((ic, 'P2'), (c_d2, '1'))
        self._create_net((c_d1, '2'), (gnd_ic, '1'))
        self._create_net((c_d2, '2'), (gnd_ic, '1'))
        self._create_net((ic, 'P3'), (y1, '1'), (c_x1, '1'))
        self._create_net((ic, 'P4'), (y1, '2'), (c_x2, '1'))
        self._create_net((c_x1, '2'), (gnd_ic, '1'))
        self._create_net((c_x2, '2'), (gnd_ic, '1'))
        self._create_net((ic, 'P5'), (r_pu, '1'))

    def _section_analog(self):
        """Op-amp + resistor network."""
        u1 = self._add_component('U', 'LM358')
        r1 = self._add_component('R', '10k')
        r2 = self._add_component('R', '100k')
        r3 = self._add_component('R', '10k')
        c1 = self._add_component('C', '100nF')

        self._create_net((u1, 'IN-'), (r1, '2'), (r2, '1'), (c1, '1'))
        self._create_net((u1, 'OUT'), (r2, '2'))
        self._create_net((r1, '1'), (r3, '2'))

    def _section_output(self):
        """Output section: connectors + protection."""
        j_out = self._add_component('J', 'CONN_3P')
        r_s1 = self._add_component('R', '220')
        r_s2 = self._add_component('R', '220')
        led = self._add_component('LED')
        gnd_out = self._add_component('GND')

        self._create_net((j_out, '1'), (r_s1, '1'))
        self._create_net((j_out, '2'), (r_s2, '1'))
        self._create_net((r_s1, '2'), (led, 'A'))
        self._create_net((led, 'K'), (gnd_out, '1'))

    def _section_passive_network(self):
        """Network of passives: R divider, RC filter."""
        r_top = self._add_component('R', '47k')
        r_bot = self._add_component('R', '10k')
        c_filt = self._add_component('C', '100nF')
        r_load = self._add_component('R', '1k')
        gnd = self._add_component('GND')

        self._create_net((r_top, '2'), (r_bot, '1'), (c_filt, '1'))
        self._create_net((r_bot, '2'), (gnd, '1'))
        self._create_net((c_filt, '2'), (r_load, '1'))

    def _section_voltage_regulator(self):
        """Voltage regulator: IC + caps."""
        r_in = self._add_component('R', '10')
        c_in = self._add_component('C_POL', '10uF')
        c_out = self._add_component('C_POL', '100uF')
        c_byp = self._add_component('C', '100nF')
        led_pwr = self._add_component('LED')
        r_led = self._add_component('R', '1k')
        gnd1 = self._add_component('GND')
        gnd2 = self._add_component('GND')

        self._create_net((r_in, '1'), (c_in, '1'))
        self._create_net((c_in, '2'), (gnd1, '1'))
        self._create_net((c_out, '1'), (c_byp, '1'), (r_led, '1'))
        self._create_net((c_out, '2'), (c_byp, '2'), (gnd2, '1'))
        self._create_net((r_led, '2'), (led_pwr, 'A'))

    def _section_filter(self):
        """RC/LC filter section."""
        r_f = self._add_component('R', '100')
        c_f1 = self._add_component('C', '10uF')
        c_f2 = self._add_component('C', '100nF')
        l_f = self._add_component('L', '10uH')
        gnd = self._add_component('GND')

        self._create_net((r_f, '2'), (l_f, '1'))
        self._create_net((l_f, '2'), (c_f1, '1'), (c_f2, '1'))
        self._create_net((c_f1, '2'), (c_f2, '2'), (gnd, '1'))

    def _section_interface(self):
        """Interface section: level shifters, protection."""
        r_pu1 = self._add_component('R', '10k')
        r_pu2 = self._add_component('R', '10k')
        d_clamp1 = self._add_component('ZD', '3.3V')
        d_clamp2 = self._add_component('ZD', '3.3V')
        j_int = self._add_component('J', 'HEADER_4P')
        gnd = self._add_component('GND')

        self._create_net((j_int, '1'), (r_pu1, '1'), (d_clamp1, 'K'))
        self._create_net((j_int, '2'), (r_pu2, '1'), (d_clamp2, 'K'))
        self._create_net((d_clamp1, 'A'), (gnd, '1'))
        self._create_net((d_clamp2, 'A'), (gnd, '1'))

    def _add_cross_connections(self):
        """Add a few random connections between nets for realism."""
        if len(self.nets) >= 3:
            # Connect some floating pins to power/ground nets
            pass  # Handled by layout

    def get_gt_text(self):
        """Generate GT label text: refdes + value pairs separated by newlines."""
        lines = []
        for refdes, comp_type, value_str in self.components:
            # For power symbols, use descriptive type labels
            if comp_type == 'GND':
                display_name = 'GND'
            elif comp_type == 'PWR':
                display_name = 'PWR_FLAG'
            elif comp_type == 'VCC':
                display_name = 'VCC'
            else:
                display_name = refdes
            lines.append(f"{display_name}\n{value_str}")
        return '\n'.join(lines)


# ============================================================
# Layout Engine
# ============================================================

class LayoutEngine:
    """Place components on a canvas and route wires."""

    def __init__(self, topology, canvas_width=800, canvas_height=600, seed=None):
        self.topo = topology
        self.canvas_w = canvas_width
        self.canvas_h = canvas_height
        self.rng = random.Random(seed)
        self.positions = {}  # refdes -> (x, y)
        self.pin_positions = {}  # (refdes, pin_name) -> (x, y)
        self.wire_segments = []  # [(x1,y1, x2,y2), ...]
        self.junction_points = []  # [(x, y), ...]

    def place_components(self):
        """Place components in a grid with some randomness."""
        n_comps = len(self.topo.components)
        if n_comps == 0:
            return

        # Determine grid
        cols = max(3, int(math.sqrt(n_comps * self.canvas_w / self.canvas_h)))
        rows = max(2, (n_comps + cols - 1) // cols)

        cell_w = (self.canvas_w - 100) / cols
        cell_h = (self.canvas_h - 100) / rows

        # Group components by category for better layout
        categories = defaultdict(list)
        for i, (refdes, comp_type, value_str) in enumerate(self.topo.components):
            comp = COMPONENTS[comp_type]
            categories[comp.category].append(i)

        # Interleave categories for visual variety
        all_indices = []
        cat_lists = list(categories.values())
        self.rng.shuffle(cat_lists)
        max_len = max(len(c) for c in cat_lists)
        for j in range(max_len):
            for cl in cat_lists:
                if j < len(cl):
                    all_indices.append(cl[j])

        for grid_idx, comp_idx in enumerate(all_indices):
            refdes, comp_type, value_str = self.topo.components[comp_idx]
            col = grid_idx % cols
            row = grid_idx // cols

            # Jitter within cell
            jx = self.rng.uniform(-cell_w * 0.15, cell_w * 0.15)
            jy = self.rng.uniform(-cell_h * 0.15, cell_h * 0.15)

            x = 60 + col * cell_w + cell_w / 2 + jx
            y = 50 + row * cell_h + cell_h / 2 + jy

            # Clamp to canvas
            x = max(30, min(self.canvas_w - 30, x))
            y = max(30, min(self.canvas_h - 30, y))

            self.positions[refdes] = (x, y)

            # Compute pin world positions
            comp = COMPONENTS[comp_type]
            pin_pos = comp.get_pin_positions(x, y, rotation=0)
            for pin_name, (px, py) in pin_pos.items():
                self.pin_positions[(refdes, pin_name)] = (px, py)

    def route_wires(self):
        """Manhattan routing for all nets."""
        self.wire_segments = []
        self.junction_points = []

        for net_idx, net in enumerate(self.topo.nets):
            if len(net) < 2:
                continue

            # Get all pin positions in this net
            points = []
            for refdes, pin_name in net:
                key = (refdes, pin_name)
                if key in self.pin_positions:
                    points.append(self.pin_positions[key])

            if len(points) < 2:
                continue

            # Route with a star topology: connect all points to a common junction
            if len(points) <= 3:
                # Direct connections in a chain
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    self._manhattan_route(p1, p2)
            else:
                # Star: find centroid, route each pin to it
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                junction = (cx, cy)
                self.junction_points.append(junction)
                for p in points:
                    self._manhattan_route(p, junction)

    def _manhattan_route(self, p1, p2):
        """Route from p1 to p2 using 1 bend (L-shaped)."""
        x1, y1 = p1
        x2, y2 = p2

        if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
            # Almost aligned, straight line
            self.wire_segments.append((x1, y1, x2, y2))
        else:
            # Choose bend direction (random for variety)
            if self.rng.random() < 0.5:
                # Horizontal then vertical
                self.wire_segments.append((x1, y1, x2, y1))
                self.wire_segments.append((x2, y1, x2, y2))
            else:
                # Vertical then horizontal
                self.wire_segments.append((x1, y1, x1, y2))
                self.wire_segments.append((x1, y2, x2, y2))


# ============================================================
# Renderer
# ============================================================

class SchematicRenderer:
    """Render circuit to matplotlib figure and save as PNG."""

    def __init__(self, topology, layout, canvas_w=800, canvas_h=600, dpi=100, seed=None):
        self.topo = topology
        self.layout = layout
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.dpi = dpi
        self.rng = random.Random(seed)

    def render(self, output_path):
        """Render and save to PNG."""
        fig_w = self.canvas_w / self.dpi
        fig_h = self.canvas_h / self.dpi

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=self.dpi)
        ax.set_xlim(0, self.canvas_w)
        ax.set_ylim(0, self.canvas_h)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')

        # Draw title block border
        self._draw_border(ax)

        # Draw wires
        for x1, y1, x2, y2 in self.layout.wire_segments:
            # Clip to canvas
            x1 = max(5, min(self.canvas_w - 5, x1))
            y1 = max(5, min(self.canvas_h - 5, y1))
            x2 = max(5, min(self.canvas_w - 5, x2))
            y2 = max(5, min(self.canvas_h - 5, y2))
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.2, alpha=0.8)

        # Draw junction dots
        for jx, jy in self.layout.junction_points:
            jx = max(5, min(self.canvas_w - 5, jx))
            jy = max(5, min(self.canvas_h - 5, jy))
            ax.plot(jx, jy, 'ko', markersize=3)

        # Draw components
        for refdes, comp_type, value_str in self.topo.components:
            if refdes not in self.layout.positions:
                continue
            x, y = self.layout.positions[refdes]
            x = max(20, min(self.canvas_w - 20, x))
            y = max(20, min(self.canvas_h - 20, y))
            comp = COMPONENTS[comp_type]

            # Rotate occasionally for variety
            rot = 0
            if self.rng.random() < 0.2:
                rot = self.rng.choice([90, 270])

            comp.draw_func(ax, x, y, rot, scale=1.0)

            # Draw refdes label (above or beside)
            label_x = x + self.rng.uniform(-10, 10)
            label_y = y + comp.height / 2 + 12
            ax.text(label_x, label_y, refdes, fontsize=7, ha='center', va='bottom',
                    fontfamily='monospace', fontweight='bold', color='#222222',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7))

            # Draw value label (below or beside)
            val_x = x + self.rng.uniform(-10, 10)
            val_y = y - comp.height / 2 - 8
            ax.text(val_x, val_y, value_str, fontsize=6.5, ha='center', va='top',
                    fontfamily='monospace', color='#444444',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7))

        plt.tight_layout(pad=0)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    def _draw_border(self, ax):
        """Draw a title-block-style border."""
        w, h = self.canvas_w, self.canvas_h
        margin = 8
        ax.plot([margin, w - margin, w - margin, margin, margin],
                [margin, margin, h - margin, h - margin, margin],
                'k-', linewidth=1.5, alpha=0.3)
        # Title block corner
        tb_w, tb_h = 160, 40
        tb_x, tb_y = w - margin - tb_w, margin
        ax.plot([tb_x, tb_x + tb_w, tb_x + tb_w, tb_x, tb_x],
                [tb_y, tb_y, tb_y + tb_h, tb_y + tb_h, tb_y],
                'k-', linewidth=0.8, alpha=0.3)


# ============================================================
# Batch Generator
# ============================================================

def generate_one_circuit(output_path, seed=None, canvas_w=800, canvas_h=600):
    """Generate a single circuit schematic and return GT text."""
    rng = random.Random(seed)
    topo_seed = rng.randint(0, 2**31)
    layout_seed = rng.randint(0, 2**31)

    # Generate topology
    topo = CircuitTopology(seed=topo_seed)
    n_sections = rng.randint(2, 6)
    topo.generate(n_sections=n_sections)

    # Layout
    layout = LayoutEngine(topo, canvas_width=canvas_w, canvas_height=canvas_h, seed=layout_seed)
    layout.place_components()
    layout.route_wires()

    # Render
    renderer = SchematicRenderer(topo, layout, canvas_w=canvas_w, canvas_h=canvas_h, dpi=100, seed=rng.randint(0, 2**31))
    renderer.render(output_path)

    return topo.get_gt_text()


def batch_generate(output_dir, n_samples, start_idx=0, canvas_sizes=None):
    """Generate batch of circuit schematics with JSONL output."""
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, 'train_synthetic.jsonl')
    if canvas_sizes is None:
        canvas_sizes = [(800, 600), (900, 650), (750, 550), (850, 620)]

    t0 = time.time()
    log_interval = max(1, n_samples // 20)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i in range(start_idx, start_idx + n_samples):
            seed = i * 137 + 42
            cw, ch = canvas_sizes[i % len(canvas_sizes)]

            img_name = f's{i:05d}.png'
            img_path = os.path.join(img_dir, img_name)

            try:
                gt_text = generate_one_circuit(img_path, seed=seed, canvas_w=cw, canvas_h=ch)
            except Exception as e:
                print(f"[GEN] Error at {i}: {e}", flush=True)
                continue

            # Build JSONL entry matching training format
            entry = {
                'images': [img_path.replace('\\', '/')],
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': 'OCR:'}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': gt_text
                    }
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            if (i - start_idx + 1) % log_interval == 0:
                elapsed = (time.time() - t0) / 60
                done = i - start_idx + 1
                rate = done / max(elapsed, 0.01)
                remaining = (n_samples - done) / max(rate, 0.01)
                print(f"[GEN] {done}/{n_samples} ({rate:.0f}/min) ETA={remaining:.0f}min | last: {img_name}", flush=True)

    tt = (time.time() - t0) / 60
    print(f"\n[GEN] DONE: {n_samples} circuits in {tt:.0f}min ({n_samples/tt:.0f}/min)", flush=True)
    return jsonl_path, img_dir


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=5000, help='Number of circuits to generate')
    ap.add_argument('--start_idx', type=int, default=0)
    ap.add_argument('--output_dir', default='g:/mimo_project/circuit_ocr/output/synthetic_circuits')
    ap.add_argument('--test', action='store_true', help='Generate just 5 for testing')
    args = ap.parse_args()

    if args.test:
        args.n_samples = 5
        args.output_dir += '_test'

    print(f"[GEN] Generating {args.n_samples} synthetic circuit schematics...", flush=True)
    print(f"[GEN] Output: {args.output_dir}", flush=True)

    jsonl_path, img_dir = batch_generate(
        args.output_dir, args.n_samples, args.start_idx,
        canvas_sizes=[(800, 600), (900, 650), (750, 550), (850, 620), (780, 580)]
    )

    print(f"[GEN] JSONL: {jsonl_path}", flush=True)
    print(f"[GEN] Images: {img_dir}", flush=True)
