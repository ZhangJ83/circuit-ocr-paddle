"""V2: High-quality synthetic circuit schematic generator.
KiCad-realistic renderer: dot grid, proper line weights, standard symbols,
title block, off-white background, anti-aliased text.

Uses matplotlib with custom styling to match KiCad's visual appearance.
"""
import os, sys, json, time, random, math, argparse
import numpy as np
from collections import defaultdict

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, Polygon, FancyArrow,
                                 Arc, FancyBboxPatch, PathPatch, Ellipse)
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ============================================================
# KiCad-like color scheme
# ============================================================
BG_COLOR = '#FFFDF8'      # Warm white (like paper)
GRID_COLOR = '#E8E4D8'    # Subtle grid dots
WIRE_COLOR = '#1A1A2E'    # Dark blue-black for wires
SYMBOL_COLOR = '#1A1A2E'  # Same for component outlines
TEXT_COLOR = '#1A1A2E'    # Primary text
VALUE_COLOR = '#444455'   # Slightly lighter for values
BORDER_COLOR = '#8888A0'  # Border lines
JUNCTION_COLOR = '#1A1A2E'

WIRE_WIDTH = 1.0
SYMBOL_WIDTH = 1.6
TEXT_SIZE_REFDES = 7.5
TEXT_SIZE_VALUE = 6.5
TEXT_SIZE_NET = 7.0
GRID_SPACING = 25  # pixels between grid dots
GRID_DOT_SIZE = 1.0

# ============================================================
# Component Drawing Functions (KiCad-style)
# ============================================================

def _resistor_us(ax, x, y, scale=1.0):
    """US zigzag resistor — classic American style."""
    w, h = 50*scale, 20*scale
    n = 4  # number of zigzag segments
    xs, ys = [], []
    xs.append(x - w/2); ys.append(y)
    for i in range(n):
        frac = (i + 0.5) / n
        cx = x - w/2 + frac * w
        sign = 1 if i % 2 == 0 else -1
        xs.append(cx); ys.append(y + sign * h/2)
    xs.append(x + w/2); ys.append(y)
    ax.plot(xs, ys, color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH, solid_capstyle='round')

def _resistor_iec(ax, x, y, scale=1.0):
    """IEC rectangular resistor."""
    w, h = 44*scale, 18*scale
    rect = Rectangle((x-w/2, y-h/2), w, h, fill='none', edgecolor=SYMBOL_COLOR,
                     linewidth=SYMBOL_WIDTH, capstyle='round')
    ax.add_patch(rect)
    # Lead wires extend beyond rect
    ax.plot([x-w/2-8*scale, x-w/2], [y, y], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)
    ax.plot([x+w/2, x+w/2+8*scale], [y, y], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)

def _capacitor(ax, x, y, scale=1.0):
    """Capacitor: two parallel plates."""
    gap, lh = 10*scale, 16*scale
    for sx, lw in [(-1, 2.0), (1, 2.0)]:
        ax.plot([x + sx*gap, x + sx*gap], [y-lh, y+lh], color=SYMBOL_COLOR,
                linewidth=lw, solid_capstyle='butt')

def _capacitor_polarized(ax, x, y, scale=1.0):
    """Polarized capacitor."""
    gap, lh = 10*scale, 16*scale
    ax.plot([x-gap, x-gap], [y-lh, y+lh], color=SYMBOL_COLOR, linewidth=2.0)
    ax.plot([x+gap, x+gap], [y-lh, y+lh], color=SYMBOL_COLOR, linewidth=2.0)
    # Plus sign
    ax.text(x+gap+10*scale, y, '+', fontsize=6, color=SYMBOL_COLOR, ha='center', va='center')

def _inductor(ax, x, y, scale=1.0):
    """Inductor: series of semicircles."""
    w, amp = 50*scale, 12*scale
    n_loops = 4
    loop_w = w / n_loops
    for i in range(n_loops):
        cx = x - w/2 + loop_w*(i + 0.5)
        arc = Arc((cx, y), loop_w*0.9, amp*2, angle=0, theta1=180, theta2=360,
                  color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
        ax.add_patch(arc)

def _diode(ax, x, y, scale=1.0):
    """Diode: triangle + bar."""
    w, h = 28*scale, 14*scale
    tri = Polygon([(x-w/2, y), (x+w/2, y-h), (x+w/2, y+h)],
                  fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(tri)
    ax.plot([x+w/2, x+w/2], [y-h, y+h], color=SYMBOL_COLOR, linewidth=2.0)

def _led(ax, x, y, scale=1.0):
    """LED: diode + light arrows."""
    _diode(ax, x, y, scale)
    # Arrows
    for arrow_y in [y-16*scale, y+16*scale]:
        ax.annotate('', xy=(x+10*scale, arrow_y), xytext=(x-4*scale, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.0))

def _zener(ax, x, y, scale=1.0):
    """Zener diode."""
    w, h = 28*scale, 14*scale
    tri = Polygon([(x-w/2, y), (x+w/2, y-h), (x+w/2, y+h)],
                  fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(tri)
    ax.plot([x+w/2, x+w/2+5*scale], [y-h, y-h-3*scale], color=SYMBOL_COLOR, linewidth=2.0)

def _bjt_npn(ax, x, y, scale=1.0):
    """NPN bipolar junction transistor."""
    r = 22*scale
    circle = Circle((x, y), r, fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(circle)
    # C (top, no arrow), B (left, no arrow), E (bottom, arrow)
    ax.plot([x, x], [y+r, y+r+12*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.plot([x-r, x-r-12*scale], [y, y], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.plot([x, x], [y-r, y-r-12*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    # Emitter arrow
    arrow_y = y - r - 5*scale
    ax.annotate('', xy=(x+6*scale, arrow_y), xytext=(x-6*scale, arrow_y),
                arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.2))

def _bjt_pnp(ax, x, y, scale=1.0):
    """PNP bipolar junction transistor."""
    r = 22*scale
    circle = Circle((x, y), r, fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(circle)
    ax.plot([x, x], [y+r, y+r+12*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.plot([x-r, x-r-12*scale], [y, y], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.plot([x, x], [y-r, y-r-12*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    # Emitter arrow (inverted direction for PNP)
    arrow_y = y - r - 5*scale
    ax.annotate('', xy=(x-6*scale, arrow_y), xytext=(x+6*scale, arrow_y),
                arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.2))

def _mosfet_n(ax, x, y, scale=1.0):
    """N-channel MOSFET."""
    r = 22*scale
    circle = Circle((x, y), r, fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(circle)
    ax.plot([x, x], [y+r, y+r+12*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)  # D
    ax.plot([x-r-8*scale, x-r], [y, y], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)  # G (gap)
    ax.plot([x, x], [y-r, y-r-12*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)  # S
    ax.annotate('', xy=(x+6*scale, y-r-5*scale), xytext=(x-6*scale, y-r-5*scale),
                arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.2))

def _opamp(ax, x, y, scale=1.0):
    """Operational amplifier."""
    w, h = 42*scale, 32*scale
    tri = Polygon([(x-w/2, y-h), (x-w/2, y+h), (x+w/2, y)],
                  fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(tri)
    ax.text(x-w/2+6*scale, y-h/3, '+', fontsize=5, color=SYMBOL_COLOR, ha='center', va='center')
    ax.text(x-w/2+6*scale, y+h/3, '-', fontsize=5, color=SYMBOL_COLOR, ha='center', va='center')
    # Output pin extends
    ax.plot([x+w/2, x+w/2+10*scale], [y, y], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)

def _ic(ax, x, y, scale=1.0):
    """IC: rectangle with pin stubs."""
    w, h = 64*scale, 48*scale
    rect = Rectangle((x-w/2, y-h/2), w, h, fill='none', edgecolor=SYMBOL_COLOR,
                     linewidth=SYMBOL_WIDTH)
    ax.add_patch(rect)
    # Pin stubs
    n_pins_per_side = 4
    for side, dx, dy in [('left', -w/2, 0), ('right', w/2, 0)]:
        for i in range(n_pins_per_side):
            py = y - h/2 + h*(i+0.5)/n_pins_per_side
            ax.plot([dx, dx+(8*scale if side=='left' else -8*scale)], [py, py],
                    color=SYMBOL_COLOR, linewidth=1.0)
            ax.plot([dx+(8*scale if side=='left' else -8*scale),
                     dx+(8*scale if side=='left' else -8*scale)],
                    [py, py], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)
    # Pin 1 marker
    ax.plot(x-w/4-2*scale, y+h/2-4*scale, 'ko', markersize=3)

def _connector(ax, x, y, scale=1.0):
    """Header/connector."""
    w, h = 48*scale, 28*scale
    # Open-bottom rectangle
    ax.plot([x-w/2, x-w/2, x+w/2, x+w/2], [y-h/2, y+h/2, y+h/2, y-h/2],
            color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    # Pin stubs
    n_pins = 3
    for i in range(n_pins):
        px = x - w/2 + w*(i+0.25)/2
        ax.plot([px, px], [y+h/2, y+h/2+10*scale], color=SYMBOL_COLOR, linewidth=1.0)

def _crystal(ax, x, y, scale=1.0):
    """Crystal oscillator."""
    w, h = 28*scale, 16*scale
    rect = Rectangle((x-w/2, y-h/2), w, h, fill='none', edgecolor=SYMBOL_COLOR,
                     linewidth=SYMBOL_WIDTH)
    ax.add_patch(rect)
    # Ground plate stubs
    for sx in [-1, 1]:
        ax.plot([x+sx*(w/2+6*scale), x+sx*(w/2+6*scale)], [y-h/2-4*scale, y+h/2+4*scale],
                color=SYMBOL_COLOR, linewidth=1.0)

def _fuse(ax, x, y, scale=1.0):
    """Fuse."""
    w, h = 28*scale, 12*scale
    rect = Rectangle((x-w/2, y-h/2), w, h, fill='none', edgecolor=SYMBOL_COLOR,
                     linewidth=SYMBOL_WIDTH)
    ax.add_patch(rect)
    ax.plot([x-w/2-8*scale, x+w/2+8*scale], [y, y], color=SYMBOL_COLOR, linewidth=1.2)

def _test_point(ax, x, y, scale=1.0):
    """Test point."""
    r = 7*scale
    circle = Circle((x, y), r, fill='none', edgecolor=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.add_patch(circle)
    ax.plot([x, x], [y+r, y+r+12*scale], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)

def _var_resistor(ax, x, y, scale=1.0):
    """Variable resistor / potentiometer."""
    _resistor_iec(ax, x, y, scale)
    w = 44*scale
    ax.annotate('', xy=(x+w/2+6*scale, y+10*scale), xytext=(x+w/4, y+14*scale),
                arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.0))

def _power_flag(ax, x, y, scale=1.0):
    """Power flag: arrow pointing to a horizontal bar."""
    arrow_h = 22*scale
    ax.plot([x, x], [y-arrow_h/2, y+arrow_h/2], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    # Arrowhead
    head_w = 10*scale
    ax.annotate('', xy=(x, y+arrow_h/2), xytext=(x, y+arrow_h/2-head_w),
                arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.5))

def _ground(ax, x, y, scale=1.0):
    """Ground symbol: three decreasing horizontal lines + vertical."""
    ax.plot([x, x], [y, y-16*scale], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)
    lw = [28*scale, 18*scale, 10*scale]
    for i, w in enumerate(lw):
        gy = y - i*5*scale
        ax.plot([x-w/2, x+w/2], [gy, gy], color=SYMBOL_COLOR, linewidth=1.5)

def _vcc_symbol(ax, x, y, scale=1.0):
    """VCC: horizontal bar."""
    ax.plot([x-12*scale, x+12*scale], [y, y], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    # Small connection dot
    ax.plot(x, y, 'o', color=SYMBOL_COLOR, markersize=2.5)

def _potentiometer(ax, x, y, scale=1.0):
    """Potentiometer: resistor + arrow wiper."""
    _resistor_iec(ax, x, y, scale)
    ax.annotate('', xy=(x+12*scale, y+14*scale), xytext=(x-8*scale, y+18*scale),
                arrowprops=dict(arrowstyle='->', color=SYMBOL_COLOR, lw=1.0))

def _switch_spst(ax, x, y, scale=1.0):
    """SPST switch."""
    ax.plot([x-14*scale, x], [y, y], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)
    ax.plot([x, x+6*scale], [y, y+10*scale], color=SYMBOL_COLOR, linewidth=SYMBOL_WIDTH)
    ax.plot([x+14*scale, x+14*scale], [y, y], color=SYMBOL_COLOR, linewidth=WIRE_WIDTH)
    # Circle contact
    ax.plot(x+14*scale, y, 'o', color=SYMBOL_COLOR, markersize=3)

def _ferrite_bead(ax, x, y, scale=1.0):
    """Ferrite bead: resistor-like with text."""
    _resistor_iec(ax, x, y, scale)
    ax.text(x, y+14*scale, 'FB', fontsize=5, color=SYMBOL_COLOR, ha='center')

# ============================================================
# Component Registry
# ============================================================

class ComponentDef:
    def __init__(self, name, category, draw_func, pins, width=60, height=40):
        self.name = name
        self.category = category
        self.draw_func = draw_func
        self.pins = pins
        self.width = width
        self.height = height

COMPONENTS = {
    'R':       ComponentDef('Resistor IEC', 'passive', _resistor_iec, {'1': (-26, 0), '2': (26, 0)}, 56, 20),
    'R_US':    ComponentDef('Resistor US', 'passive', _resistor_us, {'1': (-26, 0), '2': (26, 0)}, 56, 20),
    'C':       ComponentDef('Capacitor', 'passive', _capacitor, {'1': (-14, 0), '2': (14, 0)}, 32, 32),
    'C_POL':   ComponentDef('Polarized Cap', 'passive', _capacitor_polarized, {'1': (-14, 0), '2': (14, 0)}, 32, 32),
    'L':       ComponentDef('Inductor', 'passive', _inductor, {'1': (-28, 0), '2': (28, 0)}, 56, 24),
    'FB':      ComponentDef('Ferrite Bead', 'passive', _ferrite_bead, {'1': (-26, 0), '2': (26, 0)}, 56, 20),
    'D':       ComponentDef('Diode', 'semiconductor', _diode, {'A': (-16, 0), 'K': (18, 0)}, 36, 30),
    'LED':     ComponentDef('LED', 'semiconductor', _led, {'A': (-16, 0), 'K': (18, 0)}, 36, 30),
    'ZD':      ComponentDef('Zener Diode', 'semiconductor', _zener, {'A': (-16, 0), 'K': (18, 0)}, 36, 30),
    'Q_NPN':   ComponentDef('NPN BJT', 'semiconductor', _bjt_npn, {'C': (0, 28), 'B': (-28, 0), 'E': (0, -28)}, 56, 56),
    'Q_PNP':   ComponentDef('PNP BJT', 'semiconductor', _bjt_pnp, {'C': (0, 28), 'B': (-28, 0), 'E': (0, -28)}, 56, 56),
    'Q_NMOS':  ComponentDef('N-ch MOSFET', 'semiconductor', _mosfet_n, {'D': (0, 28), 'G': (-28, 0), 'S': (0, -28)}, 56, 56),
    'U':       ComponentDef('Op-Amp', 'ic', _opamp, {'IN+': (-22, -10), 'IN-': (-22, 10), 'OUT': (24, 0)}, 54, 36),
    'IC':      ComponentDef('IC', 'ic', _ic, {f'P{i}': (-36, -20+i*10) for i in range(1,5)} | {f'P{i+4}': (36, -20+i*10) for i in range(1,5)}, 72, 52),
    'J':       ComponentDef('Connector', 'connector', _connector, {f'{i}': (-22+i*14, 16) for i in range(1,4)}, 52, 40),
    'Y':       ComponentDef('Crystal', 'passive', _crystal, {'1': (-20, 0), '2': (20, 0)}, 46, 24),
    'F':       ComponentDef('Fuse', 'passive', _fuse, {'1': (-18, 0), '2': (18, 0)}, 40, 16),
    'RV':      ComponentDef('Var Resistor', 'passive', _var_resistor, {'1': (-26, 0), '2': (26, 0), 'W': (6, 18)}, 56, 30),
    'TP':      ComponentDef('Test Point', 'connector', _test_point, {'1': (0, 18)}, 18, 36),
    'SW':      ComponentDef('Switch SPST', 'connector', _switch_spst, {'1': (-18, 0), '2': (18, 0)}, 40, 24),
    'POT':     ComponentDef('Potentiometer', 'passive', _potentiometer, {'1': (-26, 0), '2': (26, 0), 'W': (-2, 18)}, 56, 30),
    'PWR_FLAG': ComponentDef('Power Flag', 'power', _power_flag, {'1': (0, 14)}, 24, 28),
    'GND':     ComponentDef('Ground', 'power', _ground, {'1': (0, 0)}, 30, 24),
    'VCC':     ComponentDef('VCC Symbol', 'power', _vcc_symbol, {'1': (0, 0)}, 26, 14),
}

# ============================================================
# Circuit Topology Generator
# ============================================================

class CircuitTopology:
    """Generate a random but realistic circuit topology."""

    REFDES_MAP = {
        'R': 'R', 'R_US': 'R', 'RV': 'R', 'POT': 'R',
        'C': 'C', 'C_POL': 'C',
        'L': 'L', 'FB': 'FB',
        'D': 'D', 'LED': 'LED', 'ZD': 'D',
        'Q_NPN': 'Q', 'Q_PNP': 'Q', 'Q_NMOS': 'Q',
        'U': 'U', 'IC': 'U',
        'J': 'J', 'SW': 'SW',
        'Y': 'Y', 'F': 'F', 'TP': 'TP',
        'PWR_FLAG': 'PWR_FLAG', 'GND': 'GND', 'VCC': 'VCC',
    }

    POWER_LABELS = {'PWR_FLAG', 'GND', 'VCC'}

    VALUE_POOLS = {
        'R': ['10', '22', '47', '100', '220', '470', '1k', '2.2k', '4.7k', '10k', '22k', '47k', '100k', '220k', '470k', '1M', '2.2M'],
        'R_US': ['10', '22', '47', '100', '220', '470', '1k', '2.2k', '4.7k', '10k', '22k', '47k', '100k', '220k', '470k', '1M'],
        'RV': ['10k', '50k', '100k', '500k', '1M'],
        'POT': ['10k', '50k', '100k', '500k', '1M'],
        'C': ['10pF', '22pF', '47pF', '100pF', '220pF', '1nF', '10nF', '22nF', '47nF', '100nF', '220nF', '470nF', '1uF'],
        'C_POL': ['1uF', '2.2uF', '4.7uF', '10uF', '22uF', '47uF', '100uF', '220uF', '470uF', '1000uF'],
        'L': ['1uH', '10uH', '22uH', '47uH', '100uH', '220uH', '470uH', '1mH'],
        'FB': ['120R', '300R', '600R', '1k@100MHz'],
        'D': ['1N4148', '1N4007', '1N5819', 'SS14', 'BAT54S', 'MBR0520', 'BAV99'],
        'LED': ['RED', 'GREEN', 'BLUE', 'YELLOW', 'WHITE', 'ORANGE'],
        'ZD': ['3.3V', '3.6V', '5.1V', '6.2V', '12V', '15V', 'BZX84C3V3'],
        'Q_NPN': ['2N2222', '2N3904', 'BC547', 'S8050', 'MMBT3904', 'BC848'],
        'Q_PNP': ['2N2907', '2N3906', 'BC557', 'S8550', 'MMBT3906'],
        'Q_NMOS': ['2N7002', 'BSS138', 'AO3400', 'SI2302', 'IRLML2502'],
        'U': ['LM358', 'LMV321', 'TL072', 'MCP6001', 'OPA333', 'LM324'],
        'IC': ['STM32F103', 'ATmega328P', 'ESP32', 'ATtiny85', 'RP2040', 'CH340G', 'MAX232', 'PCA9685'],
        'J': ['CONN_2P', 'CONN_3P', 'HEADER_4P', 'USB-C', 'TERM_BLK', 'PINHD_2X3'],
        'Y': ['8MHz', '12MHz', '16MHz', '25MHz', '32.768kHz'],
        'F': ['500mA', '1A', '2A', '3A', '5A', '10A'],
        'TP': ['TP'],
        'SW': ['SW'],
        'PWR_FLAG': ['+5V', '+3.3V', '+12V', 'VCC', 'VDD', '+1.8V', '+2.5V'],
        'GND': ['GND'],
        'VCC': ['+5V', '+3.3V', 'VCC', 'VDD'],
    }

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.components = []
        self.nets = []
        self.counters = defaultdict(int)

    def _add(self, comp_type, value=None):
        if value is None:
            pool = self.VALUE_POOLS.get(comp_type, ['?'])
            value = self.rng.choice(pool)

        prefix = self.REFDES_MAP.get(comp_type, comp_type[0])
        if comp_type in self.POWER_LABELS:
            refdes = prefix  # No numbering
        else:
            self.counters[prefix] += 1
            refdes = f"{prefix}{self.counters[prefix]}"

        self.components.append((refdes, comp_type, value))
        return refdes

    def _net(self, *pins):
        self.nets.append(list(pins))

    def generate(self, n_sections=None):
        if n_sections is None:
            n_sections = self.rng.randint(3, 6)

        sections = [
            self._sec_power_input,
            self._sec_voltage_regulator,
            self._sec_mcu,
            self._sec_analog,
            self._sec_interface_protection,
            self._sec_filter,
            self._sec_output,
            self._sec_passive_network,
        ]
        chosen = self.rng.sample(sections, min(n_sections, len(sections)))
        for s in chosen:
            s()
        return self

    def _sec_power_input(self):
        j = self._add('J', 'DC_IN')
        f = self._add('F')
        d = self._add('D', '1N4007')
        c1 = self._add('C', '100uF')
        c2 = self._add('C', '100nF')
        pwr = self._add('PWR_FLAG', '+5V')
        g1 = self._add('GND', 'GND')
        g2 = self._add('GND', 'GND')

        self._net((j, '1'), (f, '1'))
        self._net((f, '2'), (d, 'A'))
        self._net((d, 'K'), (c1, '1'), (c2, '1'), (pwr, '1'))
        self._net((j, '2'), (g1, '1'))
        self._net((c1, '2'), (c2, '2'), (g2, '1'))

    def _sec_mcu(self):
        ic = self._add('IC')
        for _ in range(3):
            self._add('C', '100nF')
            self._add('GND', 'GND')
        y = self._add('Y', '16MHz')
        self._add('C', '22pF')
        self._add('C', '22pF')
        self._add('R', '10k')
        self._add('GND', 'GND')

    def _sec_analog(self):
        u = self._add('U', 'LM358')
        for _ in range(3):
            self._add('R')
        self._add('C', '100nF')

    def _sec_voltage_regulator(self):
        self._add('C_POL', '10uF')
        self._add('C_POL', '100uF')
        self._add('C', '100nF')
        self._add('C', '100nF')
        led = self._add('LED', 'GREEN')
        self._add('R', '1k')
        self._add('R', '10k')
        self._add('GND', 'GND')
        self._add('GND', 'GND')

    def _sec_interface_protection(self):
        for _ in range(2):
            self._add('R', '10k')
        for _ in range(2):
            self._add('ZD', '3.3V')
        self._add('J', 'HEADER_4P')
        self._add('GND', 'GND')

    def _sec_filter(self):
        self._add('R', '100')
        self._add('C', '10uF')
        self._add('C', '100nF')
        self._add('L', '10uH')
        self._add('FB')
        self._add('GND', 'GND')

    def _sec_output(self):
        self._add('J', 'CONN_3P')
        for _ in range(2):
            self._add('R', '220')
        self._add('LED', 'RED')
        self._add('GND', 'GND')

    def _sec_passive_network(self):
        for _ in range(2):
            self._add('R', '47k')
        self._add('R', '10k')
        self._add('C', '100nF')
        self._add('R', '1k')
        self._add('GND', 'GND')

    def get_gt_text(self):
        lines = []
        for refdes, comp_type, value in self.components:
            display = refdes  # Already correct from _add
            lines.append(f"{display}\n{value}")
        return '\n'.join(lines)


# ============================================================
# Layout Engine
# ============================================================

class LayoutEngine:
    def __init__(self, topology, canvas_w=850, canvas_h=620, seed=None):
        self.topo = topology
        self.cw, self.ch = canvas_w, canvas_h
        self.rng = random.Random(seed)
        self.positions = {}
        self.pin_positions = {}  # (refdes, pin) -> (x, y)
        self.wires = []  # [(x1,y1,x2,y2)]
        self.junctions = []

    def place(self):
        comps = self.topo.components
        n = len(comps)
        if n == 0: return

        margin = 60
        avail_w = self.cw - 2*margin
        avail_h = self.ch - 2*margin

        cols = max(3, int(math.sqrt(n * avail_w / avail_h)))
        rows = max(2, (n + cols - 1) // cols)
        cell_w = avail_w / cols
        cell_h = avail_h / rows

        # Group by category for better layout
        cats = defaultdict(list)
        for i, (r, ct, v) in enumerate(comps):
            comp = COMPONENTS.get(ct, COMPONENTS['R'])
            cats[comp.category].append(i)

        all_idx = []
        cat_lists = list(cats.values())
        self.rng.shuffle(cat_lists)
        max_l = max(len(c) for c in cat_lists) if cat_lists else 0
        for j in range(max_l):
            for cl in cat_lists:
                if j < len(cl):
                    all_idx.append(cl[j])

        for grid_i, comp_i in enumerate(all_idx):
            refdes, comp_type, value = comps[comp_i]
            col, row = grid_i % cols, grid_i // cols

            jx = self.rng.uniform(-cell_w*0.12, cell_w*0.12)
            jy = self.rng.uniform(-cell_h*0.12, cell_h*0.12)

            x = margin + col*cell_w + cell_w/2 + jx
            y = margin + row*cell_h + cell_h/2 + jy
            x = max(40, min(self.cw-40, x))
            y = max(40, min(self.ch-40, y))

            self.positions[refdes] = (x, y)

            comp = COMPONENTS.get(comp_type, COMPONENTS['R'])
            for pn, (dx, dy) in comp.pins.items():
                self.pin_positions[(refdes, pn)] = (x+dx, y+dy)

    def route(self):
        for net in self.topo.nets:
            if len(net) < 2: continue
            pts = []
            for ref, pin in net:
                k = (ref, pin)
                if k in self.pin_positions:
                    pts.append(self.pin_positions[k])
                elif ref in self.positions:
                    # Use component center as fallback
                    pts.append(self.positions[ref])

            if len(pts) < 2: continue

            if len(pts) <= 3:
                for i in range(len(pts)-1):
                    self._manhattan(pts[i], pts[i+1])
            else:
                cx = sum(p[0] for p in pts)/len(pts)
                cy = sum(p[1] for p in pts)/len(pts)
                jct = (cx, cy)
                self.junctions.append(jct)
                for p in pts:
                    self._manhattan(p, jct)
                self._add_junction_dot(jct)

    def _manhattan(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        if abs(x1-x2) < 3 or abs(y1-y2) < 3:
            self.wires.append((x1, y1, x2, y2))
        else:
            # Check if a junction already exists at the bend point
            if self.rng.random() < 0.5:
                jx, jy = x2, y1
                self.wires.append((x1, y1, jx, jy))
                self.wires.append((jx, jy, x2, y2))
                self._add_junction_dot((jx, jy))
            else:
                jx, jy = x1, y2
                self.wires.append((x1, y1, jx, jy))
                self.wires.append((jx, jy, x2, y2))
                self._add_junction_dot((jx, jy))

    def _add_junction_dot(self, pt):
        # Check if any wire endpoint is near this point -> junction
        near = False
        for wx1, wy1, wx2, wy2 in self.wires:
            if (abs(wx1-pt[0])<2 and abs(wy1-pt[1])<2) or \
               (abs(wx2-pt[0])<2 and abs(wy2-pt[1])<2):
                near = True
                break
        if near:
            self.junctions.append(pt)


# ============================================================
# KiCad-Like Renderer
# ============================================================

class KiCadRenderer:
    def __init__(self, topo, layout, cw=850, ch=620, dpi=100, seed=None):
        self.topo = topo
        self.layout = layout
        self.cw, self.ch = cw, ch
        self.dpi = dpi
        self.rng = random.Random(seed)

    def render(self, path):
        fig_w, fig_h = self.cw/self.dpi, self.ch/self.dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=self.dpi,
                               facecolor=BG_COLOR)
        ax.set_xlim(0, self.cw)
        ax.set_ylim(0, self.ch)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(BG_COLOR)

        # 1. Grid dots
        self._draw_grid(ax)

        # 2. Border
        self._draw_border(ax)

        # 3. Wires
        for x1, y1, x2, y2 in self.layout.wires:
            ax.plot([x1, x2], [y1, y2], color=WIRE_COLOR, linewidth=WIRE_WIDTH,
                    solid_capstyle='round')

        # 4. Junction dots
        for jx, jy in set(self.layout.junctions):
            ax.plot(jx, jy, 'o', color=JUNCTION_COLOR, markersize=3.5, zorder=10)

        # 5. Components
        for refdes, comp_type, value in self.topo.components:
            if refdes not in self.layout.positions: continue
            x, y = self.layout.positions[refdes]

            comp = COMPONENTS.get(comp_type, COMPONENTS['R'])
            # Random rotation for variety (but limit to keep readability)
            rot = 0
            if comp_type in ('C', 'C_POL', 'D', 'LED', 'ZD', 'R', 'R_US', 'L'):
                if self.rng.random() < 0.15:
                    rot = 90

            comp.draw_func(ax, x, y, scale=1.0)

            # Skip text for power/ground symbols (value is self-explanatory)
            if comp_type in ('PWR_FLAG',):
                ax.text(x, y-18, value, fontsize=TEXT_SIZE_NET, ha='center', va='top',
                        fontfamily='monospace', fontweight='bold', color=TEXT_COLOR)
            elif comp_type == 'GND':
                pass  # Ground doesn't need label
            elif comp_type == 'VCC':
                ax.text(x+14, y+2, value, fontsize=TEXT_SIZE_NET, ha='left', va='center',
                        fontfamily='monospace', color=TEXT_COLOR)
            else:
                # Standard refdes + value labels
                ly = y + comp.height/2 + 8
                ax.text(x, ly, refdes, fontsize=TEXT_SIZE_REFDES, ha='center', va='bottom',
                        fontfamily='monospace', fontweight='bold', color=TEXT_COLOR,
                        bbox=dict(boxstyle='round,pad=0.08', facecolor=BG_COLOR,
                                  edgecolor='none', alpha=0.85))
                vy = y - comp.height/2 - 3
                ax.text(x, vy, value, fontsize=TEXT_SIZE_VALUE, ha='center', va='top',
                        fontfamily='monospace', color=VALUE_COLOR,
                        bbox=dict(boxstyle='round,pad=0.08', facecolor=BG_COLOR,
                                  edgecolor='none', alpha=0.85))

        plt.tight_layout(pad=0)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.08,
                    facecolor=BG_COLOR, edgecolor='none')
        plt.close(fig)

    def _draw_grid(self, ax):
        for gx in range(GRID_SPACING, self.cw, GRID_SPACING):
            for gy in range(GRID_SPACING, self.ch, GRID_SPACING):
                ax.plot(gx, gy, '.', color=GRID_COLOR, markersize=GRID_DOT_SIZE,
                        alpha=0.6)

    def _draw_border(self, ax):
        m = 10
        ax.plot([m, self.cw-m, self.cw-m, m, m],
                [m, m, self.ch-m, self.ch-m, m],
                color=BORDER_COLOR, linewidth=0.8, alpha=0.4)
        # Simple title block corner
        tx, ty = self.cw-m-150, m
        ax.plot([tx, self.cw-m, self.cw-m, tx, tx],
                [ty, ty, ty+30, ty+30, ty],
                color=BORDER_COLOR, linewidth=0.6, alpha=0.3)


# ============================================================
# Generate & batch
# ============================================================

def generate_one(out_path, seed=0, cw=850, ch=620):
    rng = random.Random(seed)
    topo = CircuitTopology(seed=rng.randint(0, 2**30))
    topo.generate(n_sections=rng.randint(3, 6))

    layout = LayoutEngine(topo, canvas_w=cw, canvas_h=ch, seed=rng.randint(0, 2**30))
    layout.place()
    layout.route()

    renderer = KiCadRenderer(topo, layout, cw=cw, ch=ch, dpi=100, seed=rng.randint(0, 2**30))
    renderer.render(out_path)
    return topo.get_gt_text()


def batch_generate(out_dir, n, start=0, canvas_sizes=None):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    jl_path = os.path.join(out_dir, 'train_synthetic.jsonl')

    if canvas_sizes is None:
        canvas_sizes = [(850, 620), (880, 640), (820, 600), (860, 630)]

    t0 = time.time()
    log_every = max(1, n//20)

    with open(jl_path, 'w', encoding='utf-8') as f:
        for i in range(start, start+n):
            seed = i*137 + 42
            cw, ch = canvas_sizes[i % len(canvas_sizes)]
            img_name = f's{i:05d}.png'
            img_path = os.path.join(img_dir, img_name)

            try:
                gt = generate_one(img_path, seed=seed, cw=cw, ch=ch)
            except Exception as e:
                print(f"[GEN] ERR {i}: {e}", flush=True)
                continue

            entry = {
                'images': [img_path.replace('\\', '/')],
                'messages': [
                    {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'OCR:'}]},
                    {'role': 'assistant', 'content': gt}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            done = i-start+1
            if done % log_every == 0:
                et = (time.time()-t0)/60
                rate = done/max(et, 0.01)
                print(f"[GEN] {done}/{n} ({rate:.0f}/min) ETA={((n-done)/max(rate,0.01)):.0f}m", flush=True)

    tt = (time.time()-t0)/60
    print(f"\n[GEN] DONE: {n} in {tt:.0f}m ({n/tt:.0f}/min)", flush=True)
    return jl_path, img_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=5000)
    ap.add_argument('--output_dir', default='g:/mimo_project/circuit_ocr/output/synthetic_circuits')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()

    if args.test:
        args.n_samples = 5; args.output_dir += '_test'

    print(f"[GEN] Generating {args.n_samples} circuits...", flush=True)
    jl, imgs = batch_generate(args.output_dir, args.n_samples)
    print(f"[GEN] JSONL: {jl}", flush=True)
    print(f"[GEN] Images: {imgs}", flush=True)
