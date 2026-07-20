"""Generate printable PDF of circuit annotations for real-camera training data.
20 pages with clear circuit labels, values, and pinouts in various formats.
"""
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

OUTPUT_DIR = r"g:\mimo_project\circuit_ocr\real_photo_templates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A4 at 300 DPI: 2480 x 3508 pixels
W, H = 2480, 3508
MARGIN = 200

# Try to find a good font
FONT_PATHS = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/msgothic.ttf",
]
FONT_BOLD = None
FONT_MONO = None
FONT_NORMAL = None

for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            f = ImageFont.truetype(fp, 32)
            if FONT_NORMAL is None:
                FONT_NORMAL = fp
            if "consol" in fp.lower() or "cour" in fp.lower():
                FONT_MONO = fp
            if "arial" in fp.lower() and "bold" not in fp.lower():
                if FONT_BOLD is None:
                    FONT_BOLD = fp
        except:
            pass

if FONT_NORMAL is None:
    FONT_NORMAL = FONT_PATHS[2]  # fallback to arial
if FONT_MONO is None:
    FONT_MONO = FONT_NORMAL
if FONT_BOLD is None:
    FONT_BOLD = FONT_NORMAL

print(f"Fonts: normal={FONT_NORMAL}, mono={FONT_MONO}, bold={FONT_BOLD}")

PAGES = []

def new_page(page_num):
    """Create a white page with page number."""
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    # Page number
    try:
        fn = ImageFont.truetype(FONT_NORMAL, 48)
    except:
        fn = ImageFont.load_default()
    draw.text((W - MARGIN - 100, H - MARGIN), str(page_num), fill="black", font=fn)
    return img, draw

def draw_title(draw, text, y, size=60):
    try:
        fn = ImageFont.truetype(FONT_BOLD, size)
    except:
        fn = ImageFont.load_default()
    draw.text((MARGIN, y), text, fill="black", font=fn)
    return y + size + 20

def draw_text_lines(draw, lines, y, size=36, spacing=50, color="black"):
    try:
        fn = ImageFont.truetype(FONT_MONO, size)
    except:
        fn = ImageFont.load_default()
    for line in lines:
        if line == "":
            y += spacing // 2
            continue
        draw.text((MARGIN, y), line, fill=color, font=fn)
        y += spacing
    return y

def draw_table_row(draw, cells, y, col_widths, size=36):
    try:
        fn = ImageFont.truetype(FONT_MONO, size)
    except:
        fn = ImageFont.load_default()
    x = MARGIN
    max_h = 0
    for cell, cw in zip(cells, col_widths):
        draw.text((x, y), str(cell), fill="black", font=fn)
        x += cw
        max_h = max(max_h, size + 14)
    return y + max_h

# ============================================================
# PAGE 1: Resistor values
# ============================================================
img, d = new_page(1)
y = draw_title(d, "Resistor Values (阻值标注)", MARGIN)
y += 20
lines = [
    "R1    10kΩ       ±1%",
    "R2    2.2kΩ      ±5%",
    "R3    100Ω       ±1%",
    "R4    1MΩ        ±5%",
    "R5    4.7kΩ      ±1%",
    "R6    47kΩ       ±1%",
    "R7    220Ω       ±5%",
    "R8    330Ω       ±1%",
    "R9    10Ω        ±5%",
    "R10   0Ω         (jumper)",
    "",
    "R11   1.5kΩ      ±1%",
    "R12   6.8kΩ      ±1%",
    "R13   22kΩ       ±5%",
    "R14   100kΩ      ±1%",
    "R15   470Ω       ±5%",
    "R16   3.3kΩ      ±1%",
    "R17   75kΩ       ±1%",
    "R18   150Ω       ±5%",
    "R19   2.7kΩ      ±1%",
    "R20   5.1kΩ      ±1%",
]
draw_text_lines(d, lines, y, size=42, spacing=65)
PAGES.append(img)

# ============================================================
# PAGE 2: Capacitor values
# ============================================================
img, d = new_page(2)
y = draw_title(d, "Capacitor Values (电容标注)", MARGIN)
y += 20
lines = [
    "C1    10μF       16V     Electrolytic",
    "C2    100μF      25V     Electrolytic",
    "C3    100nF      50V     Ceramic",
    "C4    0.1μF      50V     Ceramic",
    "C5    22μF       10V     Tantalum",
    "C6    1μF        25V     Ceramic X7R",
    "C7    4.7μF      16V     MLCC",
    "C8    10nF       100V    Ceramic",
    "C9    47μF       35V     Electrolytic",
    "C10   220μF      25V     Electrolytic",
    "",
    "C11   100pF      50V     NP0/C0G",
    "C12   2.2μF      10V     Ceramic X5R",
    "C13   470μF      16V     Electrolytic",
    "C14   33pF       50V     NP0",
    "C15   1000μF     6.3V    Electrolytic",
    "C16   0.01μF     50V     Ceramic",
    "",
    "Decoupling: 100nF near each power pin",
    "Bulk:       10μF + 0.1μF per rail",
]
draw_text_lines(d, lines, y, size=40, spacing=60)
PAGES.append(img)

# ============================================================
# PAGE 3: IC Pinouts
# ============================================================
img, d = new_page(3)
y = draw_title(d, "IC Pin Assignments (芯片引脚定义)", MARGIN)
y += 20

# ESP32
lines = [
    "U1: ESP32-WROOM-32E",
    "─────────────────────────────────",
    " 1   EN        14  GPIO12",
    " 2   GPIO36    15  GPIO14",
    " 3   GPIO39    16  GPIO27",
    " 4   GPIO34    17  GPIO26",
    " 5   GPIO35    18  GPIO25",
    " 6   GPIO32    19  GPIO33",
    " 7   GPIO33    20  GPIO32",
    " 8   GPIO25    21  GPIO35",
    " 9   GPIO26    22  GPIO34",
    "10   GPIO27    23  GPIO39",
    "11   GPIO14    24  GPIO36",
    "12   GPIO12    25  GND",
    "13   GND       26  VDD 3.3V",
    "─────────────────────────────────",
    "",
]
draw_text_lines(d, lines, y, size=34, spacing=50)
y += (len(lines) + 1) * 50

# STM32
lines2 = [
    "U2: STM32F103C8T6 (LQFP-48)",
    "─────────────────────────────────",
    " 1  VBAT      13  PA3/TIM2_CH4",
    " 2  PC13      14  PA4/SPI1_NSS",
    " 3  PC14-OSC  15  PA5/SPI1_SCK",
    " 4  PC15-OSC  16  PA6/SPI1_MISO",
    " 5  PD0-OSC   17  PA7/SPI1_MOSI",
    " 6  PD1-OSC   18  PB0/TIM3_CH3",
    " 7  NRST      19  PB1/TIM3_CH4",
    " 8  VSSA      20  PB10/I2C2_SCL",
    " 9  VDDA      21  PB11/I2C2_SDA",
    "10  PA0-WKUP  22  VSS_2",
    "11  PA1       23  VDD_2",
    "12  PA2       24  BOOT0",
    "─────────────────────────────────",
]
draw_text_lines(d, lines2, y, size=32, spacing=48)
PAGES.append(img)

# ============================================================
# PAGE 4: Power Supply Labels
# ============================================================
img, d = new_page(4)
y = draw_title(d, "Power Supply & Voltage Labels (电源标注)", MARGIN)
y += 20
lines = [
    "Net Label          Voltage     Current    Description",
    "────────────────────────────────────────────────────────",
    "VCC                5.0V        500mA      Main supply",
    "VDD                3.3V        300mA      Digital core",
    "VDDA               3.3V        50mA       Analog supply",
    "VREF               2.5V        10mA       ADC reference",
    "VBUS               5.0V        1000mA     USB bus power",
    "VBAT               3.7V        200mA      Battery input",
    "VSYS               3.7-5.5V    2000mA     System input",
    "VIN                7-12V       2000mA     External input",
    "VOUT               3.3V/5V     1000mA     Regulator out",
    "",
    "+5V                5.0V        2000mA     Positive rail",
    "+3.3V              3.3V        1500mA     Logic supply",
    "+12V               12.0V       3000mA     Motor/relay",
    "+24V               24.0V       5000mA     Industrial",
    "-5V                -5.0V       100mA      Op-amp neg",
    "-12V               -12.0V      300mA      RS-232/comms",
    "",
    "GND                Ground      —          Digital ground",
    "AGND               Analog GND  —          Analog ground",
    "PGND               Power GND   —          Power ground",
    "EARTH              Chassis     —          Safety earth",
]
draw_text_lines(d, lines, y, size=34, spacing=52)
PAGES.append(img)

# ============================================================
# PAGE 5: Connector Pinouts
# ============================================================
img, d = new_page(5)
y = draw_title(d, "Connector Pinouts (连接器引脚)", MARGIN)
y += 20
lines = [
    "J1: USB-C Receptacle (USB 2.0)",
    "─────────────────────────────────",
    "A1    GND          B1    GND",
    "A2    SSTXp1       B2    SSRXp1",
    "A3    SSTXn1       B3    SSRXn1",
    "A4    VBUS         B4    VBUS",
    "A5    CC1          B5    CC2",
    "A6    D+           B6    D+",
    "A7    D-           B7    D-",
    "A8    SBU1         B8    SBU2",
    "A9    VBUS         B9    VBUS",
    "A10   SSRXn2       B10   SSTXn2",
    "A11   SSRXp2       B11   SSTXp2",
    "A12   GND          B12   GND",
    "",
    "J2: 2.54mm Pin Header (1x8)",
    "─────────────────────────────────",
    "1     VCC 5V",
    "2     GND",
    "3     SDA (I2C Data)",
    "4     SCL (I2C Clock)",
    "5     GPIO4 / PWM",
    "6     GPIO5 / ADC_IN",
    "7     UART TX",
    "8     UART RX",
]
draw_text_lines(d, lines, y, size=32, spacing=45)
PAGES.append(img)

# ============================================================
# PAGE 6-7: BOM Table
# ============================================================
img, d = new_page(6)
y = draw_title(d, "Bill of Materials — Part 1 (元件清单)", MARGIN)
y += 20
headers = ["Ref", "Value", "Package", "Qty", "Note"]
col_w = [200, 300, 350, 150, 500]
try:
    fn = ImageFont.truetype(FONT_BOLD, 38)
except:
    fn = ImageFont.load_default()
x = MARGIN
for h, cw in zip(headers, col_w):
    d.text((x, y), h, fill="black", font=fn)
    x += cw
y += 60

items = [
    ["R1,R2,R3", "10kΩ ±1%", "0805 SMD", "3", "Pull-up resistors"],
    ["R4,R5", "2.2kΩ ±5%", "0805 SMD", "2", "LED current limit"],
    ["R6", "0.1Ω 1W", "2512 SMD", "1", "Current sense shunt"],
    ["C1,C2,C3", "100nF 50V", "0603 X7R", "3", "Decoupling caps"],
    ["C4,C5", "10µF 16V", "0805 X5R", "2", "Bulk capacitance"],
    ["C6", "22µF 25V", "TH 5mm", "1", "Input filter"],
    ["C7,C8", "22pF 50V", "0603 NP0", "2", "Crystal load caps"],
    ["U1", "ESP32-WROOM", "Module", "1", "MCU module"],
    ["U2", "AMS1117-3.3", "SOT-223", "1", "3.3V LDO regulator"],
    ["U3", "CH340C", "SOP-16", "1", "USB-UART bridge"],
    ["U4", "INA219", "SOT-23-8", "1", "I2C current sensor"],
    ["D1,D2", "1N4148", "SOD-123", "2", "Signal diodes"],
    ["D3", "SS34", "SMA", "1", "Schottky 3A/40V"],
    ["LED1", "Green", "0805 SMD", "1", "Power indicator"],
    ["LED2", "Blue", "0805 SMD", "1", "Status LED"],
    ["Q1", "2N7002", "SOT-23", "1", "N-channel MOSFET"],
    ["Q2", "SI2301", "SOT-23", "1", "P-channel MOSFET"],
    ["L1", "10µH 1A", "CD54", "1", "Buck inductor"],
    ["X1", "32.768kHz", "3.2x1.5mm", "1", "RTC crystal"],
    ["Y1", "40MHz", "5x3.2mm", "1", "Main oscillator"],
]
for row in items:
    draw_table_row(d, row, y, col_w, size=30)
    y += 52
PAGES.append(img)

# PAGE 7 - BOM part 2
img, d = new_page(7)
y = draw_title(d, "Bill of Materials — Part 2 (元件清单续)", MARGIN)
y += 20
x = MARGIN
for h, cw in zip(headers, col_w):
    try:
        fn = ImageFont.truetype(FONT_BOLD, 38)
    except:
        fn = ImageFont.load_default()
    d.text((x, y), h, fill="black", font=fn)
    x += cw
y += 60

items2 = [
    ["J1", "USB-C 16P", "SMD", "1", "USB connector"],
    ["J2", "2.54mm 1x8P", "TH", "1", "GPIO header"],
    ["J3", "JST-XH 2P", "TH", "1", "Battery connector"],
    ["J4", "SMA-KE", "Edge", "1", "Antenna connector"],
    ["SW1", "Tactile 6mm", "TH", "1", "Reset button"],
    ["SW2", "Tactile 6mm", "TH", "1", "Boot/User button"],
    ["F1", "500mA PTC", "1206 SMD", "1", "Resettable fuse"],
    ["RV1", "TVS 5V", "SOD-323", "1", "ESD protection"],
    ["T1", "1:1 10BaseT", "SMD-16", "1", "Ethernet transformer"],
    ["P1", "2.54mm 2x3P", "TH", "1", "SWD debug header"],
    ["P2", "2.54mm 2x4P", "TH", "1", "SPI expansion"],
    ["HS1", "Heatsink 14mm", "Al", "1", "LDO heatsink"],
    ["TP1,TP2", "Test Point", "SMD", "2", "GND reference"],
    ["JP1", "2.54mm 1x3P", "TH", "1", "Power select jumper"],
    ["BZ1", "Piezo 5V", "TH 12mm", "1", "Buzzer"],
    ["CN1", "FFC 10P 0.5mm", "SMD", "1", "LCD flex cable"],
]
for row in items2:
    draw_table_row(d, row, y, col_w, size=30)
    y += 52
PAGES.append(img)

# ============================================================
# PAGE 8: Net Labels
# ============================================================
img, d = new_page(8)
y = draw_title(d, "Net Labels & Signal Names (网络标号)", MARGIN)
y += 20
lines = [
    "Analog Signals:",
    "──────────────────",
    "AIN0, AIN1, AIN2, AIN3          ADC input channels",
    "VREF+                            ADC positive reference",
    "VREF-                            ADC negative reference",
    "TEMP_SENSOR                      NTC temperature probe",
    "CURRENT_SENSE                    INA219 shunt voltage",
    "",
    "Digital I/O:",
    "──────────────────",
    "GPIO0 ~ GPIO39                   ESP32 GPIO pins",
    "UART0_TX, UART0_RX               Serial console (CH340)",
    "UART1_TX, UART1_RX               RS-485 communication",
    "I2C0_SCL, I2C0_SDA               Sensor bus (100kHz)",
    "SPI_MOSI, SPI_MISO, SPI_SCK      SPI Flash/LCD bus",
    "SPI_CS0, SPI_CS1                 Chip select lines",
    "PWM1_OUT, PWM2_OUT               LED dimming output",
    "ENCODER_A, ENCODER_B             Rotary encoder inputs",
    "INT_ACCEL                        Accelerometer interrupt",
    "",
    "Power Nets:",
    "──────────────────",
    "VBUS, VSYS, VBAT, VDD_3V3, VDD_1V8",
    "VDD_CORE, VDD_IO, VDD_ANA, VDD_RF",
    "GND, PGND, AGND, DGND, GND_PLANE",
]
draw_text_lines(d, lines, y, size=34, spacing=48)
PAGES.append(img)

# ============================================================
# PAGE 9: Diode & Transistor
# ============================================================
img, d = new_page(9)
y = draw_title(d, "Diodes & Transistors (二极管与三极管)", MARGIN)
y += 20
lines = [
    "D1    1N4148          Signal diode, 100V/200mA",
    "D2    1N4007          Rectifier, 1000V/1A",
    "D3    SS34            Schottky, 40V/3A, SMA",
    "D4    BZX84C3V3       Zener 3.3V, 300mW, SOT-23",
    "D5    BZX84C5V1       Zener 5.1V, 300mW, SOT-23",
    "D6    1N5819          Schottky, 40V/1A, DO-41",
    "D7    MBR0520L        Schottky, 20V/500mA, SOD-123",
    "D8    LED_RED         Red LED, 2.0Vf, 20mA, 0805",
    "D9    LED_GREEN       Green LED, 2.2Vf, 20mA, 0805",
    "D10   LED_BLUE        Blue LED, 3.3Vf, 20mA, 0805",
    "",
    "Q1    2N2222A         NPN BJT, 40V/800mA, TO-92",
    "Q2    2N3904          NPN BJT, 40V/200mA, SOT-23",
    "Q3    2N7002          N-MOSFET, 60V/300mA, SOT-23",
    "Q4    AO3400A         N-MOSFET, 30V/5.7A, SOT-23",
    "Q5    SI2301          P-MOSFET, -20V/-2.3A, SOT-23",
    "Q6    BSS138          N-MOSFET level shifter",
    "Q7    BC547B          NPN BJT, 45V/100mA, TO-92",
    "Q8    BC557B          PNP BJT, -45V/-100mA, TO-92",
    "Q9    IRFZ44N         N-MOSFET, 55V/49A, TO-220",
    "Q10   IRF9540N        P-MOSFET, -100V/-23A, TO-220",
]
draw_text_lines(d, lines, y, size=30, spacing=45)
PAGES.append(img)

# ============================================================
# PAGE 10: IC Part Numbers
# ============================================================
img, d = new_page(10)
y = draw_title(d, "IC Part Numbers & Markings (芯片型号)", MARGIN)
y += 20
lines = [
    "U1     ESP32-WROOM-32E-N4            Espressif MCU",
    "U2     STM32F103C8T6                  ARM Cortex-M3",
    "U3     ATmega328P-AU                  AVR 8-bit MCU",
    "U4     CH340C                         USB-UART bridge",
    "U5     CP2102N-A02-GQFN28             USB-UART bridge",
    "U6     AMS1117-3.3                    LDO 3.3V/1A",
    "U7     MP1584EN                       Buck converter",
    "U8     XL1509-3.3                     Buck 3.3V/2A",
    "U9     TPS54331                       Buck 3A TI",
    "U10    MAX232ESE                      RS-232 driver",
    "U11    MAX485ESA                      RS-485 transceiver",
    "U12    SN65HVD230                     CAN transceiver",
    "U13    INA219BIDGSR                   I2C current sensor",
    "U14    BME280                         Temp/Hum/Press",
    "U15    MPU6050                        IMU 6-axis",
    "U16    W25Q128JVSIQ                   SPI Flash 128Mb",
    "U17    PCA9685PW                      PWM driver 16ch",
    "U18    TCA9548APWR                    I2C mux 8ch",
    "U19    MCP23017-E/SP                  I2C GPIO expander",
    "U20    ADS1115IDGSR                   I2C ADC 16-bit",
]
draw_text_lines(d, lines, y, size=28, spacing=44)
PAGES.append(img)

# ============================================================
# PAGE 11: Generic Component Labels
# ============================================================
img, d = new_page(11)
y = draw_title(d, "Component Reference Designators (元件标号)", MARGIN)
y += 20
lines = [
    "R1  R2  R3  R4  R5  R6  R7  R8  R9  R10",
    "R11 R12 R13 R14 R15 R16 R17 R18 R19 R20",
    "R21 R22 R23 R24 R25 R26 R27 R28 R29 R30",
    "",
    "C1  C2  C3  C4  C5  C6  C7  C8  C9  C10",
    "C11 C12 C13 C14 C15 C16 C17 C18 C19 C20",
    "",
    "U1  U2  U3  U4  U5  U6  U7  U8  U9  U10",
    "U11 U12 U13 U14 U15 U16 U17 U18 U19 U20",
    "",
    "J1  J2  J3  J4  J5  J6  J7  J8",
    "D1  D2  D3  D4  D5  D6  D7  D8  D9  D10",
    "L1  L2  L3  L4  L5  L6",
    "Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8",
    "Y1  Y2  X1  X2",
    "LED1 LED2 LED3 LED4 LED5",
    "SW1 SW2 SW3 SW4",
    "TP1 TP2 TP3 TP4 TP5 TP6",
    "F1  F2  RV1 RV2",
    "P1  P2  P3  P4",
    "JP1 JP2 JP3",
    "CN1 CN2 CN3",
    "BZ1  HS1",
]
draw_text_lines(d, lines, y, size=40, spacing=56)
PAGES.append(img)

# ============================================================
# PAGE 12: Schematic-style text blocks
# ============================================================
img, d = new_page(12)
y = draw_title(d, "Schematic Text Blocks (原理图文字块)", MARGIN)
y += 20
lines = [
    "POWER INPUT AND REGULATOR",
    "+5V     +5V     +5V     +3.3V",
    "U2:  NCP1117-ADJ",
    "  1 ADJ    2 VO    3 VI",
    "C1:  10μF     C2:  100nF     C3:  10μF",
    "",
    "────────────────────────────────────",
    "",
    "Buck Converter for VSYS 5V",
    "D3:  Green LED",
    "L1:  SWPA4030S3R3MT  (3.3μH)",
    "  1  IN    2  OUT",
    "+5V regulated output",
    "",
    "────────────────────────────────────",
    "",
    "Microcontroller Section",
    "U1:  pi_pico (Raspberry Pi Pico)",
    "  1 GP0           8  GND",
    "  2 GP1           9  GP6",
    "  3 GND          10  GP7",
    "  4 GP2          11  GP8",
    "  5 GP3          12  GP9",
    "  6 GP4          13  GND",
    "  7 GP5          14  GP10",
    "Y1:  12MHz Crystal",
]
draw_text_lines(d, lines, y, size=34, spacing=48)
PAGES.append(img)

# ============================================================
# PAGE 13: More schematic blocks
# ============================================================
img, d = new_page(13)
y = draw_title(d, "Schematic Blocks (cont.)", MARGIN)
y += 20
lines = [
    "USB Interface",
    "J1:  USB_B_Micro",
    "  1 VBUS    2 D-    3 D+    4 ID    5 GND",
    "D1:  D_Schottky (SS14)",
    "F1:  500mA PTC Fuse",
    "",
    "────────────────────────────────────",
    "",
    "Display Connector",
    "J3:  Conn_01x07_Pin (SPI Display)",
    "  1 GND          2 VCC 3.3V",
    "  3 SCL (SCK)    4 SDA (MOSI)",
    "  5 RES (RESET)  6 DC  (A0/Data-Cmd)",
    "  7 CS  (Chip Sel)",
    "",
    "────────────────────────────────────",
    "",
    "Audio Output",
    "U3:  LM386N-1  (Audio Amplifier)",
    "  1 GAIN    2 -IN    3 +IN",
    "  4 GND     5 VOUT   6 VS",
    "  7 BYPASS  8 GAIN",
    "R5:  10kΩ  (Volume pot)",
    "SPK1:  8Ω 0.5W Speaker",
    "",
    "────────────────────────────────────",
    "",
    "RTC Section",
    "U10:  DS3231M  (I2C RTC)",
    "Y2:   32.768kHz Crystal",
    "BAT1: CR2032 (Backup Battery)",
]
draw_text_lines(d, lines, y, size=32, spacing=46)
PAGES.append(img)

# ============================================================
# PAGE 14: Component Values Mix
# ============================================================
img, d = new_page(14)
y = draw_title(d, "Common Component Values (常用参数值)", MARGIN)
y += 20
lines = [
    "Resistors (1% E96 series):",
    "10Ω  22Ω  47Ω  100Ω  220Ω  470Ω  1kΩ",
    "2.2kΩ  4.7kΩ  10kΩ  22kΩ  47kΩ  100kΩ",
    "1MΩ  2.2MΩ  4.7MΩ  10MΩ",
    "",
    "Capacitors:",
    "10pF  22pF  33pF  100pF  1nF  10nF  100nF",
    "0.1μF  0.22μF  0.47μF  1μF  2.2μF  4.7μF",
    "10μF  22μF  47μF  100μF  220μF  470μF",
    "1000μF  2200μF  4700μF",
    "",
    "Inductors:",
    "1μH  2.2μH  3.3μH  4.7μH  6.8μH  10μH",
    "22μH  33μH  47μH  68μH  100μH  220μH",
    "470μH  1mH  2.2mH  10mH",
    "",
    "Voltages:",
    "1.2V  1.8V  2.5V  3.0V  3.3V  5.0V",
    "9V  12V  15V  24V  48V",
    "-5V  -12V  -15V",
    "",
    "Tolerances:  ±0.1%  ±0.5%  ±1%  ±2%  ±5%  ±10%  ±20%",
    "Package:  0201  0402  0603  0805  1206  1210  2512",
    "Package:  SOT-23  SOT-223  SOIC-8  TSSOP-16  QFN-32",
]
draw_text_lines(d, lines, y, size=32, spacing=48)
PAGES.append(img)

# ============================================================
# PAGE 15: Mixed format - table + text
# ============================================================
img, d = new_page(15)
y = draw_title(d, "Test Point Assignments & Silkscreen (测试点与丝印)", MARGIN)
y += 20
lines = [
    "Silkscreen Labels on PCB (PCB丝印文字):",
    "",
    "PWR     RST     BOOT    TX      RX",
    "SDA     SCL     MOSI    MISO    SCK",
    "3.3V    5V      GND     VIN     VOUT",
    "A0      A1      A2      D0      D1",
    "CH1     CH2     TRIG    ECHO",
    "ANT     NFC     IR      BT      WIFI",
    "PROG    DEBUG   RESET   USER",
    "",
    "────────────────────────────────────",
    "",
    "Test Point Assignments:",
    "TP1     GND             Test hook (black)",
    "TP2     +3.3V           Test pad 1mm",
    "TP3     +5V             Test pad 1mm",
    "TP4     UART0_TX        Test via 0.5mm",
    "TP5     UART0_RX        Test via 0.5mm",
    "TP6     I2C0_SCL        Test pad 1mm",
    "TP7     I2C0_SDA        Test pad 1mm",
    "TP8     PWM1_OUT        Test via 0.5mm",
    "TP9     RESET_N         Test pad 1mm",
    "TP10    VBUS_SENSE      Test via 0.5mm",
]
draw_text_lines(d, lines, y, size=34, spacing=48)
PAGES.append(img)

# ============================================================
# PAGE 16-20: Large-format single labels
# ============================================================
for pg, labels in enumerate([
    # Page 16
    ["10kΩ", "2.2kΩ", "100Ω", "4.7kΩ", "47kΩ", "1MΩ",
     "100nF", "10μF", "22μF", "0.1μF", "220μF", "1μF"],
    # Page 17
    ["VCC 5V", "VDD 3.3V", "GND", "VIN 12V", "VOUT 3.3V",
     "VBUS 5V", "VBAT 3.7V", "+12V", "-12V", "+24V", "AGND"],
    # Page 18
    ["U1", "U2", "U3", "R1", "R2", "C1", "C2", "J1",
     "D1", "L1", "Q1", "Y1", "LED1", "SW1", "TP1", "F1"],
    # Page 19
    ["ESP32", "STM32", "CH340", "AMS1117", "INA219",
     "MPU6050", "MAX232", "CP2102", "W25Q128", "ATmega328P",
     "BME280", "MCP23017", "PCA9685", "ADS1115", "SN65HVD230"],
    # Page 20
    ["SDA", "SCL", "MOSI", "MISO", "SCK", "TX", "RX",
     "PWM", "ADC", "GPIO", "I2C", "SPI", "UART", "INT",
     "RST", "EN", "CS", "CLK", "D+", "D-"],
], start=16):
    img, d = new_page(pg)
    label = ["Large Labels (大字标签)", "──────────────────", ""]
    for item in labels:
        label.append(item)
    draw_text_lines(d, label, MARGIN, size=56, spacing=90)
    PAGES.append(img)

# ============================================================
# Save as PDF
# ============================================================
pdf_path = os.path.join(OUTPUT_DIR, "circuit_annotation_templates.pdf")
PAGES[0].save(
    pdf_path,
    "PDF",
    save_all=True,
    append_images=PAGES[1:],
    resolution=300.0,  # force 300 DPI
)
print(f"\nPDF saved: {pdf_path}")
print(f"Pages: {len(PAGES)}")
print(f"File size: {os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB")
print("\nDone! Print this PDF and photograph each page.")
print("Name your photos: 1.jpg, 2.jpg, 3.jpg ... 20.jpg")
