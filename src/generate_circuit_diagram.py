"""
Script: src.generate_circuit_diagram
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Generate an authoritative, publication-quality circuit schematic diagram for Phase 7.3
         with explicitly separated 3.3V, 5V/VIN, and GND nets, Option D 15k:10k voltage dividers,
         labeled resistor values, and clear ESP32 pinouts.
"""

import os
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_schematic():
    # Setup high-resolution figure
    fig, ax_raw = plt.subplots(figsize=(16, 11), dpi=300)
    ax: Any = ax_raw
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A") # Slate dark background

    # Title & Metadata Banner
    ax.text(8.0, 10.4, "CloudGuard-Water: Verified ESP32 IoT Circuit Schematic (Phase 7.3)",
            color="#F8FAFC", fontsize=17, fontweight="bold", ha="center")
    ax.text(8.0, 10.0, "Isolated 3.3V / 5.0V Rails | ADC1-Only Allocation | Option D 15k:10k Dividers (M=2.5x, <=2.0V at ADC)",
            color="#94A3B8", fontsize=11, ha="center")

    # Color Palette for Nets
    C_5V = "#F97316"      # Orange for 5.0V / VIN
    C_3V3 = "#EF4444"     # Red for 3.3V
    C_GND = "#64748B"     # Slate Gray for Common Ground
    C_PH = "#A855F7"      # Purple for pH signal
    C_TDS = "#10B981"     # Emerald Green for TDS signal
    C_TURB = "#06B6D4"    # Cyan for Turbidity signal
    C_TEMP = "#F59E0B"    # Amber for DS18B20 1-Wire DATA
    C_BOX = "#1E293B"     # Box background
    C_BORDER = "#475569"  # Box border

    # =========================================================================
    # 1. Power Distribution Rails (Top & Bottom Buses)
    # =========================================================================
    # 5V Rail
    ax.plot([1.0, 15.0], [9.3, 9.3], color=C_5V, lw=4, zorder=2)
    ax.text(0.9, 9.3, "5.0V / VIN RAIL (USB Power)", color=C_5V, fontsize=10, fontweight="bold", ha="right", va="center")

    # 3.3V Rail
    ax.plot([1.0, 15.0], [8.8, 8.8], color=C_3V3, lw=4, zorder=2)
    ax.text(0.9, 8.8, "3.3V RAIL (ESP32 Onboard LDO)", color=C_3V3, fontsize=10, fontweight="bold", ha="right", va="center")

    # Common GND Rail
    ax.plot([1.0, 15.0], [0.8, 0.8], color=C_GND, lw=4, zorder=2)
    ax.text(0.9, 0.8, "COMMON GROUND (GND)", color="#94A3B8", fontsize=10, fontweight="bold", ha="right", va="center")

    # =========================================================================
    # 2. ESP32 Microcontroller Block (Center-Right)
    # =========================================================================
    esp_box = patches.FancyBboxPatch((11.0, 1.6), 4.6, 6.7, boxstyle="round,pad=0.2",
                                    facecolor="#1E293B", edgecolor="#38BDF8", lw=2.5, zorder=3)
    ax.add_patch(esp_box)
    ax.text(13.3, 8.0, "ESP32-WROOM-32", color="#38BDF8", fontsize=13, fontweight="bold", ha="center")
    ax.text(13.3, 7.65, "(30-Pin / 38-Pin DevKit)", color="#94A3B8", fontsize=9, ha="center")

    # ESP32 Pin Callouts & Terminals
    pins = [
        ("VIN", 7.1, C_5V, "5V USB In"),
        ("3V3", 6.4, C_3V3, "3.3V LDO Out"),
        ("GPIO 34", 5.5, C_PH, "ADC1_CH6 (pH In, <=2.0V)"),
        ("GPIO 35", 4.6, C_TDS, "ADC1_CH7 (TDS In, <=2.3V)"),
        ("GPIO 32", 3.7, C_TURB, "ADC1_CH4 (Turb In, <=1.8V)"),
        ("GPIO 4", 2.8, C_TEMP, "1-Wire (Temp In/Out)"),
        ("GND", 1.9, C_GND, "System Ground"),
    ]

    for name, y_pos, color, desc in pins:
        # Terminal dot
        ax.plot(11.0, y_pos, marker="o", color=color, markersize=7, zorder=5)
        ax.text(11.3, y_pos, name, color=color, fontsize=10, fontweight="bold", va="center")
        ax.text(15.3, y_pos, desc, color="#94A3B8", fontsize=8, ha="right", va="center")

    # Connect ESP32 VIN to 5V rail
    ax.plot([11.0, 10.4, 10.4], [7.1, 7.1, 9.3], color=C_5V, lw=2.5, zorder=4)
    ax.plot(10.4, 9.3, marker="o", color=C_5V, markersize=7)

    # Connect ESP32 3V3 to 3.3V rail
    ax.plot([11.0, 10.7, 10.7], [6.4, 6.4, 8.8], color=C_3V3, lw=2.5, zorder=4)
    ax.plot(10.7, 8.8, marker="o", color=C_3V3, markersize=7)

    # Connect ESP32 GND to GND rail
    ax.plot([11.0, 10.4, 10.4], [1.9, 1.9, 0.8], color=C_GND, lw=2.5, zorder=4)
    ax.plot(10.4, 0.8, marker="o", color=C_GND, markersize=7)

    # =========================================================================
    # 3. Sensor 1: Analog pH Sensor Module (Top Left)
    # =========================================================================
    ph_box = patches.FancyBboxPatch((1.0, 6.5), 3.2, 1.7, boxstyle="round,pad=0.15",
                                    facecolor=C_BOX, edgecolor=C_PH, lw=2, zorder=3)
    ax.add_patch(ph_box)
    ax.text(2.6, 7.85, "pH Sensor Module", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.6, 7.55, "(PH-4502C / SEN0161)", color="#C084FC", fontsize=9, ha="center")

    # pH Terminals inside box
    ax.text(3.9, 7.8, "VCC", color=C_5V, fontsize=8, fontweight="bold", ha="right")
    ax.text(3.9, 7.1, "AOUT", color=C_PH, fontsize=8, fontweight="bold", ha="right")
    ax.text(3.9, 6.6, "GND", color="#94A3B8", fontsize=8, fontweight="bold", ha="right")

    # pH Power & GND connections
    ax.plot([1.8, 1.8], [8.2, 9.3], color=C_5V, lw=2)  # VCC to 5V rail
    ax.plot(1.8, 9.3, marker="o", color=C_5V, markersize=6)
    ax.plot([1.4, 1.4], [6.5, 0.8], color=C_GND, lw=2) # GND to GND rail
    ax.plot(1.4, 0.8, marker="o", color=C_GND, markersize=6)

    # pH Voltage Divider Option D (R1 = 15k, R2 = 10k, M = 2.5x)
    # AOUT wire from (4.2, 7.1) -> R1 -> Node -> GPIO 34
    ax.plot([4.2, 5.0], [7.1, 7.1], color=C_PH, lw=2)
    # Resistor R1 box
    r1_box = patches.Rectangle((5.0, 6.9), 1.0, 0.4, facecolor="#334155", edgecolor=C_PH, lw=1.5, zorder=4)
    ax.add_patch(r1_box)
    ax.text(5.5, 7.1, "R1: 15k", color="#F8FAFC", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5)

    # Divider Node after R1
    ax.plot([6.0, 6.8], [7.1, 7.1], color=C_PH, lw=2)
    ax.plot(6.8, 7.1, marker="o", color=C_PH, markersize=7)
    ax.text(6.8, 7.45, "pH ADC Node (<=2.0V, M=2.5x)", color="#C084FC", fontsize=8, fontweight="bold", ha="center")

    # Resistor R2 to GND (10k)
    ax.plot([6.8, 6.8], [7.1, 6.3], color=C_GND, lw=2)
    r2_box = patches.Rectangle((6.5, 5.7), 0.6, 0.6, facecolor="#334155", edgecolor=C_GND, lw=1.5, zorder=4)
    ax.add_patch(r2_box)
    ax.text(6.8, 6.0, "R2\n10k", color="#F8FAFC", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=5)
    ax.plot([6.8, 6.8], [5.7, 0.8], color=C_GND, lw=2)
    ax.plot(6.8, 0.8, marker="o", color=C_GND, markersize=6)

    # Signal path from Node to ESP32 GPIO 34
    ax.plot([6.8, 8.5, 8.5, 11.0], [7.1, 7.1, 5.5, 5.5], color=C_PH, lw=2.5, zorder=4)

    # =========================================================================
    # 4. Sensor 2: Analog TDS Sensor Module (Middle-High Left)
    # =========================================================================
    tds_box = patches.FancyBboxPatch((1.0, 4.6), 3.2, 1.5, boxstyle="round,pad=0.15",
                                     facecolor=C_BOX, edgecolor=C_TDS, lw=2, zorder=3)
    ax.add_patch(tds_box)
    ax.text(2.6, 5.75, "TDS Sensor Module", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.6, 5.45, "(DFRobot SEN0244)", color="#6EE7B7", fontsize=9, ha="center")

    # TDS Terminals
    ax.text(3.9, 5.7, "VCC", color=C_3V3, fontsize=8, fontweight="bold", ha="right")
    ax.text(3.9, 5.1, "AOUT", color=C_TDS, fontsize=8, fontweight="bold", ha="right")
    ax.text(3.9, 4.6, "GND", color="#94A3B8", fontsize=8, fontweight="bold", ha="right")

    # TDS Power: STRICTLY 3.3V Rail (ensures output <= 2.3V max)
    ax.plot([2.2, 2.2], [6.1, 8.8], color=C_3V3, lw=2)
    ax.plot(2.2, 8.8, marker="o", color=C_3V3, markersize=6)

    # TDS GND
    ax.plot([1.8, 1.8], [4.6, 0.8], color=C_GND, lw=2)
    ax.plot(1.8, 0.8, marker="o", color=C_GND, markersize=6)

    # TDS Signal (Direct 3.3V connection to GPIO 35)
    ax.plot([4.2, 7.5, 7.5, 11.0], [5.1, 5.1, 4.6, 4.6], color=C_TDS, lw=2.5, zorder=4)
    ax.text(5.5, 5.25, "Direct (0 - 2.3V, safe on 3.3V VCC)", color="#6EE7B7", fontsize=8, fontweight="bold")

    # =========================================================================
    # 5. Sensor 3: Optical Turbidity Sensor Module (Middle-Low Left)
    # =========================================================================
    turb_box = patches.FancyBboxPatch((1.0, 2.7), 3.2, 1.5, boxstyle="round,pad=0.15",
                                      facecolor=C_BOX, edgecolor=C_TURB, lw=2, zorder=3)
    ax.add_patch(turb_box)
    ax.text(2.6, 3.85, "Turbidity Sensor", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.6, 3.55, "(SEN0189 / TS-300B)", color="#67E8F9", fontsize=9, ha="center")

    # Turbidity Terminals
    ax.text(3.9, 3.8, "VCC", color=C_5V, fontsize=8, fontweight="bold", ha="right")
    ax.text(3.9, 3.2, "AOUT", color=C_TURB, fontsize=8, fontweight="bold", ha="right")
    ax.text(3.9, 2.7, "GND", color="#94A3B8", fontsize=8, fontweight="bold", ha="right")

    # Turbidity Power: 5V Rail (required for IR LED emitter)
    ax.plot([2.6, 2.6], [4.2, 9.3], color=C_5V, lw=2)
    ax.plot(2.6, 9.3, marker="o", color=C_5V, markersize=6)

    # Turbidity GND
    ax.plot([2.2, 2.2], [2.7, 0.8], color=C_GND, lw=2)
    ax.plot(2.2, 0.8, marker="o", color=C_GND, markersize=6)

    # Turbidity Voltage Divider Option D (R3 = 15k, R4 = 10k, M = 2.5x)
    ax.plot([4.2, 5.0], [3.2, 3.2], color=C_TURB, lw=2)
    r3_box = patches.Rectangle((5.0, 3.0), 1.0, 0.4, facecolor="#334155", edgecolor=C_TURB, lw=1.5, zorder=4)
    ax.add_patch(r3_box)
    ax.text(5.5, 3.2, "R3: 15k", color="#F8FAFC", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5)

    # Divider Node after R3
    ax.plot([6.0, 6.8], [3.2, 3.2], color=C_TURB, lw=2)
    ax.plot(6.8, 3.2, marker="o", color=C_TURB, markersize=7)
    ax.text(6.8, 3.55, "Turb ADC Node (<=1.8V clear, M=2.5x)", color="#67E8F9", fontsize=8, fontweight="bold", ha="center")

    # Resistor R4 to GND (10k)
    ax.plot([6.8, 6.8], [3.2, 2.4], color=C_GND, lw=2)
    r4_box = patches.Rectangle((6.5, 1.8), 0.6, 0.6, facecolor="#334155", edgecolor=C_GND, lw=1.5, zorder=4)
    ax.add_patch(r4_box)
    ax.text(6.8, 2.1, "R4\n10k", color="#F8FAFC", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=5)
    ax.plot([6.8, 6.8], [1.8, 0.8], color=C_GND, lw=2)
    ax.plot(6.8, 0.8, marker="o", color=C_GND, markersize=6)

    # Signal path from Node to ESP32 GPIO 32
    ax.plot([6.8, 8.5, 8.5, 11.0], [3.2, 3.2, 3.7, 3.7], color=C_TURB, lw=2.5, zorder=4)

    # =========================================================================
    # 6. Sensor 4: DS18B20 Temperature Probe (Bottom Left)
    # =========================================================================
    temp_box = patches.FancyBboxPatch((1.0, 1.1), 3.2, 1.2, boxstyle="round,pad=0.15",
                                      facecolor=C_BOX, edgecolor=C_TEMP, lw=2, zorder=3)
    ax.add_patch(temp_box)
    ax.text(2.6, 1.95, "DS18B20 Temp Probe", color="#F8FAFC", fontsize=10.5, fontweight="bold", ha="center")
    ax.text(2.6, 1.65, "(Dallas 1-Wire Stainless)", color="#FCD34D", fontsize=8.5, ha="center")

    # Terminals
    ax.text(3.9, 2.0, "VDD (Red)", color=C_3V3, fontsize=7.5, fontweight="bold", ha="right")
    ax.text(3.9, 1.5, "DATA (Yel)", color=C_TEMP, fontsize=7.5, fontweight="bold", ha="right")
    ax.text(3.9, 1.15, "GND (Blk)", color="#94A3B8", fontsize=7.5, fontweight="bold", ha="right")

    # DS18B20 Wires
    # Red wire (VDD) -> 3.3V Rail
    ax.plot([3.0, 3.0], [2.3, 8.8], color=C_3V3, lw=2)
    ax.plot(3.0, 8.8, marker="o", color=C_3V3, markersize=6)

    # Black wire (GND) -> GND Rail
    ax.plot([2.6, 2.6], [1.1, 0.8], color=C_GND, lw=2)
    ax.plot(2.6, 0.8, marker="o", color=C_GND, markersize=6)

    # Yellow wire (DATA) -> GPIO 4
    ax.plot([4.2, 7.8], [1.5, 1.5], color=C_TEMP, lw=2)
    ax.plot(7.8, 1.5, marker="o", color=C_TEMP, markersize=7)

    # 4.7k Pull-up Resistor R5 strictly to 3.3V Rail!
    ax.plot([7.8, 7.8], [1.5, 2.1], color=C_3V3, lw=2)
    r5_box = patches.Rectangle((7.5, 2.1), 0.6, 0.6, facecolor="#334155", edgecolor=C_3V3, lw=1.5, zorder=4)
    ax.add_patch(r5_box)
    ax.text(7.8, 2.4, "R5\n4.7k", color="#F8FAFC", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=5)
    ax.plot([7.8, 7.8], [2.7, 8.8], color=C_3V3, lw=2)
    ax.plot(7.8, 8.8, marker="o", color=C_3V3, markersize=6)
    ax.text(8.3, 2.4, "Pull-up to 3.3V Rail\n(Protects GPIO 4)", color="#FCA5A5", fontsize=8, fontweight="bold", va="center")

    # Connect DATA line to GPIO 4
    ax.plot([7.8, 9.5, 9.5, 11.0], [1.5, 1.5, 2.8, 2.8], color=C_TEMP, lw=2.5, zorder=4)

    # =========================================================================
    # 7. Legend & Architectural Safety Callout Box (Bottom Center)
    # =========================================================================
    leg_box = patches.FancyBboxPatch((9.2, 0.15), 6.4, 1.3, boxstyle="round,pad=0.1",
                                     facecolor="#1E293B", edgecolor="#475569", lw=1.5, zorder=3)
    ax.add_patch(leg_box)
    ax.text(12.4, 1.25, "ELECTRICAL VERIFICATION & SAFETY RULES (PHASE 7.3)", color="#F8FAFC", fontsize=9, fontweight="bold", ha="center")
    ax.text(9.4, 0.95, "• 3.3V Net: Exclusively from ESP32 3V3 pin. NEVER connects to non-power pins.", color="#EF4444", fontsize=7.8)
    ax.text(9.4, 0.70, "• Option D Dividers: 15k:10k (M=2.5x). Max 5.0V stepped down to <=2.0V (within 2.45V linear region).", color="#F97316", fontsize=7.8)
    ax.text(9.4, 0.45, "• ADC Restriction: Strictly ADC1 channels (32, 34, 35) used to preserve Wi-Fi operation.", color="#38BDF8", fontsize=7.8)

    # Save output
    output_path = os.path.join("reports", "figures", "phase7_iot_wiring_diagram.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Schematic successfully saved to {output_path}")

if __name__ == "__main__":
    generate_schematic()
