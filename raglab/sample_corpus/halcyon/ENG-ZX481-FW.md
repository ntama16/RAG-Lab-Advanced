---
title: "Engineering Note: ZX-481 Firmware 2.3 Release"
doc_key: ENG-ZX481-FW
doc_type: Engineering
date: 2025-02-27
year: 2025
tags: [ZX-481, firmware, battery, release notes]
---
# Engineering Note: ZX-481 Firmware 2.3 Release

## Summary
Firmware 2.3 for the Ridgeline headlamp fixes the battery gauge drift reported in Q4 and adds a low-temperature charge guard that refuses to charge below 0 °C.

## Changes
- Battery state-of-charge estimation recalibrated; reported capacity no longer drops 15% after the first 20 cycles.
- Boost mode thermal throttle raised from 45 °C to 50 °C surface temperature.
- Lock-out now persists across battery removal.

## Known issues
Units on firmware 2.1 may report false low-capacity readings, which caused a spike in warranty claims. Ask customers to update before assessing battery health.
