---
title: "Support Playbook: TrailPulse Sync Problems"
doc_key: SUP-TPW-01
doc_type: Support
date: 2025-06-25
year: 2025
tags: [TrailPulse, sync, troubleshooting]
---
# Support Playbook: TrailPulse Sync Problems

## Watch will not pair
Confirm Bluetooth is on and the app has location permission. Hold the crown for five seconds to show the pairing code. If no code appears, place the watch on the charger for 10 seconds to wake it from deep-sleep mode.

## Slow GPS fix
A first fix can take two minutes. If later fixes also take longer than 15 seconds, the ephemeris cache is stale: sync the watch with the app, which refreshes cached ephemeris data.

## Sync errors after many retries
The API rate limit is 60 requests per minute per device. Ask the customer to wait one minute; the app backs off automatically.
