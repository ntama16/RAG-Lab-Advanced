---
title: "TrailPulse Sync API — Rate Limits"
doc_key: ENG-API-01
doc_type: Engineering
date: 2025-05-30
year: 2025
tags: [API, TrailPulse, sync, rate limit]
---
# TrailPulse Sync API — Rate Limits

## Limits
The sync API accepts 60 requests per minute per device token and 600 per minute per account. Bulk history uploads use the batch endpoint, which accepts up to 5 MB per request.

## Backoff
Clients receiving HTTP 429 must apply exponential backoff starting at 2 seconds, doubling to a maximum of 64 seconds.

## Ephemeris
Cached GPS ephemeris data is served from the /ephemeris endpoint and is valid for 7 days. The watch refreshes it on every sync to keep time-to-first-fix under 15 seconds.
