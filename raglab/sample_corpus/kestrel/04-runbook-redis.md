---
title: "RB-014: Resolver running without Redis"
doc_type: runbook
team: platform
year: 2025
---
# RB-014: Resolver running without Redis

## Symptoms
The alert resolver_cache_unavailable fires. Resolver median latency climbs above 40 milliseconds. PostgreSQL read IOPS on the primary rise sharply. Duplicate scans start to appear in tracking history because the deduplication keys are also held in Redis.

## Immediate actions
1. Confirm Redis is the cause: run `kestrel-ctl cache status`. A healthy cluster reports three primaries.
2. Shift Resolver reads to the PostgreSQL read replicas: `kestrel-ctl resolver set read-target replica`. This protects the primary, which the State Engine writes to.
3. If the outbound window (17:00 to 19:00) is within the hour, scale the Resolver from 6 to 10 pods so the retry queue does not back up.
4. Page the Platform on-call secondary if Redis has not recovered in 15 minutes.

## Recovery
When Redis returns, the cache is cold. Do not switch the read target back immediately. Run `kestrel-ctl cache warm --hours 12` first, which replays the last twelve hours of manifests into the cache and takes about eight minutes. Then set the read target back to primary-cache.

## Clean-up
Duplicates created during the outage are removed by the dedupe-sweep job. Trigger it manually with `kestrel-ctl jobs run dedupe-sweep --since <outage start>`.

## History
This runbook was written after the incident of 12 February 2025 (INC-2291).
