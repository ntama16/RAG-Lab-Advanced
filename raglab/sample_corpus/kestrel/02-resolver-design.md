---
title: Resolver Service Design
doc_type: spec
team: platform
year: 2025
---
# Resolver Service Design

## What the Resolver does
The Resolver consumes raw scan events from the event log and attaches a consignment identifier to each one. A scan carries a barcode, a depot code and a device timestamp. Roughly 96 percent of scans resolve on the first attempt through the Redis barcode cache.

## Resolution order
The Resolver tries three strategies in order. First it checks the Redis cache. On a miss it queries the consignment table in PostgreSQL by barcode. If that also misses, the scan is an orphan: the parcel was scanned before its manifest arrived. Orphans go to the retry queue and are attempted again after 5, 30 and 120 minutes. After the third failure the scan is written to the unresolved table and surfaces in the Depot Exceptions dashboard.

## Duplicate handling
Couriers often scan the same parcel twice within a few seconds. The Resolver drops a scan when the same barcode, depot code and scan type arrive within a 20 second window. Deduplication keys are held in Redis with a 60 second expiry.

## Clock skew
Handheld clocks drift. When a device timestamp is more than ten minutes ahead of the gateway receive time, the Resolver substitutes the receive time and sets the skew_corrected flag. The State Engine treats corrected events as lower confidence when ordering.

## Failure modes
If Redis is unavailable the Resolver falls back to PostgreSQL for every scan. This is safe but raises database read load roughly twentyfold, and resolution latency rises from 4 milliseconds to around 45 milliseconds at the median. The runbook RB-014 covers this case.
