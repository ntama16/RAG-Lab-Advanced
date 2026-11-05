---
title: "Incident Review INC-2291: Redis failover during outbound window"
doc_type: incident
team: platform
year: 2025
---
# Incident Review INC-2291

## Summary
On 12 February 2025 at 17:22 a Redis primary failed over during the depot outbound window. For 41 minutes the Resolver fell back to PostgreSQL. Customer tracking pages were delayed by up to six minutes and about 38,000 duplicate scan entries were shown to customers.

## Timeline
At 17:22 the node hosting Redis primary 2 was drained by an unannounced kernel patch rollout. Sentinel promoted a replica in nine seconds, but the Resolver's connection pool kept the dead address until pods were restarted at 17:51. At 17:34 the PostgreSQL primary reached 92 percent CPU and State Engine write latency tripled. At 18:03 latency returned to normal.

## Root causes
The Resolver's Redis client did not re-resolve the primary after failover because topology refresh was disabled. Separately, fallback reads went to the PostgreSQL primary rather than the replicas, so Resolver reads competed with State Engine writes.

## What went well
No events were lost. The event log retained everything and the State Engine's idempotent transitions meant replay was safe.

## Actions
Mirela Okafor's team enabled topology refresh in the Redis client and added the read-target switch used in RB-014. Tomas Brennan's team added a write-latency alert to the State Engine. The Infrastructure team now announces kernel patch rollouts 48 hours ahead and excludes 16:00 to 20:00.
