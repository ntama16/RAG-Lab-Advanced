---
title: Data Retention and the Archiver
doc_type: policy
team: platform
year: 2023
---
# Data Retention and the Archiver

## Retention periods
Raw scan events stay in the Kafka event log for seven days. Consignment state is kept in PostgreSQL for 18 months after the consignment reaches a terminal state, then moved to the archive. Archived Parquet files are kept for seven years to meet the carrier liability rules Northlight Logistics operates under.

## The Archiver job
The Archiver runs nightly at 01:30. It reads the previous day's events from the event log, writes one Parquet file per depot per day, and verifies row counts against the Ingest Gateway's daily counters. A mismatch larger than 0.01 percent fails the job and pages the Platform on-call.

## Personal data
Recipient names, phone numbers and addresses are removed from archived events after 24 months. The Archiver's redaction pass rewrites affected files in place and records the rewrite in the redaction ledger. Barcodes and depot codes are retained.

## Restores
Archived events can be replayed into a scratch event log with `kestrel-ctl archive replay`. Replays are used for claims investigations and for rebuilding state after data loss, and rely on the State Engine being idempotent.
