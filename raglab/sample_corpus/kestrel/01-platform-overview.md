---
title: Kestrel Platform Overview
doc_type: spec
team: platform
year: 2025
---
# Kestrel Platform Overview

## Purpose
Kestrel is the parcel-tracking platform operated by Northlight Logistics. It ingests scan events from depot scanners and courier handhelds, resolves each scan to a consignment, and publishes tracking state to customers. Kestrel replaced the legacy Parcelbase mainframe in March 2024.

## Core services
The platform has four services. The Ingest Gateway accepts scan events over HTTPS and MQTT and writes them to the event log. The Resolver matches each event to a consignment using the barcode and depot code. The State Engine folds events into a tracking state machine with eleven states, from Manifested to Delivered. The Notifier sends customer messages when a consignment enters a customer-visible state.

## Data stores
The event log is an Apache Kafka cluster with seven-day retention. Consignment state lives in PostgreSQL 16, partitioned by manifest month. The Resolver keeps a barcode lookup cache in Redis with a twelve-hour time to live. Historical events older than seven days are archived to object storage as Parquet files by the nightly Archiver job.

## Ownership
The Platform team, led by Mirela Okafor, owns the Ingest Gateway, the Resolver and the Archiver. The Tracking team, led by Tomas Brennan, owns the State Engine. The Customer Messaging team owns the Notifier. On-call rotations follow service ownership.

## Throughput
Kestrel handles a weekday average of 4.2 million scan events, peaking near 310 events per second in the 17:00 to 19:00 depot outbound window. The Ingest Gateway is provisioned for 900 events per second.
