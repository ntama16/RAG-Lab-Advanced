---
title: Depot Exceptions Dashboard FAQ
doc_type: faq
team: tracking
year: 2025
---
# Depot Exceptions Dashboard FAQ

## What is an unresolved scan?
A scan the Resolver could not match to any consignment after three retries over roughly two and a half hours. The usual cause is a parcel that arrived at the depot before the sender transmitted its manifest, or a damaged label that scanned as the wrong barcode.

## What should depot staff do with one?
Open the scan in the dashboard and use Find Manifest. If the manifest has since arrived, press Re-resolve. If the label is damaged, reprint it from the consignment record and rescan. Unresolved scans older than 72 hours are hidden from the default view.

## Why is a parcel showing as Lost when it is on my shelf?
The State Engine marks a consignment Lost after nine days without a scan. Scan the parcel with any handheld: the consignment reopens, the customer gets a correction, and the claim is retracted automatically.

## Why do I see the same scan twice?
Duplicates are normally removed by the Resolver within a 20 second window. If you see many duplicates at once, the deduplication cache was probably unavailable; the dedupe-sweep job removes them later.

## Who do I contact?
Dashboard problems go to the Tracking team. Scanner and handheld connectivity problems go to Depot IT, not to the Platform team.
