---
title: State Engine Reference
doc_type: spec
team: tracking
year: 2024
---
# State Engine Reference

## State machine
The State Engine folds resolved scan events into one of eleven tracking states: Manifested, Collected, At Origin Depot, In Transit, At Destination Depot, Out For Delivery, Delivery Attempted, Held At Depot, Delivered, Returned To Sender and Lost. Only six of these are customer-visible: Collected, In Transit, Out For Delivery, Delivery Attempted, Held At Depot and Delivered.

## Ordering
Events can arrive out of order. The State Engine orders by device timestamp, except that events carrying the skew_corrected flag are ordered by gateway receive time. A late event never moves a consignment backwards out of a terminal state. Delivered, Returned To Sender and Lost are terminal.

## The Lost state
A consignment moves to Lost automatically when it has had no scan for nine calendar days and is not in a terminal state. The Claims team is notified through the claims webhook. A subsequent scan reopens the consignment and sets it to the state implied by that scan, and the webhook sends a retraction.

## Idempotency
Every transition is keyed by consignment identifier and event identifier. Replaying the event log, for example after restoring PostgreSQL from backup, produces the same state.

## Dependencies
The State Engine reads from the resolved-events topic written by the Resolver and writes state to PostgreSQL. It publishes state changes to the state-changes topic, which the Notifier consumes.
