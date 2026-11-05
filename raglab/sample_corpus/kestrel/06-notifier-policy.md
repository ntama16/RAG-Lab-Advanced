---
title: Customer Notification Policy
doc_type: policy
team: messaging
year: 2024
---
# Customer Notification Policy

## When customers are notified
The Notifier sends a message each time a consignment enters a customer-visible state. Customers choose email, SMS or both when they book. Business accounts may instead register a webhook.

## Quiet hours
SMS messages are not sent between 21:00 and 07:30 in the recipient's local time. Messages raised in quiet hours are held and sent at 07:30, collapsed so that the customer receives only the most recent state. Email is not subject to quiet hours.

## Delivery Attempted
When a delivery attempt fails the message includes the reason code from the courier handheld and a link to rebook. Customers have five working days to rebook or collect before the parcel is returned to the sender. A reminder is sent on the third working day.

## Rate limits
No customer receives more than four SMS messages for a single consignment. When the limit is reached further updates go by email only, and if no email address is held they are dropped and counted in the notifier_suppressed metric.

## Retractions
If a consignment leaves the Lost state because a new scan arrives, the Notifier sends a correction to the customer and the Claims team's webhook receives a retraction. Corrections ignore the SMS rate limit.
