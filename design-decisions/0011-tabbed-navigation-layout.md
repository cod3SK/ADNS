# 0011 — Tabbed left-nav layout for the dashboard

- **Status:** Accepted
- **Phase:** 3 — Desktop packaging and distribution

## Context

The original dashboard was a single scrolling page: a header, a full-width
anomalous-flows chart, then a two-column grid of four more charts on the left and a
fixed sidebar on the right containing metrics, the anomalous-flow list, blocked IPs,
and the capture pipeline controls. As the number of panels grew the layout had two
problems:

1. **Information density vs. discoverability.** Everything was always visible,
   which meant users had to scroll past charts to reach the capture controls, and
   the blocked-IP list competed for vertical space with the anomalous-flow table.
2. **Conceptual mixing.** Visualization (charts), data browsing (flow table), active
   response (block/unblock), and system administration (capture pipeline) were
   presented as peers on the same surface, making the app feel cluttered and hard to
   explain in a demo or interview setting.

## Decision

Replace the scrolling layout with a persistent left nav rail (158 px) and four
named tabs that each own a single conceptual area:

| Tab | Contents |
|---|---|
| **Dashboard** | Signal summary metric cards + four charts (anomalous flows, threat timeline, severity mix donut, anomaly score over recent flows) |
| **Flows** | Filterable recent-flows table with per-row block action |
| **Flows Manager** | Anomalous flow list + blocked IPs with unblock actions |
| **Settings** | Capture pipeline (tshark status, interface selector, start/stop) |

The kill switch button stays in the top header outside the tab system because it is
a global emergency control that must be reachable from any context.

All state remains in the single `App` component; tab switching is a local
`activeTab` useState with conditional rendering. No routing library was introduced.

## Consequences

- Each tab has a clear purpose, which makes the app easier to narrate and easier to
  extend (a new feature goes in the tab it belongs to, not wherever there is space).
- The dashboard tab now has the full viewport width for charts instead of sharing it
  with the sidebar.
- Active-response actions (Flows Manager) are visually separated from passive
  observation (Dashboard), which makes the threat model clearer to a reviewer.
- The capture pipeline is in Settings, which is the conventional location for
  infrastructure controls in desktop and web apps.
- Cost: flows data and anomalous-flows data are fetched continuously in the
  background regardless of which tab is active (the polling intervals run in
  `useEffect` at mount). This keeps all tabs up to date when switched to, but does
  slightly more work than lazy-loading per tab. At the current polling interval
  (2 s flows, 3 s agent status) this is negligible.
- Cost: tab state is ephemeral — refreshing the page returns to Dashboard. If deep
  linking to a specific tab becomes useful, a URL hash or query param could be added
  without changing the component structure.
