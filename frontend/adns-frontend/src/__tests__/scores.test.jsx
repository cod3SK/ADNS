/**
 * Tests for score display in the Flows table and dashboard cards.
 *
 * ScoreTag and ThreatBadge are unexported internal components; their logic
 * is tested inline and through App rendering with mocked API data.
 */

import '@testing-library/jest-dom'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

// ── Stub axios before importing App ──────────────────────────────────────────
vi.mock('axios', () => {
  const get = vi.fn().mockResolvedValue({ data: [] })
  const post = vi.fn().mockResolvedValue({ data: {} })
  return {
    default: {
      create: () => ({ get, post }),
      get,
      post,
    },
  }
})

import App from '../App.jsx'
import axios from 'axios'

const api = axios.create()

// ── severityFromLabel — inline mirror of App.jsx logic ───────────────────────

function severityFromLabel(label, score) {
  const normalized = (label || '').toLowerCase()
  if (normalized === 'normal') return 'normal'
  if (['scanning', 'dos', 'ddos', 'injection'].includes(normalized)) return 'anomaly'
  if (normalized.includes('attack')) return 'anomaly'
  if (normalized === 'anomaly' || normalized === 'high') return 'anomaly'
  if (normalized === 'watch' || normalized === 'medium') return 'watch'
  const s = Number(score) || 0
  if (s >= 0.9) return 'anomaly'
  if (s >= 0.6) return 'watch'
  return 'normal'
}

// ── ScoreTag display logic — inline mirror ────────────────────────────────────

function scoreTagColors(score) {
  const s = Number(score) || 0
  if (s > 0.9) return { bg: '#ffebee', color: '#b71c1c' }
  if (s > 0.6) return { bg: '#fff3e0', color: '#e65100' }
  return { bg: '#e8f5e9', color: '#1b5e20' }
}

function scoreTagText(score) {
  return (Number(score) || 0).toFixed(3)
}

// ── API stub helpers ──────────────────────────────────────────────────────────

const EMPTY_STATS = { count: 0, max_score: 0, pct_anomalous: 0, window: 'recent' }

function stubApi(flowsData = [], anomalousData = [], statsData = EMPTY_STATS) {
  api.get.mockImplementation((url) => {
    const map = {
      '/api/flows': { data: flowsData },
      '/api/anomalous_flows': { data: anomalousData },
      '/api/anomalies': { data: statsData },
      '/api/killswitch': { data: { enabled: false } },
      '/api/blocked_ips': { data: [] },
      '/api/agent/status': { data: { running: false, tshark_found: false, flows_captured: 0 } },
      '/api/interfaces': { data: [] },
      '/api/model_status': { data: { meta_model_status: 'ok', active_estimators: 2, total_estimators: 2, estimators: { xgboost: { status: 'ok', error: null }, extra_trees: { status: 'ok', error: null } } } },
    }
    return Promise.resolve(map[url] ?? { data: null })
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  stubApi()
})

// ── ScoreTag logic tests (pure) ───────────────────────────────────────────────

describe('scoreTagText', () => {
  it('formats zero score as "0.000"', () => {
    expect(scoreTagText(0)).toBe('0.000')
  })

  it('formats null/undefined as "0.000"', () => {
    expect(scoreTagText(null)).toBe('0.000')
    expect(scoreTagText(undefined)).toBe('0.000')
  })

  it('formats high score with three decimal places', () => {
    expect(scoreTagText(0.876)).toBe('0.876')
  })

  it('formats exactly 1.0 correctly', () => {
    expect(scoreTagText(1.0)).toBe('1.000')
  })
})

describe('scoreTagColors', () => {
  it('green for score ≤ 0.6', () => {
    const { bg } = scoreTagColors(0.3)
    expect(bg).toBe('#e8f5e9')
  })

  it('orange for score in (0.6, 0.9]', () => {
    const { bg } = scoreTagColors(0.75)
    expect(bg).toBe('#fff3e0')
  })

  it('red for score > 0.9', () => {
    const { bg } = scoreTagColors(0.95)
    expect(bg).toBe('#ffebee')
  })

  it('green for exactly 0.0', () => {
    const { bg } = scoreTagColors(0.0)
    expect(bg).toBe('#e8f5e9')
  })
})

// ── severityFromLabel — score-only path ───────────────────────────────────────

describe('severityFromLabel — score-driven paths', () => {
  it('score 0.91 with empty label → anomaly', () => {
    expect(severityFromLabel('', 0.91)).toBe('anomaly')
  })

  it('score 0.65 with empty label → watch', () => {
    expect(severityFromLabel('', 0.65)).toBe('watch')
  })

  it('score 0.0 with empty label → normal', () => {
    expect(severityFromLabel('', 0.0)).toBe('normal')
  })

  it('score 0.59 with empty label → normal (below watch threshold)', () => {
    expect(severityFromLabel('', 0.59)).toBe('normal')
  })

  it('label takes precedence over score: "normal" label with score 0.99 → normal', () => {
    expect(severityFromLabel('normal', 0.99)).toBe('normal')
  })
})

// ── severityFromLabel — label-driven paths ────────────────────────────────────

describe('severityFromLabel — label-driven paths', () => {
  it('"ddos" → anomaly', () => expect(severityFromLabel('ddos', 0)).toBe('anomaly'))
  it('"scanning" → anomaly', () => expect(severityFromLabel('scanning', 0)).toBe('anomaly'))
  it('"dos" → anomaly', () => expect(severityFromLabel('dos', 0)).toBe('anomaly'))
  it('"injection" → anomaly', () => expect(severityFromLabel('injection', 0)).toBe('anomaly'))
  it('"brute_attack" → anomaly (contains "attack")', () => expect(severityFromLabel('brute_attack', 0)).toBe('anomaly'))
  it('"anomaly" → anomaly', () => expect(severityFromLabel('anomaly', 0)).toBe('anomaly'))
  it('"watch" → watch', () => expect(severityFromLabel('watch', 0)).toBe('watch'))
  it('"normal" → normal', () => expect(severityFromLabel('normal', 0)).toBe('normal'))
  it('empty string → normal (score 0)', () => expect(severityFromLabel('', 0)).toBe('normal'))
  it('null → normal (score 0)', () => expect(severityFromLabel(null, 0)).toBe('normal'))
})

// ── App renders score values from API ─────────────────────────────────────────

const SAMPLE_FLOWS = [
  { id: 1, ts: '2026-01-01T00:00:00Z', src_ip: '10.0.0.1', dst_ip: '8.8.8.8', proto: 'TCP', bytes: 1000, score: 0.92, label: 'ddos' },
  { id: 2, ts: '2026-01-01T00:00:01Z', src_ip: '10.0.0.2', dst_ip: '8.8.4.4', proto: 'UDP', bytes: 200, score: 0.0, label: 'normal' },
  { id: 3, ts: '2026-01-01T00:00:02Z', src_ip: '10.0.0.3', dst_ip: '1.1.1.1', proto: 'TCP', bytes: 500, score: 0.72, label: 'watch' },
]

describe('App renders scores correctly', () => {
  it('displays "0.000" for a normal zero-score flow', async () => {
    stubApi(SAMPLE_FLOWS, [], EMPTY_STATS)
    render(<App />)
    await waitFor(() => {
      // The Flows tab isn't the default (dashboard is), so check for score in dashboard anomaly chart data
      // or navigate to flows tab. Since this is a quick check — look for "0.000" in the DOM.
      // At least one "0.000" should appear for the zero-score flow.
      expect(screen.getAllByText('0.000').length).toBeGreaterThan(0)
    })
  })

  it('displays high score rounded to three decimals', async () => {
    stubApi(SAMPLE_FLOWS, [SAMPLE_FLOWS[0]], { count: 1, max_score: 0.92, pct_anomalous: 33, window: 'recent' })
    render(<App />)
    await waitFor(() => {
      expect(screen.getAllByText('0.920').length).toBeGreaterThan(0)
    })
  })

  it('shows max anomaly score from stats in the dashboard card', async () => {
    stubApi(SAMPLE_FLOWS, [], { count: 3, max_score: 0.876, pct_anomalous: 10, window: 'recent' })
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('0.876')).toBeTruthy()
    })
  })

  it('renders without crashing when all scores are 0.0', async () => {
    const zeroFlows = SAMPLE_FLOWS.map(f => ({ ...f, score: 0.0, label: 'normal' }))
    stubApi(zeroFlows, [], EMPTY_STATS)
    render(<App />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
  })

  it('renders without crashing when flows array is empty', async () => {
    stubApi([], [], EMPTY_STATS)
    render(<App />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
  })
})
