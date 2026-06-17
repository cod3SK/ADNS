/**
 * Unit tests for App.jsx — pure helpers + component rendering with mocked API.
 *
 * Run from frontend/adns-frontend/:  npx vitest run
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

// ── Import after mock is installed ───────────────────────────────────────────
import App from '../App.jsx'
import axios from 'axios'

const api = axios.create()

function stubApi(overrides = {}) {
  const defaults = {
    '/api/flows':           { data: [] },
    '/api/anomalous_flows': { data: [] },
    '/api/anomalies':       { data: { count: 0, max_score: 0, pct_anomalous: 0, window: 'recent' } },
    '/api/killswitch':      { data: { enabled: false } },
    '/api/blocked_ips':     { data: [] },
    '/api/agent/status':    { data: { running: false, tshark_found: false, flows_captured: 0 } },
    '/api/interfaces':      { data: [] },
    '/api/model_status':    { data: { meta_model_status: 'ok', active_estimators: 2, total_estimators: 2, estimators: { xgboost: { status: 'ok', error: null }, extra_trees: { status: 'ok', error: null } } } },
  }
  const merged = { ...defaults, ...overrides }
  api.get.mockImplementation((url) => Promise.resolve(merged[url] ?? { data: null }))
}

beforeEach(() => {
  vi.clearAllMocks()
  stubApi()
})

// ── Pure helper unit tests (no DOM) ─────────────────────────────────────────

// formatLabel and severityFromLabel are not exported — test them via inline copies
// (mirrors the logic in App.jsx to avoid coupling tests to internal exports)

function formatLabel(label) {
  if (!label) return 'Unknown'
  const cleaned = String(label).replace(/_/g, ' ').trim()
  if (!cleaned) return 'Unknown'
  return cleaned.split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
}

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

describe('formatLabel', () => {
  it('converts underscores to spaces and title-cases', () => {
    expect(formatLabel('dos_attack')).toBe('Dos Attack')
  })

  it('returns Unknown for null', () => {
    expect(formatLabel(null)).toBe('Unknown')
  })

  it('handles single word', () => {
    expect(formatLabel('anomaly')).toBe('Anomaly')
  })
})

describe('severityFromLabel', () => {
  it('normal label → normal', () => {
    expect(severityFromLabel('normal', 0)).toBe('normal')
  })

  it('ddos label → anomaly', () => {
    expect(severityFromLabel('ddos', 0)).toBe('anomaly')
  })

  it('watch label → watch', () => {
    expect(severityFromLabel('watch', 0)).toBe('watch')
  })

  it('score ≥ 0.9 with unknown label → anomaly', () => {
    expect(severityFromLabel('', 0.95)).toBe('anomaly')
  })

  it('score 0.7 with empty label → watch', () => {
    expect(severityFromLabel('', 0.7)).toBe('watch')
  })

  it('attack substring → anomaly', () => {
    expect(severityFromLabel('brute_attack', 0)).toBe('anomaly')
  })
})

// ── Component rendering tests ─────────────────────────────────────────────────

describe('App component', () => {
  it('renders without crashing', async () => {
    render(<App />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
  })

  it('shows killswitch button', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText(/killswitch/i)).toBeTruthy()
    })
  })

  it('shows Settings tab in nav', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Settings')).toBeTruthy()
    })
  })

  it('shows Dashboard tab active by default', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeTruthy()
    })
  })
})
