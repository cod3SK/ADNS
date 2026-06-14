import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Area,
  AreaChart,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Scatter,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./App.css";

const apiBase = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
// If no base is provided, rely on Vite dev proxy (/api -> http://127.0.0.1:5000)
const api = axios.create({ baseURL: apiBase });

const formatLabel = (label) => {
  if (!label) return "Unknown";
  const cleaned = String(label).replace(/_/g, " ").trim();
  if (!cleaned) return "Unknown";
  return cleaned
    .split(/\s+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const severityFromLabel = (label, score) => {
  const normalized = (label || "").toLowerCase();
  if (normalized === "normal") {
    return "normal";
  }
  if (["scanning", "dos", "ddos", "injection"].includes(normalized)) {
    return "anomaly";
  }
  if (normalized.includes("attack")) {
    return "anomaly";
  }
  if (normalized === "anomaly" || normalized === "high") {
    return "anomaly";
  }
  if (normalized === "watch" || normalized === "medium") {
    return "watch";
  }
  if (normalized === "normal" || normalized === "low") {
    return "normal";
  }
  const s = Number(score) || 0;
  if (s >= 0.9) return "anomaly";
  if (s >= 0.6) return "watch";
  return "normal";
};

export default function App() {
  const [flows, setFlows] = useState([]);
  const [anomalous, setAnomalous] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [srcFilter, setSrcFilter] = useState("");
  const [killSwitch, setKillSwitch] = useState(false);
  const [killBusy, setKillBusy] = useState(false);
  const [blockMessage, setBlockMessage] = useState("");
  const [blockedIps, setBlockedIps] = useState([]);

  const handleUnblock = async (ip) => {
    setBlockMessage("");
    try {
      await api.post("/api/unblock_ip", { ip });
      setBlockMessage(`Unblocked ${ip}`);
      await fetchLatest();
    } catch (err) {
      console.error("unblock failed", err);
      setBlockMessage("Failed to unblock IP");
    }
  };

  const fetchLatest = useCallback(async () => {
    try {
      setError("");
      const [flowsRes, statsRes, anomaliesRes] = await Promise.all([
        api.get("/api/flows"),
        api.get("/api/anomalies"),
        api.get("/api/anomalous_flows"),
      ]);
      let blocked = [];
      try {
        const blockedRes = await api.get("/api/blocked_ips");
        blocked = blockedRes.data || [];
      } catch (err) {
        console.warn("blocked_ips fetch failed", err);
      }
      const fetchedFlows = flowsRes.data || [];
      setFlows(fetchedFlows);
      setStats(statsRes.data || null);
      setAnomalous(anomaliesRes.data || []);
      setBlockedIps(blocked);
    } catch (err) {
      console.error(err);
      setError("Unable to load data from API");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLatest();
    const id = setInterval(fetchLatest, 2000);
    return () => clearInterval(id);
  }, [fetchLatest]);

  useEffect(() => {
    const fetchKillSwitch = async () => {
      try {
        const res = await api.get("/api/killswitch");
        setKillSwitch(Boolean(res.data?.enabled));
      } catch (err) {
        console.error("killswitch fetch failed", err);
      }
    };
    fetchKillSwitch();
  }, []);

  const toggleKillSwitch = async () => {
    setKillBusy(true);
    try {
      const next = !killSwitch;
      await api.post("/api/killswitch", { enabled: next });
      setKillSwitch(next);
    } catch (err) {
      console.error("killswitch toggle failed", err);
      setError("Unable to toggle killswitch");
    } finally {
      setKillBusy(false);
    }
  };

  const sortedFlows = useMemo(() => {
    return [...flows].sort((a, b) => new Date(b.ts) - new Date(a.ts));
  }, [flows]);

  const anomalousFlows = useMemo(() => {
    const source = anomalous.length ? anomalous : sortedFlows;
    return source.filter((flow) => {
      const severity = severityFromLabel(flow.label, flow.score);
      if (severity !== "normal") return true;
      const s = Number(flow.score) || 0;
      return s >= 0.6;
    });
  }, [anomalous, sortedFlows]);

  const visibleFlows = useMemo(() => {
    if (!srcFilter.trim()) {
      return sortedFlows;
    }
    const needle = srcFilter.trim().toLowerCase();
    return sortedFlows.filter((flow) =>
      (flow.src_ip || "").toLowerCase().includes(needle),
    );
  }, [sortedFlows, srcFilter]);

  const chartData = sortedFlows.map((f, i) => ({
    index: i,
    score: f.score,
    severity: severityFromLabel(f.label, f.score),
    label: f.label,
  }));

  const timelineData = useMemo(() => {
    const ordered = [...sortedFlows].reverse();
    const recent = ordered.slice(-30);
    return recent.map((flow) => ({
      tsLabel: new Date(flow.ts).toLocaleTimeString([], {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
      score: Number(flow.score) || 0,
      severity: severityFromLabel(flow.label, flow.score),
    }));
  }, [sortedFlows]);

  const severityCounts = useMemo(() => {
    return sortedFlows.reduce(
      (acc, flow) => {
        const severity = severityFromLabel(flow.label, flow.score);
        acc[severity] += 1;
        return acc;
      },
      { anomaly: 0, watch: 0, normal: 0 },
    );
  }, [sortedFlows]);

  const donutData = [
    { name: "Anomaly", value: severityCounts.anomaly, severity: "anomaly" },
    { name: "Watch", value: severityCounts.watch, severity: "watch" },
    { name: "Normal", value: severityCounts.normal, severity: "normal" },
  ];

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Live security telemetry</p>
          <h1>ADNS Dashboard</h1>
          <p className="app-subtitle">
            Anomaly Detection Network System — live traffic overview
          </p>
        </div>
        <div className="header-actions">
          <div className="kill-switch">
            <button
              type="button"
              className={`simulate-btn${killSwitch ? " is-active" : ""}`}
              onClick={toggleKillSwitch}
              disabled={killBusy}
            >
              {killSwitch ? "Killswitch ON (disable)" : "Killswitch OFF (enable)"}
            </button>
          </div>
        </div>
      </header>

      <div className="panel-row">
        <section className="panel chart-panel anomaly-panel">
          <div className="panel-heading">
            <h3>Anomalous flows</h3>
            <p>Only non-normal flows; use block buttons to respond.</p>
          </div>
          {anomalousFlows.length === 0 ? (
            <p className="empty-state">No anomalous flows yet.</p>
          ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={anomalousFlows.map((f, i) => ({
                      index: i,
                      score: f.score,
                  severity: severityFromLabel(f.label, f.score),
                  label: f.label,
                  src_ip: f.src_ip,
                }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="index"
                      label={{ value: "Flow index", position: "insideBottom", offset: -4 }}
                    />
                    <YAxis domain={[0, 1]} label={{ value: "Score", angle: -90, position: "insideLeft" }} />
                    <Tooltip
                      formatter={(value, _name, { payload }) => [
                        (Number(value) || 0).toFixed(3),
                        `${payload?.src_ip || ""} ${payload?.label || ""}`,
                      ]}
                />
                <ReferenceLine y={0.9} stroke="#b91c1c" strokeDasharray="4 4" />
                <ReferenceLine y={0.6} stroke="#ea580c" strokeDasharray="4 4" />
                <Line
                  type="monotone"
                  dataKey="score"
                  dot={({ payload }) => (
                    <circle
                      r={4}
                      fill={threatColor(payload.severity)}
                      stroke="#ffffff"
                      strokeWidth={1}
                    />
                  )}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </section>
      </div>

      <div className="content-grid">
        <div className="main-column">
          {error && <div className="app-alert">{error}</div>}
          {blockMessage && <div className="app-alert">{blockMessage}</div>}

          <section className="panel timeline-panel">
            <div className="panel-heading">
              <h3>Threat timeline</h3>
              <p>Recent flow scores with severity shading.</p>
            </div>
            {timelineData.length === 0 ? (
              <p className="empty-state">Timeline will appear once flows arrive.</p>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.7} />
                      <stop offset="40%" stopColor="#f97316" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#22c55e" stopOpacity={0.2} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="2 4" vertical={false} />
                  <XAxis
                    dataKey="tsLabel"
                    tick={{ fontSize: 10 }}
                    interval={timelineData.length > 12 ? 2 : 0}
                  />
                  <YAxis domain={[0, 1]} hide />
                  <Tooltip
                    formatter={(value, _, entry) => [
                      (Number(value) || 0).toFixed(3),
                      entry.payload.severity,
                    ]}
                  />
                  <Area
                    type="monotone"
                    dataKey="score"
                    stroke="#ea580c"
                    fill="url(#scoreGradient)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Scatter
                    data={timelineData.filter((d) => d.severity !== "normal")}
                    fill="#b91c1c"
                    shape="circle"
                  />
                </AreaChart>
              </ResponsiveContainer>
          )}
        </section>

          <section className="panel donut-panel">
            <div className="panel-heading">
              <h3>Severity mix</h3>
              <p>Breakdown of recent flows by model decision.</p>
            </div>
            <div className="donut-wrapper">
              <ResponsiveContainer width="55%" height={220}>
                <PieChart>
                  <Pie
                    data={donutData}
                    innerRadius={60}
                    outerRadius={90}
                    dataKey="value"
                    paddingAngle={2}
                  >
                    {donutData.map((entry) => (
                      <Cell key={entry.severity} fill={threatColor(entry.severity)} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value, name) => [`${value} flows`, name]}
                  />
                </PieChart>
              </ResponsiveContainer>
              <ul className="donut-legend">
                {donutData.map((entry) => (
                  <li key={entry.severity}>
                    <span
                      className="dot"
                      style={{ background: threatColor(entry.severity) }}
                    />
                    {entry.name}: {entry.value}
                  </li>
                ))}
              </ul>
            </div>
          </section>

          <section className="panel chart-panel">
            <div className="panel-heading">
              <h3>Anomaly score over recent flows</h3>
            </div>
            {chartData.length === 0 ? (
              <p className="empty-state">No flow data yet.</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="index"
                    label={{ value: "Recent flows", position: "insideBottom", offset: -4 }}
                  />
                  <YAxis domain={[0, 1]} label={{ value: "Anomaly score", angle: -90, position: "insideLeft" }} />
                  <Tooltip
                    formatter={(value, _name, { payload }) => [
                      (Number(value) || 0).toFixed(3),
                      payload?.severity || "score",
                    ]}
                  />
                  <ReferenceLine y={0.9} stroke="#b91c1c" strokeDasharray="4 4" />
                  <ReferenceLine y={0.6} stroke="#ea580c" strokeDasharray="4 4" />
                  <Line
                    type="monotone"
                    dataKey="score"
                    dot={({ payload }) => (
                      <circle
                        r={4}
                        fill={threatColor(payload.severity)}
                        stroke="#ffffff"
                        strokeWidth={1}
                      />
                    )}
                    activeDot={({ payload }) => (
                      <circle
                        r={5}
                        fill={threatColor(payload.severity)}
                        stroke="#111827"
                        strokeWidth={1}
                      />
                    )}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </section>

          <section className="panel table-panel">
            <div className="panel-heading">
              <h3>Recent flows</h3>
            </div>
            {visibleFlows.length === 0 ? (
              <p className="empty-state">No flows yet.</p>
            ) : (
              <>
                <div className="filter-row">
                  <label htmlFor="srcFilter">Filter by source IP</label>
                  <input
                    id="srcFilter"
                    type="text"
                    value={srcFilter}
                    onChange={(e) => setSrcFilter(e.target.value)}
                    placeholder="e.g. 192.168"
                  />
                </div>
                <div className="table-wrapper">
                  <table className="flow-table">
                    <thead>
                      <tr>
                        <Th>Time</Th>
                        <Th>Source IP</Th>
                        <Th>Destination IP</Th>
                        <Th>Proto</Th>
                        <Th>Bytes</Th>
                        <Th className="col-actions">Actions</Th>
                        <Th className="col-score">Score</Th>
                        <Th className="col-severity">Severity</Th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleFlows.map((f, idx) => (
                        <tr key={idx}>
                          <Td>{new Date(f.ts).toLocaleString()}</Td>
                          <Td>{f.src_ip}</Td>
                          <Td>{f.dst_ip}</Td>
                          <Td>{f.proto}</Td>
                          <Td>{f.bytes}</Td>
                          <Td clamp={false} className="col-actions">
                            <button
                              type="button"
                              className="pill-btn"
                              onClick={async () => {
                                setBlockMessage("");
                                try {
                                  await api.post("/api/block_ip", { ip: f.src_ip });
                                  setBlockMessage(`Blocked ${f.src_ip}`);
                                } catch (err) {
                                  console.error("block failed", err);
                                  setBlockMessage("Failed to block IP");
                                }
                              }}
                            >
                              Block IP
                            </button>
                          </Td>
                          <Td clamp={false} className="col-score">
                            <ScoreTag score={f.score} />
                          </Td>
                          <Td clamp={false} className="col-severity">
                            <ThreatBadge label={f.label} score={f.score} />
                          </Td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
      </div>

      <aside className="sidebar">
        <section className="panel metrics-panel">
          <div className="panel-heading">
            <div className="panel-title-group">
              <h3>Signal summary</h3>
              <p>10-minute pulse for quick health.</p>
            </div>
          </div>
          <div className="metrics-grid metrics-vertical">
            <Card
              title="Anomalies (10 min)"
              value={stats?.count ?? (loading ? "…" : "0")}
            />
            <Card
              title="Max anomaly score"
              value={
                stats?.max_score != null
                  ? stats.max_score.toFixed(3)
                  : loading
                  ? "…"
                  : "0.000"
              }
            />
            <Card
              title="% traffic anomalous"
              value={
                stats?.pct_anomalous != null
                  ? `${stats.pct_anomalous}%`
                  : loading
                  ? "…"
                  : "0%"
              }
            />
          </div>
        </section>

        <section className="panel table-panel sidebar-panel">
          <div className="panel-heading">
            <h3>Anomalous flow list</h3>
            <p>Always-visible anomalies with quick actions.</p>
          </div>
          {anomalousFlows.length === 0 ? (
            <p className="empty-state">No anomalous flows yet.</p>
          ) : (
            <div className="table-wrapper sidebar-scroll">
              <table className="flow-table">
                <thead>
                  <tr>
                    <Th>Time</Th>
                    <Th>Source IP</Th>
                    <Th>Destination IP</Th>
                    <Th>Proto</Th>
                    <Th>Bytes</Th>
                    <Th className="col-actions">Actions</Th>
                    <Th className="col-score">Score</Th>
                    <Th className="col-severity">Severity</Th>
                  </tr>
                </thead>
                <tbody>
                  {anomalousFlows.map((f, idx) => (
                    <tr key={`anomaly-${idx}`}>
                      <Td>{new Date(f.ts).toLocaleString()}</Td>
                      <Td>{f.src_ip}</Td>
                      <Td>{f.dst_ip}</Td>
                      <Td>{f.proto}</Td>
                      <Td>{f.bytes}</Td>
                      <Td clamp={false} className="col-actions">
                        <button
                          type="button"
                          className="pill-btn"
                          onClick={async () => {
                            setBlockMessage("");
                            try {
                              await api.post("/api/block_ip", { ip: f.src_ip });
                              setBlockMessage(`Blocked ${f.src_ip}`);
                            } catch (err) {
                              console.error("block failed", err);
                              setBlockMessage("Failed to block IP");
                            }
                          }}
                        >
                          Block IP
                        </button>
                      </Td>
                      <Td clamp={false} className="col-score">
                        <ScoreTag score={f.score} />
                      </Td>
                      <Td clamp={false} className="col-severity">
                        <ThreatBadge label={f.label} score={f.score} />
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

          <section className="panel sidebar-panel">
            <div className="panel-heading">
              <h3>Blocked IPs</h3>
              <p>Active OS-level blocks applied via iptables.</p>
            </div>
            {blockedIps.length === 0 ? (
              <p className="empty-state">No blocked IPs.</p>
            ) : (
              <ul className="blocked-list">
                {blockedIps.map((row) => (
                  <li key={row.ip}>
                    <div>
                      <div className="ip">{row.ip}</div>
                      <div className="ts">
                        {new Date(row.created_at).toLocaleString()}
                      </div>
                    </div>
                    <div className="row-actions">
                      <span className="score-tag score-anomaly" style={{ minWidth: 0 }}>
                        Blocked
                      </span>
                      <button
                        type="button"
                        className="pill-btn"
                        onClick={() => handleUnblock(row.ip)}
                      >
                        Unblock
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}

function Card({ title, value }) {
  return (
    <div className="metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

function Th({ children }) {
  return <th>{children}</th>;
}

function Td({ children, clamp = true }) {
  if (!clamp) {
    return <td>{children}</td>;
  }
  return (
    <td>
      <span className="cell-text">{children}</span>
    </td>
  );
}

function ScoreTag({ score }) {
  const s = Number(score) || 0;
  let bg = "#e8f5e9";
  let color = "#1b5e20";
  if (s > 0.9) {
    bg = "#ffebee";
    color = "#b71c1c";
  } else if (s > 0.6) {
    bg = "#fff3e0";
    color = "#e65100";
  }
  return (
    <span
      className="score-tag"
      style={{
        background: bg,
        color,
      }}
    >
      {s.toFixed(3)}
    </span>
  );
}

function ThreatBadge({ label, score }) {
  const severity = severityFromLabel(label, score);
  const config = severityConfig();
  const { text, color, bg, icon } = config[severity] || config.normal;
  const labelText = formatLabel(label);
  return (
    <span
      className="threat-badge"
      style={{
        color,
        background: bg,
      }}
    >
      <span className="icon">{icon}</span>
      {labelText || text}
    </span>
  );
}

function threatColor(severity) {
  const config = severityConfig();
  return config[severity]?.color ?? config.normal.color;
}

function severityConfig() {
  return {
    anomaly: {
      text: "Anomaly",
      color: "#b91c1c",
      bg: "#fee2e2",
      icon: "⚠️",
    },
    watch: {
      text: "Watch",
      color: "#b45309",
      bg: "#fff7ed",
      icon: "👁️",
    },
    normal: {
      text: "Normal",
      color: "#166534",
      bg: "#e0f2fe",
      icon: "✅",
    },
  };
}
