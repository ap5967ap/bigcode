import { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { MapContainer, Marker, Polyline, TileLayer, Tooltip as LeafletTooltip, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";

const API_BASE = "http://127.0.0.1:8000";
const ROUTE_COLORS = {
  fastest: "#ef4444",
  balanced: "#fbbf24",
  safest: "#10b981",
  agent_route: "#38bdf8"
};

const markerIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

function scoreColor(score) {
  const clamped = Math.max(0, Math.min(100, score));
  const hue = (clamped / 100) * 120;
  return `hsl(${hue} 80% 52%)`;
}

function geometryToLatLngs(geometry) {
  if (!geometry?.coordinates) return [];
  return geometry.coordinates.map(([lon, lat]) => [lat, lon]);
}

function routeToLatLngs(route) {
  if (!route?.edges) return [];
  return route.edges.flatMap((edge) => geometryToLatLngs(edge.geometry));
}

function buildBounds(routeResult, origin, destination) {
  const points = [
    ...routeToLatLngs(routeResult?.fastest),
    ...routeToLatLngs(routeResult?.balanced),
    ...routeToLatLngs(routeResult?.safest),
    ...routeToLatLngs(routeResult?.agent_route)
  ];

  if (!points.length) {
    if (origin) points.push(origin);
    if (destination) points.push(destination);
  }

  if (!points.length) return null;
  return L.latLngBounds(points);
}

function ServiceAreaBox({ bounds }) {
  if (!bounds) return null;
  const rectangleBounds = [
    [bounds.min_lat, bounds.min_lon],
    [bounds.max_lat, bounds.max_lon]
  ];
  return (
    <Polyline
      positions={[
        rectangleBounds[0],
        [bounds.min_lat, bounds.max_lon],
        rectangleBounds[1],
        [bounds.max_lat, bounds.min_lon],
        rectangleBounds[0]
      ]}
      pathOptions={{ color: "#38bdf8", weight: 1.5, opacity: 0.35, dashArray: "6 6" }}
    />
  );
}

function MapViewportController({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (!bounds || !bounds.isValid()) return;
    map.fitBounds(bounds, { padding: [48, 48], maxZoom: 16 });
  }, [bounds, map]);

  return null;
}

function MapClickHandler({ setOrigin, setDestination }) {
  useMapEvents({
    click(event) {
      const point = [event.latlng.lat, event.latlng.lng];
      setOrigin((currentOrigin) => {
        if (!currentOrigin) return point;
        setDestination(point);
        return currentOrigin;
      });
    },
    contextmenu(event) {
      setOrigin([event.latlng.lat, event.latlng.lng]);
      setDestination(null);
    }
  });
  return null;
}

function RouteCard({ label, route, color }) {
  if (!route) return null;
  return (
    <div className="route-card">
      <div className="route-card__title" style={{ color }}>
        {label}
      </div>
      <div className="route-card__metrics">
        <span>{(route.total_time / 60).toFixed(1)} min</span>
        <span>{route.mean_safety.toFixed(1)} safety</span>
      </div>
      {route.archetype ? <div className="route-card__meta">Routing for: {route.archetype} 🛡️</div> : null}
    </div>
  );
}

export default function App() {
  const [origin, setOrigin] = useState([12.9716, 77.5946]);
  const [destination, setDestination] = useState([12.985, 77.61]);
  const [travelMode, setTravelMode] = useState("walking");
  const [hourOfDay, setHourOfDay] = useState(22);
  const [isFemale, setIsFemale] = useState(false);
  const [destinationType, setDestinationType] = useState("residential");
  const [routeResult, setRouteResult] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [visibleRoutes, setVisibleRoutes] = useState({
    fastest: true,
    balanced: true,
    safest: true,
    agent_route: true
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPareto, setShowPareto] = useState(true);
  const [serviceAreaBounds, setServiceAreaBounds] = useState(null);

  useEffect(() => {
    async function loadHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) return;
        const payload = await response.json();
        setServiceAreaBounds(payload.service_area_bounds || null);
      } catch {
        // Leave bounds unset if the API is unavailable during boot.
      }
    }

    loadHealth();
  }, []);

  const mapCenter = useMemo(() => {
    if (serviceAreaBounds) {
      return [
        (serviceAreaBounds.min_lat + serviceAreaBounds.max_lat) / 2,
        (serviceAreaBounds.min_lon + serviceAreaBounds.max_lon) / 2
      ];
    }
    return origin || [12.9716, 77.5946];
  }, [origin, serviceAreaBounds]);

  const mapBounds = useMemo(
    () => buildBounds(routeResult, origin, destination),
    [routeResult, origin, destination]
  );

  async function findRoute() {
    if (!origin || !destination) {
      setError("Place both origin and destination markers.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin,
          destination,
          travel_mode: travelMode,
          hour_of_day: Number(hourOfDay),
          is_female: isFemale,
          destination_type: destinationType
        })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || `API error ${response.status}`);
      }
      const payload = await response.json();
      setRouteResult(payload);
      setServiceAreaBounds(payload.service_area_bounds || serviceAreaBounds);
      setSelectedSegment(null);
    } catch (routeError) {
      setRouteResult(null);
      setError(routeError.message);
    } finally {
      setLoading(false);
    }
  }

  const paretoData = routeResult?.pareto_frontier || [];
  const recommendedPoints = [
    routeResult?.fastest && {
      name: "Fastest",
      eta_minutes: routeResult.fastest.total_time / 60,
      safety_score: routeResult.fastest.mean_safety,
      fill: ROUTE_COLORS.fastest
    },
    routeResult?.balanced && {
      name: "Balanced",
      eta_minutes: routeResult.balanced.total_time / 60,
      safety_score: routeResult.balanced.mean_safety,
      fill: ROUTE_COLORS.balanced
    },
    routeResult?.safest && {
      name: "Safest",
      eta_minutes: routeResult.safest.total_time / 60,
      safety_score: routeResult.safest.mean_safety,
      fill: ROUTE_COLORS.safest
    }
  ].filter(Boolean);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>NightSafe Routes</h1>
          <p>AI-powered night navigation with explainable street safety scoring.</p>
        </div>

        <label className="field">
          <span>Travel mode</span>
          <select value={travelMode} onChange={(event) => setTravelMode(event.target.value)}>
            <option value="walking">Walking</option>
            <option value="cycling">Cycling</option>
            <option value="cab">Cab</option>
          </select>
        </label>

        <label className="field">
          <span>Hour: {hourOfDay}:00</span>
          <input
            type="range"
            min="0"
            max="23"
            value={hourOfDay}
            onChange={(event) => setHourOfDay(event.target.value)}
          />
        </label>

        <label className="field inline-field">
          <span>Female rider</span>
          <input type="checkbox" checked={isFemale} onChange={(event) => setIsFemale(event.target.checked)} />
        </label>

        <label className="field">
          <span>Destination type</span>
          <select value={destinationType} onChange={(event) => setDestinationType(event.target.value)}>
            <option value="residential">Residential</option>
            <option value="commercial">Commercial</option>
            <option value="transit">Transit</option>
          </select>
        </label>

        <button className="primary-btn" onClick={findRoute} disabled={loading}>
          {loading ? "Finding route..." : "Find Route"}
        </button>
        {error ? <div className="error-box">{error}</div> : null}
        {serviceAreaBounds ? (
          <div className="marker-help">
            Service area: {serviceAreaBounds.min_lat.toFixed(3)}-{serviceAreaBounds.max_lat.toFixed(3)} lat,{" "}
            {serviceAreaBounds.min_lon.toFixed(3)}-{serviceAreaBounds.max_lon.toFixed(3)} lon
          </div>
        ) : null}

        <div className="route-toggle-group">
          {Object.entries(visibleRoutes).map(([routeKey, visible]) => (
            <label key={routeKey} className="toggle-pill">
              <input
                type="checkbox"
                checked={visible}
                onChange={(event) =>
                  setVisibleRoutes((current) => ({ ...current, [routeKey]: event.target.checked }))
                }
              />
              <span style={{ color: ROUTE_COLORS[routeKey] }}>{routeKey.replace("_", " ")}</span>
            </label>
          ))}
        </div>

        <RouteCard label="Fastest Route" route={routeResult?.fastest} color={ROUTE_COLORS.fastest} />
        <RouteCard label="Balanced Route" route={routeResult?.balanced} color={ROUTE_COLORS.balanced} />
        <RouteCard label="Safest Route" route={routeResult?.safest} color={ROUTE_COLORS.safest} />
        <RouteCard label="Agent Route" route={routeResult?.agent_route} color={ROUTE_COLORS.agent_route} />

        <div className="marker-help">Left click sets destination after origin. Right click resets origin.</div>
      </aside>

      <main className="map-wrap">
        <MapContainer center={mapCenter} zoom={14} className="route-map" preferCanvas>
          <TileLayer
            attribution='&copy; CARTO'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <MapViewportController bounds={mapBounds} />
          <MapClickHandler setOrigin={setOrigin} setDestination={setDestination} />
          <ServiceAreaBox bounds={serviceAreaBounds} />
          {origin ? <Marker position={origin} icon={markerIcon} /> : null}
          {destination ? <Marker position={destination} icon={markerIcon} /> : null}

          {routeResult
            ? ["fastest", "balanced", "safest", "agent_route"].flatMap((routeKey) =>
                visibleRoutes[routeKey]
                  ? routeResult[routeKey].edges.map((edge) => (
                      <Polyline
                        key={`${routeKey}-${edge.edge_id}`}
                        positions={geometryToLatLngs(edge.geometry)}
                        pathOptions={{
                          color: routeKey === "agent_route" ? ROUTE_COLORS.agent_route : scoreColor(edge.safety_score),
                          weight: routeKey === "agent_route" ? 6 : 5,
                          opacity: 0.9
                        }}
                        eventHandlers={{
                          click: () =>
                            setSelectedSegment({
                              edgeId: edge.edge_id,
                              score: edge.safety_score,
                              topFeatures: edge.top_features,
                              explanation: edge.explanation
                            })
                        }}
                      >
                        <LeafletTooltip sticky>
                          <div className="map-tooltip">
                            <strong>{edge.safety_score.toFixed(1)}</strong>
                            <div>{edge.explanation}</div>
                          </div>
                        </LeafletTooltip>
                      </Polyline>
                    ))
                  : []
              )
            : null}
        </MapContainer>
      </main>

      <section className={`pareto-panel ${showPareto ? "" : "collapsed"}`}>
        <button className="collapse-btn" onClick={() => setShowPareto((value) => !value)}>
          {showPareto ? "Hide Pareto Frontier" : "Show Pareto Frontier"}
        </button>
        {showPareto ? (
          <ResponsiveContainer width="100%" height={180}>
            <ScatterChart margin={{ top: 16, right: 18, bottom: 12, left: 8 }}>
              <CartesianGrid stroke="#1f3a4a" />
              <XAxis dataKey="eta_minutes" name="ETA" unit=" min" stroke="#7dd3fc" />
              <YAxis dataKey="safety_score" name="Safety" stroke="#7dd3fc" domain={[0, 100]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Legend />
              <Scatter name="Pareto variants" data={paretoData} fill="#38bdf8" />
              <Scatter name="Recommended" data={recommendedPoints} fill="#fbbf24" />
            </ScatterChart>
          </ResponsiveContainer>
        ) : null}
      </section>

      <aside className={`segment-drawer ${selectedSegment ? "open" : ""}`}>
        <button className="drawer-close" onClick={() => setSelectedSegment(null)}>
          ×
        </button>
        <h2>Segment explanation</h2>
        {selectedSegment ? (
          <>
            <div className="segment-score">{selectedSegment.score.toFixed(1)} / 100</div>
            <p>{selectedSegment.explanation}</p>
            <div className="feature-bars">
              {selectedSegment.topFeatures.map(([feature, value]) => (
                <div key={`${selectedSegment.edgeId}-${feature}`} className="feature-bar-row">
                  <span>{feature}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${Math.min(100, Math.abs(value) * 8)}%`,
                        background: value >= 0 ? "#14b8a6" : "#f97316"
                      }}
                    />
                  </div>
                  <em>{Number(value).toFixed(2)}</em>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p>Click any route segment to inspect its top safety factors.</p>
        )}
      </aside>
    </div>
  );
}
