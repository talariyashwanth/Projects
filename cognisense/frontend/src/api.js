const API_BASE_URL = "http://127.0.0.1:8000/api";

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error("API Offline");
    return await res.json();
  } catch (err) {
    return { status: "offline", error: err.message };
  }
}

export async function predictCognitiveLoad(features) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features)
  });
  if (!res.ok) throw new Error("Prediction API failed");
  return await res.json();
}

export async function predictRawEvents(events) {
  const res = await fetch(`${API_BASE_URL}/predict-raw-events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(events)
  });
  if (!res.ok) throw new Error("Raw events prediction failed");
  return await res.json();
}

export async function fetchModelInsights() {
  const res = await fetch(`${API_BASE_URL}/model-insights`);
  if (!res.ok) throw new Error("Model insights fetch failed");
  return await res.json();
}

export async function simulateSession(durationMinutes = 15, sessionType = "coding_challenge") {
  const res = await fetch(`${API_BASE_URL}/simulate-session?duration_minutes=${durationMinutes}&session_type=${sessionType}`, {
    method: "POST"
  });
  if (!res.ok) throw new Error("Session simulation failed");
  return await res.json();
}
