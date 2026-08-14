const API_BASE_URL = "http://127.0.0.1:8000/api";

/**
 * Turn a failed response into an Error carrying the server's actual message.
 * FastAPI validation errors arrive as {detail: [{loc, msg}, ...]}, which is far
 * more useful for debugging than a generic "request failed".
 */
async function describeFailure(res, fallback) {
  let detail;
  try {
    const body = await res.json();
    if (Array.isArray(body.detail)) {
      detail = body.detail
        .map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`)
        .join('; ');
    } else if (typeof body.detail === 'string') {
      detail = body.detail;
    }
  } catch {
    // Response wasn't JSON — fall back to the status line.
  }
  return new Error(detail ? `${fallback} (${res.status}) — ${detail}` : `${fallback} (${res.status})`);
}

async function requestJson(url, options, fallback) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (err) {
    // Network-level failure: the API is unreachable rather than returning an error.
    throw new Error(`${fallback} — cannot reach the API at ${API_BASE_URL}. Is the backend running?`);
  }
  if (!res.ok) throw await describeFailure(res, fallback);
  return res.json();
}

const jsonPost = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body)
});

export async function fetchHealth() {
  try {
    return await requestJson(`${API_BASE_URL}/health`, undefined, "Health check failed");
  } catch (err) {
    // Health is the one call that resolves instead of throwing, so the UI can
    // render an "offline" state rather than an error boundary.
    return { status: "offline", error: err.message };
  }
}

export function predictCognitiveLoad(features) {
  return requestJson(`${API_BASE_URL}/predict`, jsonPost(features), "Prediction failed");
}

export function predictRawEvents(events) {
  return requestJson(`${API_BASE_URL}/predict-raw-events`, jsonPost(events),
    "Raw event prediction failed");
}

export function predictPersonalized(features, personalBaseline) {
  return requestJson(`${API_BASE_URL}/predict-personalized`,
    jsonPost({ features, personal_baseline: personalBaseline }),
    "Personalized prediction failed");
}

export function calibrateBaseline(windows) {
  return requestJson(`${API_BASE_URL}/calibrate`, jsonPost({ windows }),
    "Calibration failed");
}

export function fetchModelInsights() {
  return requestJson(`${API_BASE_URL}/model-insights`, undefined,
    "Model insights fetch failed");
}

export function fetchFeatureBaselines() {
  return requestJson(`${API_BASE_URL}/feature-baselines`, undefined,
    "Feature baselines fetch failed");
}

export function simulateSession(durationMinutes = 15, sessionType = "coding_challenge") {
  const query = `?duration_minutes=${durationMinutes}&session_type=${sessionType}`;
  return requestJson(`${API_BASE_URL}/simulate-session${query}`, { method: "POST" },
    "Session simulation failed");
}
