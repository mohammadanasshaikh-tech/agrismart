const API_BASE = "http://localhost:8000";

let selectedFile = null;
let lastPrediction = null;
let diseaseChart = null;
let healthChart = null;
let trendChart = null;
let cropChart = null;
let activityChart = null;
let fieldMap = null;
let currentCoords = null;

// safe helper: never throws if element or handler target is missing
function on(el, event, handler) {
  if (!el) { console.warn("Skipped listener -- element not found for event:", event); return; }
  el.addEventListener(event, handler);
}
function byId(id) {
  const el = document.getElementById(id);
  if (!el) console.warn("Missing element with id:", id);
  return el;
}

// ===================== SIDEBAR ROUTING =====================
const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");

navItems.forEach((item) => {
  on(item, "click", () => {
    const target = item.dataset.view;
    navItems.forEach((n) => n.classList.remove("active"));
    item.classList.add("active");
    views.forEach((v) => v.classList.remove("active"));
    const targetView = byId(`view-${target}`);
    if (targetView) targetView.classList.add("active");

    if (target === "dashboard") loadDashboard();
    if (target === "map" && fieldMap) {
      setTimeout(() => fieldMap.invalidateSize(), 50);
    }
  });
});

// ===================== HEALTH CHECK =====================
const statusDot = byId("statusDot");
const statusText = byId("statusText");

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("bad status");
    if (statusDot) { statusDot.classList.add("online"); statusDot.classList.remove("offline"); }
    if (statusText) statusText.textContent = "Backend connected";
  } catch (e) {
    if (statusDot) { statusDot.classList.add("offline"); statusDot.classList.remove("online"); }
    if (statusText) statusText.textContent = "Backend offline";
  }
}
checkBackend();
setInterval(checkBackend, 15000);

// ===================== DASHBOARD =====================
async function loadDashboard() {
  try {
    const res = await fetch(`${API_BASE}/api/dashboard-stats`);
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    renderDashboard(await res.json());
  } catch (e) {
    console.error("Dashboard load failed:", e.message);
  }

  try {
    const res2 = await fetch(`${API_BASE}/api/history?limit=100`);
    if (res2.ok) {
      const data = await res2.json();
      const predictions = data.predictions || [];
      renderRecentStrip(predictions.slice(0, 8));
      renderConfidenceTrend(predictions);
      renderCropBreakdown(predictions);
      renderActivityChart(predictions);
      renderThisWeekStat(predictions);
    }
  } catch (e) {
    console.error("Recent scans load failed:", e.message);
  }
}

function renderDashboard(data) {
  const set = (id, val) => { const el = byId(id); if (el) el.textContent = val; };

  set("statTotal", data.total_predictions);
  set("statHealthy", data.healthy_count);
  set("statDiseased", data.diseased_count);
  set("statConfidence", data.total_predictions > 0 ? `${Math.round(data.average_confidence * 100)}%` : "--");
  set("statSustainability", data.average_sustainability_score !== null ? `${data.average_sustainability_score}/100` : "--");

  const chartEmptyHint = byId("chartEmptyHint");
  const diseaseCanvas = byId("diseaseChart");

  Chart.defaults.color = "#9a9fab";
  Chart.defaults.borderColor = "#2e3440";

  if (diseaseCanvas) {
    if (data.top_diseases.length === 0) {
      if (chartEmptyHint) chartEmptyHint.hidden = false;
      diseaseCanvas.hidden = true;
    } else {
      if (chartEmptyHint) chartEmptyHint.hidden = true;
      diseaseCanvas.hidden = false;
      if (diseaseChart) diseaseChart.destroy();
      diseaseChart = new Chart(diseaseCanvas, {
        type: "bar",
        data: {
          labels: data.top_diseases.map((d) => d.display_name),
          datasets: [{
            label: "Detections",
            data: data.top_diseases.map((d) => d.count),
            backgroundColor: "#d97757",
            borderRadius: 4,
          }],
        },
        options: {
          indexAxis: "y",
          plugins: { legend: { display: false } },
          scales: { x: { beginAtZero: true, ticks: { stepSize: 1 } } },
        },
      });
    }
  }

  const healthCanvas = byId("healthChart");
  if (healthCanvas) {
    if (healthChart) healthChart.destroy();
    healthChart = new Chart(healthCanvas, {
      type: "doughnut",
      data: {
        labels: ["Healthy", "Diseased"],
        datasets: [{
          data: [data.healthy_count, data.diseased_count],
          backgroundColor: ["#6fae7f", "#e0b872"],
          borderColor: "#15181e",
          borderWidth: 2,
        }],
      },
      options: { plugins: { legend: { position: "bottom" } } },
    });
  }
}

function renderRecentStrip(predictions) {
  const strip = byId("recentStrip");
  const emptyHint = byId("recentEmptyHint");
  if (!strip) return;
  strip.innerHTML = "";

  if (predictions.length === 0) {
    if (emptyHint) emptyHint.hidden = false;
    return;
  }
  if (emptyHint) emptyHint.hidden = true;

  predictions.forEach((p) => {
    const div = document.createElement("div");
    div.className = "recent-thumb";
    div.title = p.display_name;
    div.innerHTML = `
      <img src="${p.image_url ? API_BASE + p.image_url : ''}" alt="${p.display_name}" onerror="this.style.opacity=0.2">
      <span class="thumb-badge">${p.is_healthy ? "✅" : "⚠️"}</span>
    `;
    on(div, "click", () => openDetailModal(p.id));
    strip.appendChild(div);
  });
}

function renderThisWeekStat(predictions) {
  const el = byId("statThisWeek");
  if (!el) return;
  const now = new Date();
  const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const count = predictions.filter((p) => new Date(p.timestamp + "Z") >= weekAgo).length;
  el.textContent = count;
}

function renderConfidenceTrend(predictions) {
  const canvas = byId("confidenceTrendChart");
  const emptyHint = byId("trendEmptyHint");
  if (!canvas) return;

  if (predictions.length < 2) {
    if (emptyHint) emptyHint.hidden = false;
    canvas.hidden = true;
    return;
  }
  if (emptyHint) emptyHint.hidden = true;
  canvas.hidden = false;

  // history comes back most-recent-first; put in chronological order, keep last 20
  const chronological = [...predictions].reverse().slice(-20);
  const labels = chronological.map((p) => {
    const d = new Date(p.timestamp + "Z");
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  });
  const values = chronological.map((p) => Math.round(p.confidence * 100));

  if (trendChart) trendChart.destroy();
  trendChart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Confidence %",
        data: values,
        borderColor: "#e0b872",
        backgroundColor: "rgba(224,184,114,0.15)",
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointBackgroundColor: "#e0b872",
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, ticks: { callback: (v) => `${v}%` } },
      },
    },
  });
}

function renderCropBreakdown(predictions) {
  const canvas = byId("cropBreakdownChart");
  const emptyHint = byId("cropEmptyHint");
  if (!canvas) return;

  if (predictions.length === 0) {
    if (emptyHint) emptyHint.hidden = false;
    canvas.hidden = true;
    return;
  }
  if (emptyHint) emptyHint.hidden = true;
  canvas.hidden = false;

  const counts = {};
  predictions.forEach((p) => {
    const crop = (p.display_name || "Unknown").split(" - ")[0].trim();
    counts[crop] = (counts[crop] || 0) + 1;
  });
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const palette = ["#d97757", "#e0b872", "#6fae7f", "#8ea9c7", "#c2673f", "#a888c9", "#d9705a"];

  if (cropChart) cropChart.destroy();
  cropChart = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: entries.map(([crop]) => crop),
      datasets: [{
        data: entries.map(([, count]) => count),
        backgroundColor: entries.map((_, i) => palette[i % palette.length]),
        borderColor: "#15181e",
        borderWidth: 2,
      }],
    },
    options: { plugins: { legend: { position: "bottom" } } },
  });
}

function renderActivityChart(predictions) {
  const canvas = byId("activityChart");
  const emptyHint = byId("activityEmptyHint");
  if (!canvas) return;

  if (predictions.length === 0) {
    if (emptyHint) emptyHint.hidden = false;
    canvas.hidden = true;
    return;
  }
  if (emptyHint) emptyHint.hidden = true;
  canvas.hidden = false;

  // build a 14-day window ending today, zero-filled
  const days = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() - i);
    days.push(d);
  }
  const counts = days.map((day) => {
    const next = new Date(day);
    next.setDate(next.getDate() + 1);
    return predictions.filter((p) => {
      const t = new Date(p.timestamp + "Z");
      return t >= day && t < next;
    }).length;
  });
  const labels = days.map((d) => d.toLocaleDateString(undefined, { month: "short", day: "numeric" }));

  if (activityChart) activityChart.destroy();
  activityChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Scans",
        data: counts,
        backgroundColor: "#6fae7f",
        borderRadius: 4,
        maxBarThickness: 28,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
    },
  });
}

on(byId("refreshDashboardBtn"), "click", loadDashboard);
loadDashboard();

// ===================== LOCATION CAPTURE =====================
const attachLocationToggle = byId("attachLocationToggle");
const locationStatus = byId("locationStatus");

on(attachLocationToggle, "change", () => {
  if (!attachLocationToggle.checked) {
    currentCoords = null;
    if (locationStatus) locationStatus.textContent = "";
    return;
  }
  if (!navigator.geolocation) {
    if (locationStatus) locationStatus.textContent = "Geolocation not supported.";
    attachLocationToggle.checked = false;
    return;
  }
  if (locationStatus) locationStatus.textContent = "Getting location...";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      currentCoords = { lat: pos.coords.latitude, lon: pos.coords.longitude };
      if (locationStatus) locationStatus.textContent = `📍 ${currentCoords.lat.toFixed(3)}, ${currentCoords.lon.toFixed(3)}`;
    },
    () => {
      if (locationStatus) locationStatus.textContent = "Could not get location.";
      attachLocationToggle.checked = false;
    }
  );
});

// ===================== IMAGE UPLOAD =====================
const uploadBox = byId("uploadBox");
const uploadLabel = byId("uploadLabel");
const imageInput = byId("imageInput");
const previewImg = byId("previewImg");
const predictBtn = byId("predictBtn");
const predictBtnText = byId("predictBtnText");
const predictSpinner = byId("predictSpinner");
const resultBox = byId("resultBox");
const resultImg = byId("resultImg");
const resultClass = byId("resultClass");
const confidenceFill = byId("confidenceFill");
const confidenceText = byId("confidenceText");
const precautionText = byId("precautionText");
const topkList = byId("topkList");
const predictError = byId("predictError");
const resetBtn = byId("resetBtn");

on(uploadBox, "click", () => imageInput && imageInput.click());
["dragenter", "dragover"].forEach((evt) =>
  on(uploadBox, evt, (e) => { e.preventDefault(); uploadBox.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach((evt) =>
  on(uploadBox, evt, (e) => { e.preventDefault(); uploadBox.classList.remove("drag-over"); })
);
on(uploadBox, "drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});
on(imageInput, "change", () => {
  const file = imageInput.files[0];
  if (file) handleFileSelected(file);
});

function handleFileSelected(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    if (previewImg) { previewImg.src = e.target.result; previewImg.hidden = false; }
  };
  reader.readAsDataURL(file);
  if (uploadLabel) uploadLabel.textContent = file.name;
  if (predictBtn) predictBtn.disabled = false;
  if (resultBox) resultBox.hidden = true;
  if (predictError) predictError.hidden = true;
}

on(resetBtn, "click", () => {
  selectedFile = null;
  lastPrediction = null;
  if (imageInput) imageInput.value = "";
  if (previewImg) { previewImg.hidden = true; previewImg.src = ""; }
  if (uploadLabel) uploadLabel.textContent = "Click or drag a leaf image here";
  if (predictBtn) predictBtn.disabled = true;
  if (resultBox) resultBox.hidden = true;
  if (predictError) predictError.hidden = true;
});

// ===================== PREDICT =====================
on(predictBtn, "click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  if (predictBtnText) predictBtnText.textContent = "Analyzing...";
  if (predictSpinner) predictSpinner.hidden = false;
  if (predictError) predictError.hidden = true;
  if (resultBox) resultBox.hidden = true;

  const formData = new FormData();
  formData.append("file", selectedFile);
  if (currentCoords) {
    formData.append("latitude", currentCoords.lat);
    formData.append("longitude", currentCoords.lon);
  }

  try {
    const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    lastPrediction = data;
    renderPrediction(data);
    updateSustainabilityHint(data);
  } catch (e) {
    if (predictError) { predictError.textContent = `Prediction failed: ${e.message}`; predictError.hidden = false; }
  } finally {
    predictBtn.disabled = false;
    if (predictBtnText) predictBtnText.textContent = "Analyze Leaf";
    if (predictSpinner) predictSpinner.hidden = true;
  }
});

function renderPrediction(data) {
  if (resultImg && previewImg) resultImg.src = previewImg.src;
  if (resultClass) resultClass.textContent = data.display_name;
  const pct = Math.round(data.confidence * 100);
  if (confidenceFill) confidenceFill.style.width = `${pct}%`;
  if (confidenceText) confidenceText.textContent = `Confidence: ${pct}%`;
  if (precautionText) precautionText.textContent = data.precaution;

  if (topkList) {
    topkList.innerHTML = "";
    data.top_k.slice(1).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.display_name} (${Math.round(item.confidence * 100)}%)`;
      topkList.appendChild(li);
    });
  }

  if (resultBox) resultBox.hidden = false;
}

function updateSustainabilityHint(data) {
  const hint = byId("sustainabilityHint");
  if (!hint) return;
  hint.textContent = data.is_healthy
    ? `Detected: ${data.display_name} (healthy) -- will be used as "healthy" in the score.`
    : `Detected: ${data.display_name} at ${Math.round(data.confidence * 100)}% confidence -- will be factored into the score.`;
}

// ===================== SUSTAINABILITY SCORE =====================
const soilMoisture = byId("soilMoisture");
const resourceLevel = byId("resourceLevel");
const scoreBtn = byId("scoreBtn");
const scoreBtnText = byId("scoreBtnText");
const scoreSpinner = byId("scoreSpinner");
const scoreResultBox = byId("scoreResultBox");
const scoreValue = byId("scoreValue");
const breakdownList = byId("breakdownList");
const formulaText = byId("formulaText");
const suggestionsList = byId("suggestionsList");
const scoreError = byId("scoreError");

on(scoreBtn, "click", async () => {
  scoreBtn.disabled = true;
  if (scoreBtnText) scoreBtnText.textContent = "Computing...";
  if (scoreSpinner) scoreSpinner.hidden = false;
  if (scoreError) scoreError.hidden = true;
  if (scoreResultBox) scoreResultBox.hidden = true;

  const payload = {
    soil_moisture_pct: parseFloat(soilMoisture.value),
    is_healthy: lastPrediction ? lastPrediction.is_healthy : true,
    disease_confidence: lastPrediction ? lastPrediction.confidence : 0.0,
    fertilizer_pesticide_level: resourceLevel.value,
  };

  try {
    const res = await fetch(`${API_BASE}/api/sustainability-score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }
    renderScore(await res.json());
  } catch (e) {
    if (scoreError) { scoreError.textContent = `Score computation failed: ${e.message}`; scoreError.hidden = false; }
  } finally {
    scoreBtn.disabled = false;
    if (scoreBtnText) scoreBtnText.textContent = "Compute Sustainability Score";
    if (scoreSpinner) scoreSpinner.hidden = true;
  }
});

function renderScore(data) {
  if (scoreValue) scoreValue.textContent = data.final_score;
  if (breakdownList) {
    breakdownList.innerHTML = "";
    Object.entries(data.breakdown).forEach(([key, value]) => {
      const li = document.createElement("li");
      li.textContent = `${key.replace(/_/g, " ")}: ${value}`;
      breakdownList.appendChild(li);
    });
  }
  if (formulaText) formulaText.textContent = `Formula: ${data.formula}`;
  if (suggestionsList) {
    suggestionsList.innerHTML = "";
    data.suggestions.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      suggestionsList.appendChild(li);
    });
  }
  if (scoreResultBox) scoreResultBox.hidden = false;
}

// ===================== WEATHER ADVISORY =====================
const latInput = byId("latInput");
const lonInput = byId("lonInput");
const locBtn = byId("locBtn");
const weatherBtn = byId("weatherBtn");
const weatherBtnText = byId("weatherBtnText");
const weatherSpinner = byId("weatherSpinner");
const weatherResultBox = byId("weatherResultBox");
const rainToday = byId("rainToday");
const rainTomorrow = byId("rainTomorrow");
const tempMax = byId("tempMax");
const weatherActions = byId("weatherActions");
const weatherSource = byId("weatherSource");
const weatherError = byId("weatherError");

on(locBtn, "click", () => {
  if (!navigator.geolocation) {
    if (weatherError) { weatherError.textContent = "Geolocation not supported by this browser."; weatherError.hidden = false; }
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      if (latInput) latInput.value = pos.coords.latitude.toFixed(4);
      if (lonInput) lonInput.value = pos.coords.longitude.toFixed(4);
    },
    () => {
      if (weatherError) { weatherError.textContent = "Could not get location -- enter latitude/longitude manually."; weatherError.hidden = false; }
    }
  );
});

on(weatherBtn, "click", async () => {
  weatherBtn.disabled = true;
  if (weatherBtnText) weatherBtnText.textContent = "Fetching...";
  if (weatherSpinner) weatherSpinner.hidden = false;
  if (weatherError) weatherError.hidden = true;
  if (weatherResultBox) weatherResultBox.hidden = true;

  const lat = parseFloat(latInput.value);
  const lon = parseFloat(lonInput.value);
  const diseaseDetected = lastPrediction ? !lastPrediction.is_healthy : false;
  const params = new URLSearchParams({ lat, lon, disease_detected: diseaseDetected });

  try {
    const res = await fetch(`${API_BASE}/api/weather-advisory?${params.toString()}`);
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }
    renderWeather(await res.json());
  } catch (e) {
    if (weatherError) { weatherError.textContent = `Weather advisory failed: ${e.message}`; weatherError.hidden = false; }
  } finally {
    weatherBtn.disabled = false;
    if (weatherBtnText) weatherBtnText.textContent = "Get Weather Advisory";
    if (weatherSpinner) weatherSpinner.hidden = true;
  }
});

function renderWeather(data) {
  if (rainToday) rainToday.textContent = data.rain_probability_today_pct;
  if (rainTomorrow) rainTomorrow.textContent = data.rain_probability_tomorrow_pct;
  if (tempMax) tempMax.textContent = data.temperature_max_c;
  if (weatherActions) {
    weatherActions.innerHTML = "";
    data.actions.forEach((a) => {
      const li = document.createElement("li");
      li.textContent = a;
      weatherActions.appendChild(li);
    });
  }
  if (weatherSource) weatherSource.textContent = `Source: ${data.source}`;
  if (weatherResultBox) weatherResultBox.hidden = false;
}

// ===================== HISTORY =====================
const loadHistoryBtn = byId("loadHistoryBtn");
const historyError = byId("historyError");
const historyEmpty = byId("historyEmpty");
const historyGrid = byId("historyGrid");

on(loadHistoryBtn, "click", async () => {
  if (historyError) historyError.hidden = true;
  try {
    const res = await fetch(`${API_BASE}/api/history?limit=50`);
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    const data = await res.json();
    renderHistoryGrid(data.predictions);
  } catch (e) {
    if (historyError) { historyError.textContent = `Failed to load history: ${e.message}`; historyError.hidden = false; }
  }
});

function renderHistoryGrid(predictions) {
  if (!historyGrid) return;
  historyGrid.innerHTML = "";
  if (predictions.length === 0) {
    if (historyEmpty) historyEmpty.hidden = false;
    return;
  }
  if (historyEmpty) historyEmpty.hidden = true;

  predictions.forEach((p) => {
    const card = document.createElement("div");
    card.className = "history-card";
    const time = new Date(p.timestamp + "Z").toLocaleDateString();
    const statusClass = p.is_healthy ? "healthy" : "diseased";
    const statusLabel = p.is_healthy ? "Healthy" : "Diseased";

    card.innerHTML = `
      <img class="history-card-img" src="${p.image_url ? API_BASE + p.image_url : ''}" alt="${p.display_name}" onerror="this.style.opacity=0.15">
      <div class="history-card-body">
        <div class="history-card-title">${p.display_name}</div>
        <div class="history-card-meta">
          <span>${time}</span>
          <span class="status-pill ${statusClass}">${statusLabel}</span>
        </div>
      </div>
    `;
    on(card, "click", () => openDetailModal(p.id));
    historyGrid.appendChild(card);
  });
}

// ===================== DETAIL MODAL =====================
const modalOverlay = byId("modalOverlay");
const modalBody = byId("modalBody");
const modalClose = byId("modalClose");

function closeModal() {
  if (modalOverlay) modalOverlay.hidden = true;
  if (modalBody) modalBody.innerHTML = ""; // clear so stale content never lingers
}

async function openDetailModal(id) {
  if (!modalBody || !modalOverlay) return;
  modalBody.innerHTML = `<p class="hint">Loading...</p>`;
  modalOverlay.hidden = false;
  try {
    const res = await fetch(`${API_BASE}/api/predictions/${id}`);
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    const p = await res.json();
    try {
      renderModal(p);
    } catch (renderErr) {
      console.error("renderModal failed:", renderErr);
      modalBody.innerHTML = `<p class="error-box">Couldn't display this scan's details.</p>`;
    }
  } catch (e) {
    modalBody.innerHTML = `<p class="error-box">Failed to load detail: ${e.message}</p>`;
  }
}

function renderModal(p) {
  if (!modalBody) return;
  const time = new Date(p.timestamp + "Z").toLocaleString();
  const statusClass = p.is_healthy ? "healthy" : "diseased";
  const statusLabel = p.is_healthy ? "Healthy" : "Diseased";

  let topkHtml = "";
  if (p.top_k && p.top_k.length > 1) {
    topkHtml = `<div class="topk"><h4>Alternatives considered</h4><ul>` +
      p.top_k.slice(1).map((t) => `<li>${t.display_name} (${Math.round(t.confidence * 100)}%)</li>`).join("") +
      `</ul></div>`;
  }

  modalBody.innerHTML = `
    ${p.image_url ? `<img src="${API_BASE + p.image_url}" alt="${p.display_name}">` : ""}
    <h3>${p.display_name}</h3>
    <div class="modal-detail-row"><span>Status</span> <span class="status-pill ${statusClass}">${statusLabel}</span></div>
    <div class="modal-detail-row"><span>Confidence</span> <span>${Math.round(p.confidence * 100)}%</span></div>
    <div class="modal-detail-row"><span>Scanned</span> <span>${time}</span></div>
    ${p.latitude ? `<div class="modal-detail-row"><span>Location</span> <span>${p.latitude.toFixed(4)}, ${p.longitude.toFixed(4)}</span></div>` : ""}
    <div class="modal-detail-row"><span>Precaution</span> <span>${p.precaution}</span></div>
    ${topkHtml}
  `;
}

on(modalClose, "click", closeModal);
on(modalOverlay, "click", (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modalOverlay && !modalOverlay.hidden) closeModal();
});

// ===================== FIELD MAP =====================
const loadMapBtn = byId("loadMapBtn");
const mapError = byId("mapError");
const mapEmpty = byId("mapEmpty");
const mapContainer = byId("mapContainer");

on(loadMapBtn, "click", async () => {
  if (mapError) mapError.hidden = true;
  try {
    const res = await fetch(`${API_BASE}/api/map-points`);
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    const data = await res.json();
    renderMap(data.points);
  } catch (e) {
    if (mapError) { mapError.textContent = `Failed to load map: ${e.message}`; mapError.hidden = false; }
  }
});

function renderMap(points) {
  if (!mapContainer) return;
  if (points.length === 0) {
    if (mapEmpty) mapEmpty.hidden = false;
    mapContainer.hidden = true;
    return;
  }
  if (mapEmpty) mapEmpty.hidden = true;
  mapContainer.hidden = false;

  if (fieldMap) { fieldMap.remove(); fieldMap = null; }

  fieldMap = L.map(mapContainer).setView([points[0].latitude, points[0].longitude], 10);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(fieldMap);

  const bounds = [];
  points.forEach((p) => {
    const marker = L.marker([p.latitude, p.longitude]).addTo(fieldMap);
    const imgHtml = p.image_url ? `<img class="map-popup-img" src="${API_BASE + p.image_url}">` : "";
    marker.bindPopup(`
      ${imgHtml}
      <div class="map-popup-title">${p.display_name}</div>
      <div>${p.is_healthy ? "✅ Healthy" : "⚠️ " + Math.round(p.confidence * 100) + "% confidence"}</div>
    `);
    bounds.push([p.latitude, p.longitude]);
  });

  if (bounds.length > 1) fieldMap.fitBounds(bounds, { padding: [30, 30] });
  setTimeout(() => fieldMap.invalidateSize(), 100);
}

// ===================== SUPPORTED CLASSES =====================
const loadClassesBtn = byId("loadClassesBtn");
const classesError = byId("classesError");
const classesGrid = byId("classesGrid");

on(loadClassesBtn, "click", async () => {
  if (classesError) classesError.hidden = true;
  try {
    const res = await fetch(`${API_BASE}/api/classes`);
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    const data = await res.json();
    renderClasses(data.classes);
  } catch (e) {
    if (classesError) { classesError.textContent = `Failed to load classes: ${e.message}`; classesError.hidden = false; }
  }
});

function renderClasses(classes) {
  if (!classesGrid) return;
  classesGrid.innerHTML = "";
  classes.forEach((c) => {
    const div = document.createElement("div");
    div.className = "class-chip";
    div.textContent = c.replace(/___/g, " - ").replace(/_/g, " ");
    classesGrid.appendChild(div);
  });
  classesGrid.hidden = false;
}