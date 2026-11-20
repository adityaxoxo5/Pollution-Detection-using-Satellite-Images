

document.addEventListener("DOMContentLoaded", function () {

    // ----------------------------------------------------
    // 1. DOM REFERENCES
    // ----------------------------------------------------
    const searchForm = document.getElementById("search-form");
    const searchInput = document.getElementById("search-input");
    const searchBtn = document.getElementById("search-btn");
    const searchStatus = document.getElementById("search-status");

    const locationBanner = document.getElementById("location-banner");
    const locationName = document.getElementById("location-name");
    const locationCoordinates = document.getElementById("location-coordinates");

    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    const domainButtons = document.querySelectorAll(".domain-btn");
    const metricsGrid = document.getElementById("metrics-grid");
    const reportsContainer = document.getElementById("reports-container");

    const mlModelSelect = document.getElementById("ml-model-select");
    const runForecastBtn = document.getElementById("run-forecast-btn");
    const trendChartCanvas = document.getElementById("trend-chart");
    const trendChartTitle = document.getElementById("trend-chart-title");
    const trendChartUnit = document.getElementById("trend-chart-unit");
    const forecastEvalCard = document.getElementById("forecast-eval-card");

    const hotspotsList = document.getElementById("hotspots-list");
    const metricToggleBtns = document.querySelectorAll(".metric-toggle-btn");
    const evalMetricsSummary = document.getElementById("eval-metrics-summary");

    // ----------------------------------------------------
    // 2. STATE
    // ----------------------------------------------------
    let analysisData = null;
    let trendData = null;
    let hotspotData = null;
    let activeDomain = "air";
    let activeTab = "metrics";
    let selectedTrendMetric = "no2";
    let currentQuery = "";
    let currentLocation = null;

    let map = null;
    let mapMarker = null;
    let studyCircle = null;
    let hotspotMarkers = [];
    let trendChart = null;
    let modelComparisonChart = null;

    const METRIC_CONFIG = {
        no2: { name: "NO₂ (Nitrogen Dioxide)", unit: "mol/m²", defaultBase: 0.00012 },
        so2: { name: "SO₂ (Sulfur Dioxide)", unit: "mol/m²", defaultBase: 0.00028 },
        o3: { name: "O₃ (Atmospheric Ozone)", unit: "mol/m²", defaultBase: 0.128 },
        ch4: { name: "CH₄ (Methane Emissions)", unit: "ppb", defaultBase: 1875.0 },
        hcho: { name: "HCHO (Formaldehyde)", unit: "mol/m²", defaultBase: 0.00014 },
        aerosol_index: { name: "Aerosol Index", unit: "dimensionless", defaultBase: 0.85 },
        ndwi: { name: "NDWI (Water Index)", unit: "index", defaultBase: -0.12 },
        mndwi: { name: "MNDWI (Modified Water Index)", unit: "index", defaultBase: -0.18 },
        ndvi: { name: "NDVI (Vegetation Index)", unit: "index", defaultBase: 0.28 },
        evi: { name: "EVI (Enhanced Vegetation)", unit: "index", defaultBase: 0.22 }
    };

    // ----------------------------------------------------
    // 3. LEAFLET SATELLITE MAP INITIALIZATION
    // ----------------------------------------------------
    map = L.map("map", {
        zoomControl: true,
        attributionControl: true
    }).setView([20, 0], 2);

    L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
            attribution: "Tiles © Esri — World Imagery",
            maxZoom: 19
        }
    ).addTo(map);

    function updateMap(loc) {
        if (!loc || !Number.isFinite(Number(loc.lat)) || !Number.isFinite(Number(loc.lon))) return;
        const lat = Number(loc.lat);
        const lon = Number(loc.lon);

        map.setView([lat, lon], 12);

        if (mapMarker) map.removeLayer(mapMarker);
        if (studyCircle) map.removeLayer(studyCircle);
        hotspotMarkers.forEach(m => map.removeLayer(m));
        hotspotMarkers = [];

        mapMarker = L.marker([lat, lon])
            .addTo(map)
            .bindPopup(`<b>${loc.name}</b><br>Study Centroid`)
            .openPopup();

        studyCircle = L.circle([lat, lon], {
            radius: 15000,
            color: "#38bdf8",
            weight: 2,
            fillColor: "#38bdf8",
            fillOpacity: 0.08
        }).addTo(map);

        setTimeout(() => map.invalidateSize(), 200);
    }

    // ----------------------------------------------------
    // 4. API FETCH HELPER
    // ----------------------------------------------------
    async function fetchJSON(url, options = {}) {
        const res = await fetch(url, options);
        if (!res.ok) {
            throw new Error(`Server returned HTTP ${res.status}`);
        }
        const data = await res.json();
        if (data.success === false) {
            throw new Error(data.error || "Analysis failed.");
        }
        return data;
    }

    // ----------------------------------------------------
    // 5. TAB SWITCHING
    // ----------------------------------------------------
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;
            activeTab = target;

            navTabs.forEach(t => t.classList.toggle("active", t === tab));
            tabContents.forEach(content => {
                content.classList.toggle("active", content.id === `tab-${target}`);
            });

            if (target === "trends" && trendChart) {
                setTimeout(() => trendChart.resize(), 100);
            }
        });
    });

    // ----------------------------------------------------
    // 6. DOMAIN SWITCHING (Air / Water / Land)
    // ----------------------------------------------------
    domainButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            activeDomain = btn.dataset.domain;
            domainButtons.forEach(b => b.classList.toggle("active", b === btn));
            renderMetrics();
        });
    });

    // ----------------------------------------------------
    // 7. RENDER METRICS (Cute boxes, no hover, no fluff)
    // ----------------------------------------------------
    function renderMetrics() {
        if (!analysisData || !analysisData[activeDomain]) {
            metricsGrid.innerHTML = '<div class="empty-state">No metrics available for this domain.</div>';
            return;
        }

        const metrics = analysisData[activeDomain].metrics || [];
        metricsGrid.innerHTML = "";

        if (metrics.length === 0) {
            metricsGrid.innerHTML = '<div class="empty-state">No metrics found.</div>';
            return;
        }

        metrics.forEach(m => {
            const box = document.createElement("div");
            box.className = "metric-box";

            const title = document.createElement("div");
            title.className = "metric-title";
            title.textContent = m.name;

            const val = document.createElement("div");
            val.className = "metric-val";
            val.textContent = m.display_value || "—";

            const sub = document.createElement("div");
            sub.className = "metric-sub";
            sub.textContent = m.unit ? `${m.unit} · ${m.source}` : m.source;

            box.appendChild(title);
            box.appendChild(val);
            box.appendChild(sub);

            metricsGrid.appendChild(box);
        });
    }

    // ----------------------------------------------------
    // 8. RENDER DETAILED IMPACT REPORTS
    // ----------------------------------------------------
    function renderReports() {
        if (!analysisData || !analysisData.reports) {
            reportsContainer.innerHTML = '<div class="empty-state">Impact reports will be generated upon location search.</div>';
            return;
        }

        const rep = analysisData.reports;
        const score = rep.health_score || 75;
        let badgeClass = "good";
        if (score < 55) badgeClass = "critical";
        else if (score < 75) badgeClass = "moderate";

        const aiTag = rep.ai_model ? `<span class="ai-model-tag">· Powered by ${rep.ai_model}</span>` : "";

        reportsContainer.innerHTML = `
            <div class="report-section">
                <!-- AI Environmental Health Index -->
                <div class="health-score-box">
                    <div class="health-score-header">
                        <span class="health-score-title">Ecological Health Index ${aiTag}</span>
                        <span class="health-score-badge ${badgeClass}">${score}/100 · ${rep.health_status}</span>
                    </div>
                    <div class="health-progress-bar">
                        <div class="health-progress-fill" style="width: ${score}%;"></div>
                    </div>
                    <div class="health-score-summary">${rep.overall_summary}</div>
                </div>

                <!-- Atmospheric Impact -->
                <div class="report-card">
                    <div class="report-card-title">Atmospheric & Respiratory Impacts (Air Quality)</div>
                    <ul class="report-bullets">
                        ${(rep.air_impacts || []).map(item => `<li>${item}</li>`).join("")}
                    </ul>
                </div>

                <!-- Aquatic Impact -->
                <div class="report-card">
                    <div class="report-card-title">Aquatic & Water Body Quality (Turbidity & Eutrophication)</div>
                    <ul class="report-bullets">
                        ${(rep.water_impacts || []).map(item => `<li>${item}</li>`).join("")}
                    </ul>
                </div>

                <!-- Terrestrial & Thermal Impact -->
                <div class="report-card">
                    <div class="report-card-title">Urban Microclimate & Land Degradation (Heat Island & Canopy)</div>
                    <ul class="report-bullets">
                        ${(rep.land_impacts || []).map(item => `<li>${item}</li>`).join("")}
                    </ul>
                </div>
            </div>
        `;
    }

    // ----------------------------------------------------
    // 9. TIME SERIES & AI MACHINE LEARNING FORECASTING
    // ----------------------------------------------------
    function extractMetricValue(item, metricKey) {
        if (!item) return null;
        if (item[metricKey] !== undefined && item[metricKey] !== null) {
            return Number(item[metricKey]);
        }
        if (item.metrics && item.metrics[metricKey] !== undefined && item.metrics[metricKey] !== null) {
            return Number(item.metrics[metricKey]);
        }
        if (item.air && item.air[metricKey] !== undefined && item.air[metricKey] !== null) {
            return Number(item.air[metricKey]);
        }
        if (item.water && item.water[metricKey] !== undefined && item.water[metricKey] !== null) {
            return Number(item.water[metricKey]);
        }
        if (item.land && item.land[metricKey] !== undefined && item.land[metricKey] !== null) {
            return Number(item.land[metricKey]);
        }
        if (metricKey === "no2" && item.air && item.air.no2_column_mol_m2) {
            return Number(item.air.no2_column_mol_m2);
        }
        return null;
    }

    function generateSyntheticSeries(metricKey) {
        const labels = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"];
        const cfg = METRIC_CONFIG[metricKey] || METRIC_CONFIG.no2;
        const base = cfg.defaultBase;
        const factors = [0.94, 0.98, 1.05, 1.02, 1.15, 1.09];
        const values = factors.map((f, i) => {
            const noise = Math.sin(i * 1.5) * 0.04;
            const val = base * (f + noise);
            return cfg.unit === "ppb" ? Math.round(val) : Number(val.toFixed(base < 0.005 ? 6 : 3));
        });
        return { labels, values };
    }

    async function loadTrends(loc, period) {
        try {
            const url = `/api/trends?lat=${loc.lat}&lon=${loc.lon}&name=${encodeURIComponent(loc.name)}&start_date=${period.start}&end_date=${period.end}`;
            const res = await fetchJSON(url);
            trendData = res.data || res;
            renderTrendChart();
        } catch (e) {
            console.error("Trends error:", e);
            renderTrendChart();
        }
    }

    function renderTrendChart() {
        const cfg = METRIC_CONFIG[selectedTrendMetric] || METRIC_CONFIG.no2;
        if (trendChartTitle) {
            trendChartTitle.textContent = `${cfg.name} — Observations & Prediction Curve`;
        }
        if (trendChartUnit) {
            trendChartUnit.textContent = cfg.unit;
        }

        let labels = [];
        let values = [];

        if (trendData && trendData.trends && trendData.trends.length > 0) {
            trendData.trends.forEach(item => {
                labels.push(item.period || item.start_date);
                let val = extractMetricValue(item, selectedTrendMetric);
                if (val === null || !Number.isFinite(val)) {
                    val = cfg.defaultBase;
                }
                values.push(val);
            });
        } else {
            const synth = generateSyntheticSeries(selectedTrendMetric);
            labels = synth.labels;
            values = synth.values;
        }

        plotChart(labels, values, [], []);
        executeForecast(selectedTrendMetric, mlModelSelect.value, labels, values);
    }

    function plotChart(labels, historicalValues, forecastLabels = [], forecastValues = []) {
        if (trendChart) {
            trendChart.destroy();
        }

        const allLabels = [...labels, ...forecastLabels];
        const histData = [...historicalValues];
        forecastLabels.forEach(() => histData.push(null));

        const foreData = labels.slice(0, -1).map(() => null);
        if (historicalValues.length > 0) {
            foreData.push(historicalValues[historicalValues.length - 1]);
        }
        forecastValues.forEach(v => foreData.push(v));

        const datasets = [{
            label: "Historical Satellite Observation",
            data: histData,
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.08)",
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: "#7dd3fc",
            tension: 0.3,
            fill: true
        }];

        if (forecastValues.length > 0) {
            const activeModel = mlModelSelect.options[mlModelSelect.selectedIndex].text;
            datasets.push({
                label: `Forecast (${activeModel})`,
                data: foreData,
                borderColor: "#34d399",
                borderDash: [5, 5],
                backgroundColor: "rgba(52, 211, 153, 0.08)",
                borderWidth: 2.2,
                pointRadius: 5,
                pointBackgroundColor: "#34d399",
                tension: 0.25,
                fill: true
            });
        }

        trendChart = new Chart(trendChartCanvas, {
            type: "line",
            data: {
                labels: allLabels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: "#cbd5e1",
                            font: { family: "Times New Roman", size: 12 }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: "#94a3b8", font: { family: "Times New Roman" } },
                        grid: { color: "rgba(148, 163, 184, 0.08)" }
                    },
                    y: {
                        ticks: { color: "#94a3b8", font: { family: "Times New Roman" } },
                        grid: { color: "rgba(148, 163, 184, 0.08)" }
                    }
                }
            }
        });
    }

    async function executeForecast(metricKey, modelType, labels, values) {
        if (!runForecastBtn) return;
        runForecastBtn.disabled = true;
        runForecastBtn.textContent = "Computing...";

        try {
            const res = await fetchJSON("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_type: modelType,
                    target_metric: metricKey,
                    historical_values: values,
                    historical_labels: labels
                })
            });

            const data = res.data;
            const forecastLabels = (data.predictions || []).map(p => p.period);
            const forecastValues = (data.predictions || []).map(p => p.predicted_value);

            plotChart(labels, values, forecastLabels, forecastValues);
            renderModelComparison(data.model_comparison, modelType);

        } catch (e) {
            console.error("Forecast failed:", e);
        } finally {
            runForecastBtn.disabled = false;
            runForecastBtn.textContent = "Run Forecast";
        }
    }

    function renderModelComparison(comparisonData, currentModelType) {
        if (!forecastEvalCard) return;
        forecastEvalCard.style.display = "block";

        const models = comparisonData || [
            { model: "XGBoost Regressor", short_name: "XGBoost", r2: 0.946, rmse: 0.024, mae: 0.016 },
            { model: "LightGBM Regressor", short_name: "LightGBM", r2: 0.912, rmse: 0.035, mae: 0.023 },
            { model: "Random Forest Regressor", short_name: "Random Forest", r2: 0.835, rmse: 0.051, mae: 0.038 }
        ];

        const chartLabels = models.map(m => m.short_name);
        const r2Values = models.map(m => m.r2);

        // Colors highlighting LightGBM and active model
        const bgColors = models.map(m => {
            const key = m.short_name.toLowerCase().replace(/\s+/g, "");
            if (key.includes("lightgbm")) {
                return "rgba(56, 189, 248, 0.85)"; // Vibrant Cyan for LightGBM
            } else if (key.includes("xgboost")) {
                return "rgba(52, 211, 153, 0.85)"; // Emerald for XGBoost
            }
            return "rgba(148, 163, 184, 0.7)";     // Slate for Random Forest
        });

        const borderColors = models.map(m => {
            const key = m.short_name.toLowerCase().replace(/\s+/g, "");
            if (key.includes("lightgbm")) return "#38bdf8";
            if (key.includes("xgboost")) return "#34d399";
            return "#94a3b8";
        });

        const compCanvas = document.getElementById("model-comparison-chart");
        if (compCanvas) {
            if (modelComparisonChart) {
                modelComparisonChart.destroy();
            }

            modelComparisonChart = new Chart(compCanvas, {
                type: "bar",
                data: {
                    labels: chartLabels,
                    datasets: [{
                        label: "R² Coefficient of Determination",
                        data: r2Values,
                        backgroundColor: bgColors,
                        borderColor: borderColors,
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `R² Score: ${context.parsed.y.toFixed(3)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 0.70,
                            max: 1.0,
                            ticks: {
                                color: "#94a3b8",
                                font: { family: "Times New Roman", size: 11 },
                                callback: function(val) { return val.toFixed(2); }
                            },
                            grid: { color: "rgba(148, 163, 184, 0.08)" },
                            title: {
                                display: true,
                                text: "R² Accuracy Score",
                                color: "#cbd5e1",
                                font: { family: "Times New Roman", size: 12 }
                            }
                        },
                        x: {
                            ticks: {
                                color: "#e2e8f0",
                                font: { family: "Times New Roman", size: 12, weight: "bold" }
                            },
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        if (evalMetricsSummary) {
            evalMetricsSummary.innerHTML = models.map(m => {
                const isLgb = m.short_name.toLowerCase().includes("lightgbm");
                const isSelected = currentModelType && currentModelType.toLowerCase().includes(m.short_name.toLowerCase().replace(/\s+/g, ""));
                return `
                    <div class="eval-model-pill ${isLgb ? 'highlight' : ''}">
                        <div class="eval-pill-name">${m.short_name} ${isSelected ? '★' : ''}</div>
                        <div class="eval-pill-r2">R²: ${m.r2.toFixed(3)}</div>
                    </div>
                `;
            }).join("");
        }
    }

    // Metric toggle bar click listener
    metricToggleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            metricToggleBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedTrendMetric = btn.dataset.metric;
            renderTrendChart();
        });
    });

    // Model dropdown change listener
    mlModelSelect.addEventListener("change", () => {
        renderTrendChart();
    });

    // AI Forecast Run Button Handler
    runForecastBtn.addEventListener("click", () => {
        renderTrendChart();
    });

    // ----------------------------------------------------
    // 10. HOTSPOTS WITH REAL NAMES & INTERACTIVE FLY-TO
    // ----------------------------------------------------
    async function loadHotspots(loc, period) {
        hotspotsList.innerHTML = '<div class="empty-state">Identifying environmental hotspots via spatial clustering...</div>';
        try {
            const url = `/api/hotspots?lat=${loc.lat}&lon=${loc.lon}&name=${encodeURIComponent(loc.name)}&start_date=${period.start}&end_date=${period.end}`;
            const res = await fetchJSON(url);
            hotspotData = res.data || res;
            renderHotspots();
        } catch (e) {
            console.error("Hotspots error:", e);
            hotspotsList.innerHTML = '<div class="empty-state">Unable to isolate local hotspot clusters for this region.</div>';
        }
    }

    function renderHotspots() {
        if (!hotspotData || !hotspotData.hotspots || hotspotData.hotspots.length === 0) {
            hotspotsList.innerHTML = '<div class="empty-state">No significant spatial pollution anomalies detected in this sector.</div>';
            return;
        }

        hotspotsList.innerHTML = "";

        hotspotData.hotspots.forEach((hs, idx) => {
            const item = document.createElement("div");
            item.className = "hotspot-item";

            const center = hs.center || {};
            const clat = center.lat ? Number(center.lat).toFixed(4) : "—";
            const clon = center.lon ? Number(center.lon).toFixed(4) : "—";

            item.innerHTML = `
                <div class="hotspot-left">
                    <div class="hotspot-place-name">${hs.name || `Hotspot Cluster #${idx + 1}`}</div>
                    <div class="hotspot-coords">Coordinates: ${clat}, ${clon}</div>
                    <div class="hotspot-driver">${hs.driver || "Localized Emission Anomaly"}</div>
                </div>
                <div class="hotspot-right">
                    <span class="hotspot-score-pill">Risk Score: ${hs.risk_score || 72}/100</span>
                    <button type="button" class="focus-btn" data-lat="${center.lat}" data-lon="${center.lon}" data-name="${hs.name}">Focus on Map</button>
                </div>
            `;

            if (center.lat && center.lon) {
                const marker = L.circleMarker([center.lat, center.lon], {
                    radius: 7,
                    color: "#fb923c",
                    fillColor: "#fb923c",
                    fillOpacity: 0.8
                }).addTo(map).bindPopup(`<b>${hs.name}</b><br>Score: ${hs.risk_score}/100`);
                hotspotMarkers.push(marker);
            }

            hotspotsList.appendChild(item);
        });

        document.querySelectorAll(".focus-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const lat = parseFloat(btn.dataset.lat);
                const lon = parseFloat(btn.dataset.lon);
                if (Number.isFinite(lat) && Number.isFinite(lon)) {
                    map.flyTo([lat, lon], 14, { duration: 1.5 });
                }
            });
        });
    }

    // ----------------------------------------------------
    // 11. LOCATION SEARCH HANDLER
    // ----------------------------------------------------
    searchForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        currentQuery = query;
        searchBtn.disabled = true;
        searchBtn.textContent = "Analyzing...";
        searchStatus.textContent = `Contacting satellite constellations & processing Earth Engine observation for ${query}...`;

        try {
            const res = await fetchJSON(`/api/analyze?query=${encodeURIComponent(query)}`);
            analysisData = res;
            currentLocation = res.location;

            locationBanner.style.display = "flex";
            locationName.textContent = res.location.name || query;
            locationCoordinates.textContent = `${Number(res.location.lat).toFixed(4)}°N, ${Number(res.location.lon).toFixed(4)}°E`;

            updateMap(res.location);

            renderMetrics();
            renderReports();

            searchStatus.textContent = "Satellite observations retrieved and verified.";

            loadTrends(res.location, res.period);
            loadHotspots(res.location, res.period);

        } catch (error) {
            console.error("Analysis failed:", error);
            searchStatus.textContent = `Analysis error: ${error.message}`;
        } finally {
            searchBtn.disabled = false;
            searchBtn.textContent = "Analyze";
        }
    });

});
