/**
 * Leaflet Road Quality Map Logic
 * Handles corridor rendering, discrete condition coloring,
 * segment inspector drawer, ingestion triggers, live video upload,
 * multi-tier layer toggles, and simulated decay.
 */

// Discrete condition colors as specified in section 6
const COLOR_BANDS = {
  "Good": "#2ECC71",
  "Fair": "#F1C40F",
  "Poor": "#E67E22",
  "Bad": "#E74C3C",
  "Severe": "#8E44AD"
};

let map;
let geojsonLayer = null;
let currentSegmentsData = null;
let selectedEdgeId = null;
let currentObservations = [];
let defectPinMarker = null;
let selectedVideoFile = null;
let latestUploadResult = null;
let currentLayerFilter = "all";

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupEventListeners();
  loadSegments();
  updateDemoStatus();
});

function initMap() {
  // Center on Bengaluru ORR (Marathahalli - Kadubeesanahalli - Bellandur stretch)
  map = L.map("map", {
    zoomControl: false
  }).setView([12.935, 77.680], 14);

  // Position zoom controls in top right
  L.control.zoom({ position: "topright" }).addTo(map);

  // CARTO Basemaps — authenticated raster tiles (voyager light style)
  const CARTO_API_KEY = "cb1_3xz5_1_26a32e3a05e08026b2b7a55c";
  L.tileLayer(
    `https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=${CARTO_API_KEY}`,
    {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 19
    }
  ).addTo(map);
}

function setupEventListeners() {
  // Ingestion Trigger Buttons
  document.getElementById("btn-run-tier0").addEventListener("click", () => triggerIngest(0));
  document.getElementById("btn-run-tier1").addEventListener("click", () => triggerIngest(1));
  document.getElementById("btn-run-tier2").addEventListener("click", () => triggerIngest(2));

  // Simulation Controls (Decay & Reset)
  document.getElementById("btn-simulate-decay").addEventListener("click", () => simulateDecay(14.0));
  const decay30Btn = document.getElementById("btn-simulate-decay-30");
  if (decay30Btn) {
    decay30Btn.addEventListener("click", () => simulateDecay(30.0));
  }
  document.getElementById("btn-reset-db").addEventListener("click", resetDatabase);

  // Inspector Close
  document.getElementById("btn-close-inspector").addEventListener("click", closeInspector);

  // Inspector Observation Filter
  const filterSelect = document.getElementById("insp-filter-select");
  if (filterSelect) {
    filterSelect.addEventListener("change", () => renderObservationList());
  }

  // Filter Radios (Layer Views)
  const filterRadios = document.querySelectorAll('input[name="layer-filter"]');
  filterRadios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      applyFilter(e.target.value);
    });
  });

  // Video Upload Modal Controls (New Feature for Tier 1)
  setupVideoUploadModal();

  // IMU Upload Modal Controls
  setupImuUploadModal();

  // GSV Key Config Controls
  setupGsvConfigModal();

  // Lightbox Close
  const closeLightboxBtn = document.getElementById("btn-close-lightbox");
  if (closeLightboxBtn) {
    closeLightboxBtn.addEventListener("click", closeLightbox);
  }
}

function updateStatus(msg, isError = false) {
  const el = document.getElementById("status-message");
  el.textContent = msg;
  el.style.color = isError ? "#e74c3c" : "#F3B518";
}

async function loadSegments(tierFilter = null) {
  const filterToUse = tierFilter !== null ? tierFilter : currentLayerFilter;
  updateStatus(`UPDATING ROAD NETWORK TELEMETRY [VIEW: ${filterToUse.toUpperCase()}]...`);
  try {
    const url = (filterToUse && filterToUse !== "all") ? `/segments?tier_filter=${filterToUse}` : "/segments";
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentSegmentsData = data;
    renderGeoJson(data);
    updateStatus("TELEMETRY SYNCED // READY");
  } catch (err) {
    console.error("Failed to load road segments:", err);
    updateStatus(`FAILED TO LOAD SEGMENTS: ${err.message}`, true);
  }
}

function getSegmentStyle(feature) {
  const props = feature.properties || {};
  const bucket = props.severity_bucket || "Good";
  const color = COLOR_BANDS[bucket] || "#2ECC71";
  const confidence = props.confidence || 0.5;
  const isSelected = selectedEdgeId === props.edge_id;

  return {
    color: isSelected ? "#FFFFFF" : color,
    weight: isSelected ? 8 : (5 + Math.min(3, confidence * 3)),
    opacity: 0.92,
    lineCap: "round",
    lineJoin: "round"
  };
}

function renderGeoJson(geojsonData) {
  if (geojsonLayer) {
    map.removeLayer(geojsonLayer);
  }

  geojsonLayer = L.geoJSON(geojsonData, {
    style: getSegmentStyle,
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      
      // Simple tooltip on hover
      layer.bindTooltip(`
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px;">
          <strong>${p.name || 'ORR Segment'}</strong><br/>
          Score: ${p.score} [${p.severity_bucket}]<br/>
          Confidence: ${Math.round(p.confidence * 100)}%
        </div>
      `, { sticky: true, opacity: 0.9 });

      // Click opens the Inspector Drawer
      layer.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        selectSegment(p.edge_id);
      });
    }
  }).addTo(map);

  // Fit bounds if first load and segments exist
  if (geojsonData.features && geojsonData.features.length > 0 && !selectedEdgeId) {
    try {
      const bounds = geojsonLayer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [40, 40] });
      }
    } catch (e) {
      console.warn("Bounds fitting skipped:", e);
    }
  }
}

async function selectSegment(edgeId) {
  selectedEdgeId = edgeId;
  // Re-style segments to highlight selected
  if (geojsonLayer) {
    geojsonLayer.setStyle(getSegmentStyle);
  }

  const inspector = document.getElementById("inspector-panel");
  inspector.classList.add("open");

  try {
    const res = await fetch(`/segments/${edgeId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    populateInspector(data);
  } catch (err) {
    console.error("Inspector fetch error:", err);
  }
}

function closeInspector() {
  selectedEdgeId = null;
  const inspector = document.getElementById("inspector-panel");
  inspector.classList.remove("open");
  if (defectPinMarker) {
    map.removeLayer(defectPinMarker);
    defectPinMarker = null;
  }
  if (geojsonLayer) {
    geojsonLayer.setStyle(getSegmentStyle);
  }
}

function populateInspector(data) {
  const edge = data.edge || {};
  currentObservations = data.observations || [];

  document.getElementById("insp-edge-id").textContent = edge.edge_id || "--";
  document.getElementById("insp-edge-name").textContent = edge.name || "Outer Ring Road";
  
  const scoreVal = document.getElementById("insp-score");
  scoreVal.textContent = edge.score !== undefined ? Math.round(edge.score) : "--";
  
  const band = edge.severity_bucket || "Good";
  const bandEl = document.getElementById("insp-band");
  bandEl.textContent = `BAND: ${band.toUpperCase()}`;
  bandEl.style.color = COLOR_BANDS[band] || "#2ECC71";

  document.getElementById("insp-conf").textContent = `CONFIDENCE: ${Math.round((edge.confidence || 0) * 100)}%`;
  document.getElementById("insp-defect").textContent = (edge.dominant_defect || "none").toUpperCase();
  document.getElementById("insp-length").textContent = `${edge.length_m ? Math.round(edge.length_m) : 0} METERS`;
  
  const dateStr = edge.last_observed_at ? new Date(edge.last_observed_at).toLocaleString() : "NONE";
  document.getElementById("insp-date").textContent = dateStr;

  // Evidence Matrix counts
  const breakdown = edge.tier_breakdown || {};
  document.getElementById("insp-t0-count").textContent = breakdown.tier0 || 0;
  document.getElementById("insp-t1-count").textContent = breakdown.tier1 || 0;
  document.getElementById("insp-t2-count").textContent = breakdown.tier2 || 0;

  renderObservationList();
}

function renderObservationList() {
  const listEl = document.getElementById("insp-obs-list");
  const filterSelect = document.getElementById("insp-filter-select");
  const filterVal = filterSelect ? filterSelect.value : "defects";

  let filtered = currentObservations;

  if (filterVal === "defects") {
    const defectsOnly = currentObservations.filter(obs => {
      const dt = (obs.defect_type || "").toLowerCase();
      return dt !== "none" && dt !== "clear" && dt !== "smooth";
    });
    // If segment has defects, display them. If it has only clear road captures, display them so Street View images are visible.
    filtered = defectsOnly.length > 0 ? defectsOnly : currentObservations;
  } else if (filterVal === "potholes") {
    filtered = currentObservations.filter(obs => (obs.defect_type || "").toLowerCase() === "pothole");
  } else if (filterVal === "imu_verified") {
    filtered = currentObservations.filter(obs => obs.source_tier === 2 && obs.vibration_match === 1);
  } else if (filterVal === "suppressed") {
    filtered = currentObservations.filter(obs => obs.source_tier === 2 && obs.vibration_match === 0);
  }

  // Sort: detections with thumbnails first, then by confidence descending
  filtered.sort((a, b) => {
    const aHasThumb = !!(a.metadata && a.metadata.thumbnail_url);
    const bHasThumb = !!(b.metadata && b.metadata.thumbnail_url);
    if (aHasThumb !== bHasThumb) return aHasThumb ? -1 : 1;
    return (b.confidence_raw || 0) - (a.confidence_raw || 0);
  });

  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="empty-obs">No observations matching the selected filter.</div>';
    return;
  }

  listEl.innerHTML = filtered.map((obs, idx) => {
    const tier = obs.source_tier;
    const isSuppressed = (obs.vibration_match === 0 && tier === 2);
    const cardClass = `obs-card tier-${tier} ${isSuppressed ? 'suppressed' : ''}`;
    const meta = obs.metadata || {};
    const notes = meta.correlation_result || meta.source || (meta.notes ? meta.notes : "Detection logged");
    const dt = (obs.defect_type || 'clear').toLowerCase();
    const isClear = ['clear', 'none', 'smooth'].includes(dt);

    let thumbnailHtml = "";
    if (meta.thumbnail_url) {
      const confPct = Math.round((obs.confidence_raw || 0.75) * 100);
      const badgeText = isClear ? 'STREET VIEW INSPECTED' : 'YOLOv8 DETECTED';
      const badgeClass = isClear ? 'clear-badge' : 'yolo-badge';
      const captionText = isClear ? 'ROAD CONDITION: CLEAR ROAD' : `DEFECT: ${dt.toUpperCase()} [${obs.severity}]`;
      thumbnailHtml = `
        <div class="obs-thumbnail-wrap">
          <div class="obs-thumbnail-header">
            <span class="${badgeClass}">${badgeText}</span>
            <span class="mono">${confPct}% CONF</span>
          </div>
          <div class="obs-thumbnail-link" onclick="openLightbox('${meta.thumbnail_url}', '${captionText} // CONF: ${confPct}%')">
            <img src="${meta.thumbnail_url}" alt="Road Inspection Snapshot" class="obs-thumbnail" loading="lazy" />
          </div>
          <div class="obs-thumbnail-caption">
            <span>${captionText}</span>
            <span class="view-hint mono">CLICK TO EXPAND ↗</span>
          </div>
        </div>
      `;
    }

    const pinBtn = (obs.lat && obs.lon) ? `
      <button class="btn-pin-defect mono" onclick="pinDefectOnMap(${obs.lat}, ${obs.lon}, '${obs.defect_type || 'defect'}', ${obs.confidence_raw || 0.8})">
        PIN LOC: ${obs.lat.toFixed(4)}, ${obs.lon.toFixed(4)} 📍
      </button>
    ` : '';

    const tagText = isClear ? 'CLEAR (GOOD PAVEMENT)' : `${dt.toUpperCase()} [${obs.severity}]`;
    const tagStyle = isClear ? 'background: #27ae60; color: #fff;' : '';

    return `
      <div class="${cardClass}">
        <div class="obs-card-header">
          <span class="obs-defect-tag" style="${tagStyle}">${tagText}</span>
          <span class="mono">TIER ${tier} // TRUST: ${obs.source_trust}</span>
        </div>
        <div class="mono" style="color: #8C877F; font-size: 9px; margin-bottom: 3px;">
          Captured: ${new Date(obs.capture_date).toLocaleDateString()} | Raw Conf: ${obs.confidence_raw}
        </div>
        <div class="obs-notes">${notes}</div>
        ${pinBtn}
        ${thumbnailHtml}
      </div>
    `;
  }).join("");
}

function pinDefectOnMap(lat, lon, defectType, conf) {
  if (defectPinMarker) {
    map.removeLayer(defectPinMarker);
  }

  const iconHtml = `
    <div style="background-color: #F3B518; color: #1E1F1E; font-weight: 700; font-family: monospace; font-size: 10px; padding: 2px 5px; border: 2px solid #000; border-radius: 3px; box-shadow: 0 2px 6px rgba(0,0,0,0.6); white-space: nowrap;">
      ⚠️ ${defectType.toUpperCase()} (${Math.round(conf * 100)}%)
    </div>
  `;

  const customIcon = L.divIcon({
    html: iconHtml,
    className: "defect-map-pin",
    iconSize: [120, 24],
    iconAnchor: [60, 12]
  });

  defectPinMarker = L.marker([lat, lon], { icon: customIcon }).addTo(map);
  map.setView([lat, lon], 16, { animate: true });
}

function openLightbox(imgUrl, caption) {
  const modal = document.getElementById("image-lightbox-modal");
  const img = document.getElementById("lightbox-img");
  const cap = document.getElementById("lightbox-caption");
  if (modal && img) {
    img.src = imgUrl;
    if (cap) cap.textContent = caption;
    modal.classList.add("active");
  }
}

function closeLightbox() {
  const modal = document.getElementById("image-lightbox-modal");
  if (modal) modal.classList.remove("active");
}

async function triggerIngest(tierNumber) {
  const btn = document.getElementById(`btn-run-tier${tierNumber}`);
  const origText = btn.textContent;
  btn.disabled = true;
  btn.textContent = `INGESTING TIER ${tierNumber}...`;
  updateStatus(`TRIGGERED TIER ${tierNumber} INGESTION PIPELINE...`);

  let payload = {};
  if (tierNumber === 1) {
    const densityEl = document.getElementById("tier1-scan-density");
    const mode = densityEl ? densityEl.value : "dense";
    if (mode === "ultra") {
      payload = { frame_step: 2, conf_threshold: 0.35 };
    } else if (mode === "full") {
      payload = { frame_step: 1, conf_threshold: 0.35 };
    } else if (mode === "standard") {
      payload = { sample_rate: 3.0, conf_threshold: 0.35 };
    } else {
      payload = { sample_rate: 10.0, conf_threshold: 0.35 };
    }
  } else if (tierNumber === 2) {
    const vibeModeEl = document.getElementById("tier2-vibe-mode");
    payload = {
      vibration_mode: vibeModeEl ? vibeModeEl.value : "real_imu",
      num_frames: 15
    };
  }

  try {
    const res = await fetch(`/ingest/tier${tierNumber}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.status === "success") {
      updateStatus(`TIER ${tierNumber} COMPLETE: ${result.message}`);
      await loadSegments();
      await updateDemoStatus();

      // Smooth zoom and pan transition to newly updated road segments
      if (result.touched_edges && result.touched_edges.length > 0 && geojsonLayer) {
        zoomToTouchedEdges(result.touched_edges);
      } else if (selectedEdgeId) {
        selectSegment(selectedEdgeId);
      }
    } else {
      updateStatus(`TIER ${tierNumber} ERROR: ${result.message}`, true);
    }
  } catch (err) {
    console.error("Ingest error:", err);
    updateStatus(`TIER ${tierNumber} INGESTION FAILED: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = origText;
  }
}

function zoomToTouchedEdges(touchedEdgeIds) {
  if (!geojsonLayer || !touchedEdgeIds || touchedEdgeIds.length === 0) return;

  const touchedLayers = [];
  geojsonLayer.eachLayer(layer => {
    if (layer.feature && touchedEdgeIds.includes(layer.feature.properties.edge_id)) {
      touchedLayers.push(layer);
    }
  });

  if (touchedLayers.length > 0) {
    const group = L.featureGroup(touchedLayers);
    const bounds = group.getBounds();
    if (bounds.isValid()) {
      map.flyToBounds(bounds, { padding: [50, 50], duration: 1.2 });
    }
  }

  // Auto-select a priority candidate segment to showcase previews
  const defectCandidate = touchedEdgeIds.find(id => ['25187665', '25187903', '380929043', '311412704', '25186966', '1239003786'].includes(id)) || touchedEdgeIds[0];
  setTimeout(() => {
    selectSegment(defectCandidate);
  }, 1200);
}

// ----------------- Video Upload Modal Implementation (New Feature) ----------------- //

function setupVideoUploadModal() {
  const modal = document.getElementById("video-upload-modal");
  const openBtn = document.getElementById("btn-open-upload-modal");
  const closeBtn = document.getElementById("btn-close-upload-modal");
  const cancelBtn = document.getElementById("btn-cancel-video-upload");
  const browseBtn = document.getElementById("btn-browse-video");
  const fileInput = document.getElementById("video-file-input");
  const dropzone = document.getElementById("video-dropzone");
  const submitBtn = document.getElementById("btn-submit-video-upload");
  const viewResultsBtn = document.getElementById("btn-view-uploaded-results");

  if (!modal || !openBtn) return;

  const openModal = () => {
    modal.classList.add("active");
    resetUploadModalState();
  };

  const closeModal = () => {
    modal.classList.remove("active");
  };

  openBtn.addEventListener("click", openModal);
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeModal);

  // File browsing
  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // Drag and drop
  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  // Submit Upload & Ingestion
  if (submitBtn) {
    submitBtn.addEventListener("click", submitVideoUpload);
  }

  // View Results Button
  if (viewResultsBtn) {
    viewResultsBtn.addEventListener("click", () => {
      closeModal();
      if (latestUploadResult && latestUploadResult.touched_edges) {
        loadSegments().then(() => {
          zoomToTouchedEdges(latestUploadResult.touched_edges);
        });
      }
    });
  }
}

function handleFileSelected(file) {
  selectedVideoFile = file;
  const badge = document.getElementById("selected-file-badge");
  const nameEl = document.getElementById("selected-file-name");
  const sizeEl = document.getElementById("selected-file-size");

  if (badge && nameEl && sizeEl) {
    nameEl.textContent = file.name;
    const mbSize = (file.size / (1024 * 1024)).toFixed(2);
    sizeEl.textContent = `(${mbSize} MB)`;
    badge.style.display = "flex";
  }
}

function resetUploadModalState() {
  selectedVideoFile = null;
  latestUploadResult = null;
  const fileInput = document.getElementById("video-file-input");
  if (fileInput) fileInput.value = "";
  const badge = document.getElementById("selected-file-badge");
  if (badge) badge.style.display = "none";
  const progressWrap = document.getElementById("upload-progress-wrap");
  if (progressWrap) progressWrap.style.display = "none";
  const resultBox = document.getElementById("upload-result-box");
  if (resultBox) resultBox.style.display = "none";
  const submitBtn = document.getElementById("btn-submit-video-upload");
  if (submitBtn) {
    submitBtn.style.display = "inline-block";
    submitBtn.disabled = false;
  }
  const viewResultsBtn = document.getElementById("btn-view-uploaded-results");
  if (viewResultsBtn) viewResultsBtn.style.display = "none";
}

async function submitVideoUpload() {
  if (!selectedVideoFile) {
    alert("Please select or drop a road video file (.mp4, .mov, .avi) to upload.");
    return;
  }

  const submitBtn = document.getElementById("btn-submit-video-upload");
  const progressWrap = document.getElementById("upload-progress-wrap");
  const progressBar = document.getElementById("upload-progress-bar");
  const progressText = document.getElementById("upload-progress-text");
  const resultBox = document.getElementById("upload-result-box");
  const viewResultsBtn = document.getElementById("btn-view-uploaded-results");

  submitBtn.disabled = true;
  progressWrap.style.display = "block";
  resultBox.style.display = "none";
  progressBar.style.width = "25%";
  progressText.textContent = "UPLOADING VIDEO FOOTAGE TO BACKEND...";

  const corridor = document.getElementById("upload-corridor").value;
  const sampleRate = document.getElementById("upload-sample-rate").value;
  const maxFrames = document.getElementById("upload-max-frames").value;
  const confThreshold = document.getElementById("upload-confidence").value;

  const formData = new FormData();
  formData.append("video_file", selectedVideoFile);
  formData.append("corridor", corridor);
  formData.append("sample_rate", sampleRate);
  formData.append("max_frames", maxFrames);
  formData.append("conf_threshold", confThreshold);

  // Progressive fake loading animation to show user that YOLOv8 analysis is actively running
  let progress = 30;
  const progressInterval = setInterval(() => {
    if (progress < 85) {
      progress += Math.floor(Math.random() * 8) + 2;
      progressBar.style.width = `${progress}%`;
      if (progress > 45 && progress < 70) {
        progressText.textContent = "RUNNING YOLOV8 DEFECT DETECTOR (models/best.pt)...";
      } else if (progress >= 70) {
        progressText.textContent = "INTERPOLATING GPS & SNAPPING TO ORR SEGMENTS...";
      }
    }
  }, 700);

  try {
    const res = await fetch("/ingest/tier1/upload", {
      method: "POST",
      body: formData
    });

    clearInterval(progressInterval);
    const result = await res.json();

    if (res.ok && result.status === "success") {
      progressBar.style.width = "100%";
      progressText.textContent = "ANALYSIS COMPLETE // SAVED & FUSED TO ROAD GRAPH";
      latestUploadResult = result;

      // Populate result box
      document.getElementById("res-potholes").textContent = result.potholes_detected || 0;
      document.getElementById("res-obs").textContent = result.observations_count || 0;
      document.getElementById("res-edges-count").textContent = result.touched_edges_count || 0;
      
      const listEl = document.getElementById("res-edges-list");
      if (listEl && result.touched_edges) {
        listEl.innerHTML = `UPDATED EDGES: ${result.touched_edges.map(id => `<span class="edge-tag">${id}</span>`).join(" ")}`;
      }

      resultBox.style.display = "block";
      submitBtn.style.display = "none";
      if (viewResultsBtn) viewResultsBtn.style.display = "inline-block";
      updateStatus(`UPLOAD COMPLETE: ${result.message}`);
    } else {
      throw new Error(result.detail || result.message || "Upload failed");
    }
  } catch (err) {
    clearInterval(progressInterval);
    progressBar.style.width = "0%";
    progressText.textContent = `ERROR: ${err.message}`;
    progressText.style.color = "#e74c3c";
    submitBtn.disabled = false;
    alert(`Video ingestion failed: ${err.message}`);
  }
}

// ----------------- IMU CSV Upload Modal ----------------- //

function setupImuUploadModal() {
  const modal = document.getElementById("imu-upload-modal");
  const openBtn = document.getElementById("btn-open-imu-upload");
  const closeBtn = document.getElementById("btn-close-imu-modal");
  const cancelBtn = document.getElementById("btn-cancel-imu-upload");
  const submitBtn = document.getElementById("btn-submit-imu-upload");
  const fileInput = document.getElementById("imu-file-input");
  const statusEl = document.getElementById("imu-upload-status");

  if (!modal || !openBtn) return;

  openBtn.addEventListener("click", () => {
    modal.classList.add("active");
    if (fileInput) fileInput.value = "";
    if (statusEl) statusEl.textContent = "";
  });

  const closeModal = () => modal.classList.remove("active");
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeModal);

  if (submitBtn) {
    submitBtn.addEventListener("click", async () => {
      if (!fileInput.files || !fileInput.files[0]) {
        alert("Please select a CSV file containing timestamp, acc_z.");
        return;
      }

      submitBtn.disabled = true;
      if (statusEl) statusEl.textContent = "UPLOADING ACCELEROMETER STREAM...";

      const formData = new FormData();
      formData.append("imu_file", fileInput.files[0]);

      try {
        const res = await fetch("/ingest/tier2/upload_imu", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (res.ok && data.status === "success") {
          if (statusEl) statusEl.textContent = `SUCCESS: Loaded ${data.sample_count} records.`;
          updateStatus(data.message);
          setTimeout(closeModal, 1200);
        } else {
          throw new Error(data.detail || data.message || "Failed");
        }
      } catch (err) {
        if (statusEl) statusEl.textContent = `ERROR: ${err.message}`;
      } finally {
        submitBtn.disabled = false;
      }
    });
  }
}

// ----------------- GSV API Key Config Modal ----------------- //

function setupGsvConfigModal() {
  const modal = document.getElementById("gsv-config-modal");
  const openBtn = document.getElementById("btn-config-gsv");
  const closeBtn = document.getElementById("btn-close-gsv-modal");
  const cancelBtn = document.getElementById("btn-cancel-gsv");
  const saveBtn = document.getElementById("btn-save-gsv");
  const keyInput = document.getElementById("gsv-key-input");
  const statusEl = document.getElementById("gsv-key-status");

  if (!modal || !openBtn) return;

  openBtn.addEventListener("click", () => {
    modal.classList.add("active");
    if (statusEl) statusEl.textContent = "";
  });

  const closeModal = () => modal.classList.remove("active");
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeModal);

  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const keyVal = keyInput.value.trim();
      try {
        const res = await fetch("/config/gsv_key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: keyVal })
        });
        const data = await res.json();
        if (data.status === "success") {
          if (statusEl) statusEl.textContent = "KEY STORED IN ENVIRONMENT.";
          updateStatus(data.message);
          setTimeout(closeModal, 1000);
        }
      } catch (err) {
        if (statusEl) statusEl.textContent = `ERROR: ${err.message}`;
      }
    });
  }
}

// ----------------- Simulation & Decay ----------------- //

async function simulateDecay(days = 14.0) {
  updateStatus(`ADVANCING SIMULATION CLOCK (+${days} DAYS)...`);
  try {
    const res = await fetch("/demo/simulate_decay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ days: days })
    });
    const data = await res.json();
    updateStatus(`DECAY SIMULATED: ${data.message}`);
    await loadSegments();
    await updateDemoStatus();
    if (selectedEdgeId) {
      selectSegment(selectedEdgeId);
    }
  } catch (err) {
    updateStatus(`DECAY SIMULATION FAILED: ${err.message}`, true);
  }
}

async function resetDatabase() {
  if (!confirm("Reset database to initial baseline? All ingested observations will be cleared.")) return;
  updateStatus("RESETTING ROAD DATABASE...");
  try {
    const res = await fetch("/demo/reset", { method: "POST" });
    const data = await res.json();
    updateStatus(data.message);
    closeInspector();
    await loadSegments();
    await updateDemoStatus();
  } catch (err) {
    updateStatus(`RESET FAILED: ${err.message}`, true);
  }
}

async function updateDemoStatus() {
  try {
    const res = await fetch("/demo/status");
    if (!res.ok) return;
    const data = await res.json();
    if (data.simulated_now) {
      const d = new Date(data.simulated_now);
      document.getElementById("simulated-clock-val").textContent = d.toISOString().split("T")[0];
    }
  } catch (e) {
    console.warn("Status update error:", e);
  }
}

function applyFilter(filterVal) {
  currentLayerFilter = filterVal;
  loadSegments(filterVal);
}
