const API_BASE = 'http://localhost:5000/api';

// Estado
const appState = {
  mode: null, // 'single' o 'dual'
  models: { ml_models: [], dl_models: [] },
  dualConfig: null,
  signals: [],
  selectedModel: null,
  selectedModelType: null,
  selectedSignals: []
};

// DOM
const elements = {
  classificationMode: document.getElementById('classificationMode'),
  dualInfo: document.getElementById('dualInfo'),
  dualConfigText: document.getElementById('dualConfigText'),
  modelPanel: document.getElementById('modelPanel'),
  modelType: document.getElementById('modelType'),
  modelSelect: document.getElementById('modelSelect'),
  signalList: document.getElementById('signalList'),
  btnRefreshSignals: document.getElementById('btnRefreshSignals'),
  btnClassify: document.getElementById('btnClassify'),
  resultsContainer: document.getElementById('resultsContainer'),
  fileInput: document.getElementById('fileInput'),
  btnUpload: document.getElementById('btnUpload')
};

// Init
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
  loadDualConfig();
  loadSignals();
  setupEventListeners();
});

// Listeners
function setupEventListeners() {
  elements.classificationMode.addEventListener('change', handleModeChange);
  elements.modelType.addEventListener('change', handleModelTypeChange);
  elements.modelSelect.addEventListener('change', handleModelSelection);
  elements.btnRefreshSignals.addEventListener('click', loadSignals);
  elements.btnClassify.addEventListener('click', classifySignals);
  elements.btnUpload.addEventListener('click', uploadSelectedFiles);
}

// Modo de Clasificación
function handleModeChange(e) {
  const mode = e.target.value;
  appState.mode = mode;

  if (mode === 'dual') {
    // Modo dual: ocultar/deshabilitar selección de modelos
    elements.modelPanel.style.opacity = '0.5';
    elements.modelType.disabled = true;
    elements.modelSelect.disabled = true;

    // Mostrar info del sistema dual
    if (appState.dualConfig && appState.dualConfig.available) {
      elements.dualInfo.style.display = 'block';
      elements.dualConfigText.innerHTML = `
        <br>• Especialista SAFE: <strong>${appState.dualConfig.safe_specialist.name}</strong> 
        (precision ${(appState.dualConfig.safe_specialist.precision * 100).toFixed(1)}%)
        <br>• Especialista RISK: <strong>${appState.dualConfig.risk_specialist.name}</strong> 
        (recall ${(appState.dualConfig.risk_specialist.recall * 100).toFixed(1)}%)
        <br><small>Metadata: ${appState.dualConfig.metadata_date}</small>
      `;
    } else {
      elements.dualInfo.style.display = 'block';
      elements.dualConfigText.innerHTML = '<br>⚠️ Sistema dual no disponible. Se requiere metadata.json.';
    }
  } else if (mode === 'single') {
    // Modo independiente: habilitar selección de modelos
    elements.modelPanel.style.opacity = '1';
    elements.modelType.disabled = false;
    elements.dualInfo.style.display = 'none';
  } else {
    // Sin selección
    elements.modelPanel.style.opacity = '1';
    elements.modelType.disabled = true;
    elements.modelSelect.disabled = true;
    elements.dualInfo.style.display = 'none';
  }

  updateClassifyButton();
}

// Modelos
async function loadModels() {
  try {
    const response = await fetch(`${API_BASE}/models`);
    const data = await response.json();
    appState.models = data;
  } catch (err) {
    showError('No se pudieron cargar los modelos disponibles');
    console.error(err);
  }
}

async function loadDualConfig() {
  try {
    const response = await fetch(`${API_BASE}/dual-config`);
    const data = await response.json();
    appState.dualConfig = data;
  } catch (err) {
    console.error('Error cargando configuración dual:', err);
  }
}

function handleModelTypeChange(e) {
  const type = e.target.value;
  appState.selectedModelType = type;
  elements.modelSelect.disabled = !type;
  elements.modelSelect.innerHTML = '<option value="">-- Seleccionar modelo --</option>';

  if (type === 'ml') {
    appState.models.ml_models.forEach(model => {
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = model;
      elements.modelSelect.appendChild(opt);
    });
  } else if (type === 'dl') {
    appState.models.dl_models.forEach(model => {
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = model;
      elements.modelSelect.appendChild(opt);
    });
  }
  updateClassifyButton();
}

function handleModelSelection(e) {
  appState.selectedModel = e.target.value;
  updateClassifyButton();
}

// Señales
async function loadSignals() {
  elements.signalList.innerHTML = '<div class="loading"><div class="spinner"></div><p>Cargando señales...</p></div>';
  try {
    const response = await fetch(`${API_BASE}/signals`);
    const data = await response.json();
    appState.signals = data.signals || [];
    if (appState.signals.length === 0) {
      elements.signalList.innerHTML = '<div class="error">No hay señales disponibles</div>';
    } else {
      displaySignals();
    }
  } catch (err) {
    elements.signalList.innerHTML = '<div class="error">Error al cargar señales</div>';
    console.error(err);
  }
}

function displaySignals() {
  elements.signalList.innerHTML = '';
  appState.selectedSignals = [];
  appState.signals.forEach(signal => {
    const item = document.createElement('div');
    item.className = 'signal-item';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = signal;
    checkbox.id = `signal-${signal}`;

    const label = document.createElement('label');
    label.htmlFor = `signal-${signal}`;
    label.textContent = signal;
    label.style.cursor = 'pointer';
    label.style.flex = '1';
    label.style.margin = '0';

    item.appendChild(checkbox);
    item.appendChild(label);

    item.addEventListener('click', (e) => {
      if (e.target !== checkbox) checkbox.checked = !checkbox.checked;
      toggleSignalSelection(signal, checkbox.checked);
      item.classList.toggle('selected', checkbox.checked);
    });

    elements.signalList.appendChild(item);
  });
  updateClassifyButton();
}

function toggleSignalSelection(signal, isSelected) {
  if (isSelected) {
    if (!appState.selectedSignals.includes(signal)) appState.selectedSignals.push(signal);
  } else {
    appState.selectedSignals = appState.selectedSignals.filter(s => s !== signal);
  }
  updateClassifyButton();
}

// Subir archivos
async function uploadSelectedFiles() {
  const files = elements.fileInput.files;
  if (!files || files.length === 0) {
    alert('Selecciona uno o más .mat primero.');
    return;
  }

  const form = new FormData();
  for (const f of files) form.append('files', f);

  try {
    elements.btnUpload.disabled = true;
    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
    const data = await res.json();

    if (data.errors && data.errors.length) {
      alert('Errores al subir:\n' + data.errors.join('\n'));
    }
    if (data.uploaded && data.uploaded.length) {
      await loadSignals();
      data.uploaded.forEach(name => toggleSignalSelection(name, true));
    }
  } catch (err) {
    alert('Fallo la subida.');
    console.error(err);
  } finally {
    elements.btnUpload.disabled = false;
    elements.fileInput.value = '';
  }
}

// Clasificar
function updateClassifyButton() {
  let canClassify = false;

  if (appState.mode === 'dual') {
    // Modo dual: solo necesita señales y que el sistema esté disponible
    canClassify = appState.selectedSignals.length > 0 &&
      appState.dualConfig &&
      appState.dualConfig.available;
  } else if (appState.mode === 'single') {
    // Modo independiente: necesita modelo y señales
    canClassify = appState.selectedModel &&
      appState.selectedModelType &&
      appState.selectedSignals.length > 0;
  }

  elements.btnClassify.disabled = !canClassify;
}

async function classifySignals() {
  elements.btnClassify.disabled = true;
  elements.resultsContainer.innerHTML =
    '<div class="loading"><div class="spinner"></div><p>Clasificando señales...</p></div>';

  try {
    const requestBody = {
      mode: appState.mode,
      signal_files: appState.selectedSignals
    };

    // Solo agregar model_name y model_type en modo single
    if (appState.mode === 'single') {
      requestBody.model_name = appState.selectedModel;
      requestBody.model_type = appState.selectedModelType;
    }

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) throw new Error(`Error ${response.status}: ${response.statusText}`);
    const data = await response.json();
    displayResults(data);
  } catch (err) {
    elements.resultsContainer.innerHTML = `<div class="error">❌ Error: ${err.message}</div>`;
    console.error(err);
  } finally {
    elements.btnClassify.disabled = false;
  }
}

// Render resultados
function displayResults(data) {
  const { mode, results } = data;

  let html = `<div class="panel"><h2 class="panel-title">📈 Resultados de Clasificación</h2>`;

  if (mode === 'dual') {
    const config = data.config;
    html += `
      <p><strong>Modo:</strong> Sistema Dual 🔥</p>
      <p><strong>Especialista SAFE:</strong> ${config.safe_specialist.name}</p>
      <p><strong>Especialista RISK:</strong> ${config.risk_specialist.name}</p>
      <p><strong>Señales procesadas:</strong> ${results.length}</p>
    `;
  } else {
    html += `
      <p><strong>Modo:</strong> Modelo Independiente</p>
      <p><strong>Modelo utilizado:</strong> ${data.model} (${data.model_type.toUpperCase()})</p>
      <p><strong>Señales procesadas:</strong> ${results.length}</p>
    `;
  }

  html += `</div><div class="results-grid">`;

  results.forEach((r, idx) => {
    if (r.error) {
      html += `
        <div class="result-card" style="background:#ff6b6b;color:white;">
          <h3>${r.signal}</h3>
          <p>❌ Error: ${r.error}</p>
        </div>
      `;
      return;
    }

    const cardClass = r.is_risk ? 'risk' : 'safe';
    const icon = r.is_risk ? '⚠️' : '✅';
    const alertMessage = r.is_risk ? '¡ALERTA! Movimiento de riesgo detectado' : 'Movimiento seguro';

    html += `
      <div class="result-card ${cardClass}">
        <div class="result-header">
          <h3 style="font-size:.9em;margin:0;">${r.signal}</h3>
          <span class="risk-badge ${cardClass}">${r.risk_label}</span>
        </div>

        <div class="alert-icon">${icon}</div>
        <p style="font-size:1.2em;font-weight:bold;margin:10px 0;">${alertMessage}</p>
    `;

    // Sistema Dual: mostrar confianza y detalles
    if (mode === 'dual' && r.confidence) {
      html += `
        <div style="margin:10px 0;">
          <strong>Confianza:</strong>
          <span class="confidence-badge confidence-${r.confidence}">${r.confidence.toUpperCase()}</span>
        </div>
        <div style="font-size:0.9em;margin-top:10px;background:rgba(255,255,255,.2);padding:10px;border-radius:5px;">
          <strong>Decisión tomada por:</strong> ${r.decision_by}
        </div>
      `;

      // Detalles de cada modelo
      if (r.dual_details) {
        html += `<div style="margin-top:15px;font-size:0.85em;">`;

        for (const [modelName, details] of Object.entries(r.dual_details)) {
          if (details && details.prediction !== null) {
            const predLabel = details.prediction === 1 ? 'RISK' : 'SAFE';
            html += `
              <div style="background:rgba(255,255,255,.15);padding:8px;margin:5px 0;border-radius:5px;">
                <strong>${modelName}:</strong> ${predLabel}
                ${details.probability ?
                ` (Risk: ${(details.probability[1] * 100).toFixed(1)}%)` : ''}
              </div>
            `;
          }
        }

        html += `</div>`;
      }
    }

    // Estadísticas de ventanas
    if (r.n_windows) {
      const riskPct = mode === 'dual' ?
        (r.dual_details && r.dual_details[Object.keys(r.dual_details)[0]]) ?
          ((r.dual_details[Object.keys(r.dual_details)[0]].risk_windows / r.n_windows) * 100).toFixed(1) : 0
        : r.risk_percentage ? r.risk_percentage.toFixed(1) : 0;

      html += `
        <div style="background:rgba(255,255,255,.2);padding:10px;border-radius:8px;margin:10px 0;font-size:.9em;">
          <strong>📊 Ventanas analizadas:</strong><br>
          Total: ${r.n_windows}
        </div>
      `;
    }

    // Probabilidades (solo modo single)
    if (mode === 'single' && r.probability) {
      html += `
        <div style="margin:15px 0;">
          <strong>Probabilidades (promedio):</strong><br>
          <div style="display:flex;gap:10px;margin-top:5px;">
            <div style="flex:1;background:rgba(255,255,255,.3);padding:5px;border-radius:5px;">
              Seguro: ${(r.probability[0] * 100).toFixed(1)}%
            </div>
            <div style="flex:1;background:rgba(255,255,255,.3);padding:5px;border-radius:5px;">
              Riesgo: ${(r.probability[1] * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      `;
    }

    // Metadata
    if (r.metadata && (r.metadata.subject || r.metadata.movement || r.metadata.repetition)) {
      html += `
        <div class="metadata">
          ${r.metadata.subject !== null ? `<div><strong>Sujeto:</strong> ${r.metadata.subject}</div>` : ''}
          ${r.metadata.movement !== null ? `<div><strong>Movimiento:</strong> ${r.metadata.movement}</div>` : ''}
          ${r.metadata.repetition !== null ? `<div><strong>Repetición:</strong> ${r.metadata.repetition}</div>` : ''}
        </div>
      `;
    }

    html += `<canvas id="chart-${idx}" width="400" height="200"></canvas></div>`;
  });

  html += '</div>';
  elements.resultsContainer.innerHTML = html;

  // Dibujar señales
  results.forEach((r, i) => {
    if (r.signal_data) drawSignalChart(`chart-${i}`, r.signal_data);
  });
}

function drawSignalChart(canvasId, signalData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width, height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const padding = 40;
  const plotW = width - 2 * padding;
  const plotH = height - 2 * padding;

  const data = Array.isArray(signalData) ? signalData : [signalData];
  const n = data.length;
  const firstChannel = data.map(row => Array.isArray(row) ? row[0] : row);
  const minVal = Math.min(...firstChannel);
  const maxVal = Math.max(...firstChannel);
  const range = (maxVal - minVal) || 1;

  // Ejes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.stroke();

  // Label
  ctx.fillStyle = '#333';
  ctx.font = '12px Arial';
  ctx.fillText('EMG canal 1 (submuestreado)', width / 2 - 60, height - 10);

  // Señal
  ctx.strokeStyle = '#667eea';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = padding + (i / (n - 1)) * plotW;
    const y = height - padding - ((firstChannel[i] - minVal) / range) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function showError(message) {
  elements.resultsContainer.innerHTML = `<div class="error">❌ ${message}</div>`;
}
