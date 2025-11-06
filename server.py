"""
Backend Flask para clasificación binaria de movimientos de riesgo STC
Proyecto: Herramienta para clasificación de movimientos asociados al STC
Autor: Karen Nicolle Arango Valencia
Universidad: Pontificia Universidad Javeriana - Cali

VERSIÓN ACTUALIZADA: Compatible 100% con el notebook de Colab
Pipeline: Butterworth + Notch → Z-score → Ventanas → Features/Secuencias → Predicción
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import numpy as np
import pickle
from scipy.io import loadmat
from scipy import signal as scipy_signal
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import pywt
import os

app = Flask(__name__, static_folder=None)
CORS(app)

# ============================================================================
# CONFIGURACIÓN - IDÉNTICA AL COLAB
# ============================================================================

class Config:
    # Parámetros de señal
    FS = 2000  # Hz
    WINDOW_SIZE_MS = 300  # ms
    WINDOW_SIZE = int(FS * WINDOW_SIZE_MS / 1000)  # 600 samples
    OVERLAP = 0.5
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 300 samples
    N_CHANNELS = 12
    
    # Filtrado
    LOWCUT = 20  # Hz
    HIGHCUT = 450  # Hz
    NOTCH_FREQ = 50  # Hz
    NOTCH_Q = 30
    
    # Clasificación binaria
    RISK_MOVEMENTS = [13, 14, 15, 16]
    SAFE_MOVEMENTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]

cfg = Config()

# ============================================================================
# RUTAS
# ============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
EXTERNAL_SIGNALS_DIR = Path(r'C:\Users\karen\Desktop\ninaproV5\DB2_E1_only\test')
UPLOADS_DIR = BASE_DIR / 'signals'
FRONTEND_DIR = BASE_DIR / 'frontend'

ALLOWED_EXT = {'.mat'}

# ============================================================================
# PREPROCESAMIENTO - IDÉNTICO AL COLAB
# ============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filtro pasa-banda Butterworth"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data, axis=0)

def notch_filter(data, freq, fs, Q=30):
    """Filtro notch para eliminar ruido de línea"""
    b, a = scipy_signal.iirnotch(freq, Q, fs)
    return scipy_signal.filtfilt(b, a, data, axis=0)

def apply_filters(emg_data):
    """Aplica filtrado Butterworth + Notch (igual que en Colab)"""
    filtered = butter_bandpass_filter(emg_data, cfg.LOWCUT, cfg.HIGHCUT, cfg.FS)
    filtered = notch_filter(filtered, cfg.NOTCH_FREQ, cfg.FS, cfg.NOTCH_Q)
    return filtered

def create_windows(emg, window_size, step_size):
    """Crea ventanas deslizantes"""
    n_samples = emg.shape[0]
    windows = []
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = emg[start:end]
        windows.append(window)
    
    return np.array(windows, dtype=np.float32)

# ============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS - 144 FEATURES COMPLETAS
# ============================================================================

def extract_time_domain_features(window):
    """Features en dominio del tiempo (MAV, WL, ZC, SSC, RMS)"""
    features = []
    
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        
        # MAV (Mean Absolute Value)
        mav = np.mean(np.abs(signal))
        
        # WL (Waveform Length)
        wl = np.sum(np.abs(np.diff(signal)))
        
        # ZC (Zero Crossing) - con threshold
        zc = np.sum(np.diff(np.sign(signal)) != 0)
        
        # SSC (Slope Sign Change)
        diff_signal = np.diff(signal)
        ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
        
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(signal**2))
        
        features.extend([mav, wl, zc, ssc, rms])
    
    return np.array(features)

def extract_frequency_features(window, fs=2000):
    """Features en dominio de frecuencia (Mean Freq, Median Freq)"""
    features = []
    
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        
        # FFT
        fft = np.fft.rfft(signal)
        psd = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        
        # Mean frequency
        mean_freq = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        
        # Median frequency
        cumsum_psd = np.cumsum(psd)
        median_idx = np.where(cumsum_psd >= cumsum_psd[-1]/2)[0]
        if len(median_idx) > 0:
            median_freq = freqs[median_idx[0]]
        else:
            median_freq = 0.0
        
        features.extend([mean_freq, median_freq])
    
    return np.array(features)

def extract_wavelet_features(window, wavelet='db4', level=4):
    """Features wavelet (energía por nivel) - IDÉNTICO AL COLAB"""
    features = []
    
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        
        # DWT
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Energía de cada nivel (5 niveles: cA4, cD4, cD3, cD2, cD1)
        for coeff in coeffs:
            energy = np.sum(coeff**2)
            features.append(energy)
    
    return np.array(features)

def extract_all_features(window):
    """
    Combina todas las features - 144 TOTALES
    - Tiempo: 5 features × 12 canales = 60
    - Frecuencia: 2 features × 12 canales = 24
    - Wavelet: 5 niveles × 12 canales = 60
    Total: 144 features
    """
    td = extract_time_domain_features(window)
    fd = extract_frequency_features(window)
    wt = extract_wavelet_features(window)
    return np.concatenate([td, fd, wt])

# ============================================================================
# PIPELINE COMPLETO PARA ML
# ============================================================================

def preprocess_signal_ml(emg_data, scaler):
    """
    Pipeline completo para ML (IDÉNTICO AL COLAB):
    1. Filtrado (Butterworth + Notch)
    2. Normalización Z-score de TODA la señal
    3. Crear ventanas
    4. Extraer 144 features por ventana
    5. Retornar features (ya normalizadas por el scaler global)
    """
    # 1. Filtrado
    emg_filtered = apply_filters(emg_data)
    
    # 2. Normalización Z-score (ANTES de ventanas)
    emg_normalized = scaler.transform(emg_filtered)
    
    # 3. Ventanas
    windows = create_windows(emg_normalized, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    
    if len(windows) == 0:
        return np.array([]), windows
    
    # 4. Extraer features
    features_list = []
    for window in windows:
        features = extract_all_features(window)
        features_list.append(features)
    
    features = np.array(features_list, dtype=np.float32)
    
    return features, windows

# ============================================================================
# PIPELINE COMPLETO PARA DL
# ============================================================================

def preprocess_signal_dl(emg_data, scaler):
    """
    Pipeline completo para DL (IDÉNTICO AL COLAB):
    1. Filtrado (Butterworth + Notch)
    2. Normalización Z-score de TODA la señal
    3. Crear ventanas
    4. Retornar ventanas en formato (n_windows, timesteps, channels)
    """
    # 1. Filtrado
    emg_filtered = apply_filters(emg_data)
    
    # 2. Normalización Z-score (ANTES de ventanas)
    emg_normalized = scaler.transform(emg_filtered)
    
    # 3. Ventanas
    windows = create_windows(emg_normalized, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    
    # Shape: (n_windows, 600, 12)
    return windows

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Devuelve los modelos disponibles
    Nombres EXACTOS del Colab:
    - ML: model_ensemble_knn.pkl, model_svm_rbf.pkl
    - DL: model_cnn_lstm.h5, model_bilstm.h5
    """
    ml_models = []
    dl_models = []
    
    if MODELS_DIR.exists():
        # ML: buscar .pkl (excepto scaler.pkl)
        for pkl in MODELS_DIR.glob('*.pkl'):
            if pkl.stem not in ['scaler', 'config']:
                ml_models.append(pkl.stem)
        
        # DL: buscar .h5 y .keras
        for h5 in MODELS_DIR.glob('*.h5'):
            dl_models.append(h5.stem)
        for keras_file in MODELS_DIR.glob('*.keras'):
            if keras_file.stem not in dl_models:
                dl_models.append(keras_file.stem)
    
    return jsonify({
        'ml_models': sorted(ml_models),
        'dl_models': sorted(dl_models)
    })

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Lista señales disponibles en carpeta externa y uploads"""
    signals = []
    
    # Carpeta externa
    if EXTERNAL_SIGNALS_DIR.exists():
        for mat_file in EXTERNAL_SIGNALS_DIR.glob('*.mat'):
            signals.append(mat_file.name)
    
    # Uploads
    if UPLOADS_DIR.exists():
        for mat_file in UPLOADS_DIR.glob('*.mat'):
            if mat_file.name not in signals:
                signals.append(mat_file.name)
    
    return jsonify({'signals': sorted(signals)})

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Sube archivos .mat a la carpeta signals/"""
    if 'files' not in request.files:
        return jsonify({'error': 'No se enviaron archivos'}), 400
    
    files = request.files.getlist('files')
    uploaded = []
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
        
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            errors.append(f'{file.filename}: Solo se permiten archivos .mat')
            continue
        
        filename = secure_filename(file.filename)
        filepath = UPLOADS_DIR / filename
        
        try:
            file.save(str(filepath))
            uploaded.append(filename)
        except Exception as e:
            errors.append(f'{filename}: {str(e)}')
    
    return jsonify({'uploaded': uploaded, 'errors': errors})

def _resolve_signal_path(filename: str):
    """Busca el archivo primero en uploads, luego en carpeta externa"""
    p1 = UPLOADS_DIR / filename
    if p1.exists():
        return p1
    p2 = EXTERNAL_SIGNALS_DIR / filename
    if p2.exists():
        return p2
    return None

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Clasificación de señales
    
    Request JSON:
    {
      "model_name": "model_ensemble_knn" | "model_cnn_lstm",
      "model_type": "ml" | "dl",
      "signal_files": ["S1_E1_A1.mat", ...]
    }
    
    Response JSON:
    {
      "model": "model_ensemble_knn",
      "model_type": "ml",
      "results": [
        {
          "signal": "S1_E1_A1.mat",
          "prediction": 0,  // 0=Safe, 1=Risk
          "risk_label": "SEGURO",
          "is_risk": false,
          "probability": [0.85, 0.15],  // [Safe, Risk]
          "n_windows": 100,
          "risk_windows": 15,
          "safe_windows": 85,
          "risk_percentage": 15.0,
          "metadata": {"subject": 1, "movement": 5, "repetition": 3},
          "signal_data": [[...], [...]]  // Submuestreada para gráfico
        }
      ]
    }
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        model_type = data.get('model_type')
        signal_files = data.get('signal_files', [])
        
        if not model_name or not model_type or not signal_files:
            return jsonify({'error': 'Faltan parámetros requeridos'}), 400
        
        # ============================================================================
        # CARGAR MODELO Y SCALER
        # ============================================================================
        
        if model_type == 'ml':
            # Modelo ML
            model_path = MODELS_DIR / f'{model_name}.pkl'
            if not model_path.exists():
                return jsonify({'error': f'Modelo {model_name}.pkl no encontrado'}), 404
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Scaler (OBLIGATORIO)
            scaler_path = MODELS_DIR / 'scaler.pkl'
            if not scaler_path.exists():
                return jsonify({'error': 'scaler.pkl no encontrado en models/'}), 404
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        elif model_type == 'dl':
            # Modelo DL
            model_path = MODELS_DIR / f'{model_name}.h5'
            if not model_path.exists():
                model_path = MODELS_DIR / f'{model_name}.keras'
            if not model_path.exists():
                return jsonify({'error': f'Modelo {model_name}.h5/.keras no encontrado'}), 404
            
            model = keras.models.load_model(str(model_path), compile=False)
            
            # Scaler (OBLIGATORIO también para DL)
            scaler_path = MODELS_DIR / 'scaler.pkl'
            if not scaler_path.exists():
                return jsonify({'error': 'scaler.pkl no encontrado en models/'}), 404
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        else:
            return jsonify({'error': 'Tipo de modelo inválido (debe ser ml o dl)'}), 400
        
        # ============================================================================
        # PROCESAR SEÑALES
        # ============================================================================
        
        results = []
        
        for signal_file in signal_files:
            # Buscar archivo
            signal_path = _resolve_signal_path(signal_file)
            if signal_path is None:
                results.append({
                    'signal': signal_file,
                    'error': 'Archivo no encontrado'
                })
                continue
            
            # Cargar .mat
            try:
                mat_data = loadmat(str(signal_path))
            except Exception as e:
                results.append({
                    'signal': signal_file,
                    'error': f'Error al cargar .mat: {str(e)}'
                })
                continue
            
            # Extraer EMG (buscar 'emg' o primera matriz 2D)
            emg = None
            if 'emg' in mat_data:
                emg = mat_data['emg']
            else:
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                        emg = value
                        break
            
            if emg is None:
                results.append({
                    'signal': signal_file,
                    'error': 'No se encontró matriz EMG 2D en el archivo'
                })
                continue
            
            # Validar forma
            if emg.ndim != 2:
                results.append({
                    'signal': signal_file,
                    'error': f'EMG debe ser 2D, recibido: {emg.shape}'
                })
                continue
            
            # Extraer metadata
            def _get_scalar(key):
                if key in mat_data:
                    try:
                        return int(np.array(mat_data[key]).flatten()[0])
                    except:
                        return None
                return None
            
            metadata = {
                'subject': _get_scalar('subject'),
                'movement': _get_scalar('restimulus') or _get_scalar('stimulus'),
                'repetition': _get_scalar('rerepetition') or _get_scalar('repetition')
            }
            
            # ============================================================================
            # PREDICCIÓN
            # ============================================================================
            
            try:
                if model_type == 'ml':
                    # Pipeline ML
                    features, windows = preprocess_signal_ml(emg, scaler)
                    
                    if len(features) == 0:
                        results.append({
                            'signal': signal_file,
                            'error': 'No se generaron ventanas válidas'
                        })
                        continue
                    
                    # Predicción
                    predictions = model.predict(features)
                    
                    # Probabilidades (si disponible)
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(features)
                        avg_proba = np.mean(probas, axis=0).tolist()
                    else:
                        avg_proba = None
                    
                    # Agregación: promedio de predicciones
                    final_pred = int(np.round(np.mean(predictions)))
                    
                    # Estadísticas
                    n_windows = len(predictions)
                    risk_windows = int(np.sum(predictions == 1))
                    safe_windows = int(np.sum(predictions == 0))
                    risk_pct = 100.0 * risk_windows / n_windows if n_windows > 0 else 0.0
                
                else:  # DL
                    # Pipeline DL
                    windows = preprocess_signal_dl(emg, scaler)
                    
                    if len(windows) == 0:
                        results.append({
                            'signal': signal_file,
                            'error': 'No se generaron ventanas válidas'
                        })
                        continue
                    
                    # Predicción
                    probas = model.predict(windows, verbose=0)
                    predictions = (probas > 0.5).astype(int).flatten()
                    
                    # Probabilidades promedio
                    avg_proba_risk = float(np.mean(probas))
                    avg_proba = [1.0 - avg_proba_risk, avg_proba_risk]
                    
                    # Agregación: promedio de predicciones
                    final_pred = int(np.round(np.mean(predictions)))
                    
                    # Estadísticas
                    n_windows = len(predictions)
                    risk_windows = int(np.sum(predictions == 1))
                    safe_windows = int(np.sum(predictions == 0))
                    risk_pct = 100.0 * risk_windows / n_windows if n_windows > 0 else 0.0
                
                # ============================================================================
                # RESULTADO
                # ============================================================================
                
                is_risk = bool(final_pred == 1)
                
                results.append({
                    'signal': signal_file,
                    'metadata': metadata,
                    'prediction': final_pred,
                    'risk_label': 'RIESGO' if is_risk else 'SEGURO',
                    'is_risk': is_risk,
                    'probability': avg_proba,
                    'n_windows': n_windows,
                    'risk_windows': risk_windows,
                    'safe_windows': safe_windows,
                    'risk_percentage': float(risk_pct),
                    'signal_shape': list(emg.shape),
                    'signal_data': emg[::10, :].tolist()  # Submuestrear para frontend
                })
            
            except Exception as e:
                import traceback
                results.append({
                    'signal': signal_file,
                    'error': f'Error en predicción: {str(e)}',
                    'traceback': traceback.format_exc()
                })
        
        return jsonify({
            'model': model_name,
            'model_type': model_type,
            'results': results
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ============================================================================
# FRONTEND
# ============================================================================

@app.route('/')
def index():
    """Sirve el frontend"""
    index_path = FRONTEND_DIR / 'index.html'
    if index_path.exists():
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    return jsonify({'status': 'API activa', 'message': 'Frontend no encontrado'})

@app.route('/<path:path>')
def serve_static(path):
    """Sirve archivos estáticos del frontend"""
    file_path = FRONTEND_DIR / path
    if file_path.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return jsonify({'error': 'Recurso no encontrado'}), 404

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Crear directorios
    MODELS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    FRONTEND_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 SERVIDOR CLASIFICADOR STC")
    print("="*70)
    print(f"\n📂 Rutas configuradas:")
    print(f"   • Modelos:   {MODELS_DIR}")
    print(f"   • Uploads:   {UPLOADS_DIR}")
    print(f"   • Externas:  {EXTERNAL_SIGNALS_DIR}")
    print(f"   • Frontend:  {FRONTEND_DIR}")
    
    print(f"\n⚙️  Configuración del pipeline:")
    print(f"   • Frecuencia:      {cfg.FS} Hz")
    print(f"   • Ventanas:        {cfg.WINDOW_SIZE} samples ({cfg.WINDOW_SIZE_MS}ms)")
    print(f"   • Overlap:         {cfg.OVERLAP*100:.0f}%")
    print(f"   • Filtrado:        Butterworth ({cfg.LOWCUT}-{cfg.HIGHCUT} Hz) + Notch ({cfg.NOTCH_FREQ} Hz)")
    print(f"   • Features ML:     144 (tiempo + frecuencia + wavelet)")
    print(f"   • Normalización:   Z-score (StandardScaler)")
    
    print(f"\n🎯 Clasificación:")
    print(f"   • Risk:  {cfg.RISK_MOVEMENTS}")
    print(f"   • Safe:  {cfg.SAFE_MOVEMENTS}")
    
    print(f"\n✅ Pipeline 100% compatible con notebook de Colab")
    print(f"\n🌐 Servidor escuchando en http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)