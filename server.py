"""
Backend Flask para clasificación binaria de movimientos de riesgo STC
Proyecto: Herramienta para clasificación de movimientos asociados al STC
Autor: Karen Nicolle Arango Valencia
Universidad: Pontificia Universidad Javeriana - Cali

VERSIÓN CON SISTEMA DUAL:
- Modo Independiente: Un solo modelo (ML o DL)
- Modo Dual: RandomForest + Ensemble_KNN en cascada
Pipeline: Butterworth + Notch → Z-score → Ventanas → Features/Secuencias → Predicción
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import numpy as np
import pickle
import json
from scipy.io import loadmat
from scipy import signal as scipy_signal
from werkzeug.utils import secure_filename
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
import pywt
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
keras.config.enable_unsafe_deserialization()
print("⚙️  Unsafe deserialization habilitada para Lambda layers")

app = Flask(__name__, static_folder=None)
CORS(app)

# ============================================================================
# CONFIGURACIÓN - IDÉNTICA AL COLAB
# ============================================================================

class Config:
    # Parámetros de señal
    FS = 2000  # Hz
    WINDOW_SIZE_MS = 500  # ms
    WINDOW_SIZE = int(FS * WINDOW_SIZE_MS / 1000)
    OVERLAP = 0.25
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
    N_CHANNELS = 12
    
    # Filtrado
    LOWCUT = 20  # Hz
    HIGHCUT = 450  # Hz
    NOTCH_FREQ = 50  # Hz
    NOTCH_Q = 30
    
    # Clasificación binaria
    RISK_MOVEMENTS = [13, 14, 15, 16]
    SAFE_MOVEMENTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]

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

# Variable global para cachear configuración del sistema dual
DUAL_SYSTEM_CONFIG = None
OPTIMIZED_THRESHOLDS = None 

# ============================================================================
# PREPROCESAMIENTO - IDÉNTICO AL COLAB
# ============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data, axis=0)

def notch_filter(data, freq, fs, Q=30):
    b, a = scipy_signal.iirnotch(freq, Q, fs)
    return scipy_signal.filtfilt(b, a, data, axis=0)

def apply_filters(emg_data):
    filtered = butter_bandpass_filter(emg_data, cfg.LOWCUT, cfg.HIGHCUT, cfg.FS)
    filtered = notch_filter(filtered, cfg.NOTCH_FREQ, cfg.FS, cfg.NOTCH_Q)
    return filtered

def create_windows(emg, window_size, step_size):
    n_samples = emg.shape[0]
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = emg[start:end]
        windows.append(window)
    return np.array(windows, dtype=np.float32)

# ============================================================================
# CUSTOM OBJECTS PARA MODELOS DL
# ============================================================================

def attention_sum(x):
    """Función nombrada para reemplazar lambda en atención"""
    return K.sum(x, axis=1)

# Custom objects para deserializar modelos
CUSTOM_OBJECTS = {
    'attention_sum': attention_sum
}

# ============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================================

def extract_time_domain_features(window):
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        mav = np.mean(np.abs(signal))
        wl = np.sum(np.abs(np.diff(signal)))
        zc = np.sum(np.diff(np.sign(signal)) != 0)
        diff_signal = np.diff(signal)
        ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
        rms = np.sqrt(np.mean(signal**2))
        features.extend([mav, wl, zc, ssc, rms])
    return np.array(features)

def extract_frequency_features(window, fs=2000):
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        fft = np.fft.rfft(signal)
        psd = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        total_power = np.sum(psd)
        if total_power > 0:
            mean_freq = np.sum(freqs * psd) / total_power
        else:
            mean_freq = 0.0
        cumsum_psd = np.cumsum(psd)
        median_idx = np.where(cumsum_psd >= cumsum_psd[-1]/2)[0]
        if len(median_idx) > 0:
            median_freq = freqs[median_idx[0]]
        else:
            median_freq = 0.0
        features.extend([mean_freq, median_freq])
    return np.array(features)

def extract_wavelet_features(window, wavelet='db4', level=4):
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        for coeff in coeffs:
            energy = np.sum(coeff**2)
            features.append(energy)
    return np.array(features)

def extract_all_features(window):
    td = extract_time_domain_features(window)
    fd = extract_frequency_features(window)
    wt = extract_wavelet_features(window)
    return np.concatenate([td, fd, wt])

# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def preprocess_signal_ml(emg_data, scaler):
    emg_filtered = apply_filters(emg_data)
    emg_normalized = scaler.transform(emg_filtered)
    windows = create_windows(emg_normalized, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    
    if len(windows) == 0:
        return np.array([]), windows
    
    features_list = []
    for window in windows:
        features = extract_all_features(window)
        features_list.append(features)
    
    features = np.array(features_list, dtype=np.float32)
    return features, windows

def preprocess_signal_dl(emg_data, scaler):
    emg_filtered = apply_filters(emg_data)
    emg_normalized = scaler.transform(emg_filtered)
    windows = create_windows(emg_normalized, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    return windows

# ============================================================================
# SISTEMA DUAL - CONFIGURACIÓN
# ============================================================================

def load_dual_system_config():
    """
    Lee metadata.json y determina los mejores modelos
    Especialista RISK: Mayor recall
    Especialista SAFE: Mayor precision
    """
    global DUAL_SYSTEM_CONFIG
    
    if DUAL_SYSTEM_CONFIG is not None:
        return DUAL_SYSTEM_CONFIG
    
    # Buscar metadata.json más reciente
    metadata_files = list(MODELS_DIR.glob('**/metadata.json'))
    if not metadata_files:
        return None
    
    metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    metadata_path = metadata_files[0]
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        global OPTIMIZED_THRESHOLDS
        OPTIMIZED_THRESHOLDS = metadata.get('optimized_thresholds', {})
        results = metadata.get('results', {})

        if not results:
            return None
        
        # Filtrar solo modelos ML disponibles
        ml_models = {}
        for model_name, metrics in results.items():
            model_path = MODELS_DIR / f'{model_name}.pkl'
            if model_path.exists():
                ml_models[model_name] = metrics
        
        if len(ml_models) < 2:
            return None
        
        # Especialista en RISK (mayor recall)
        risk_specialist = max(ml_models.items(), key=lambda x: x[1].get('recall', 0))
        
        # Especialista en SAFE (mayor precision)
        safe_specialist = max(ml_models.items(), key=lambda x: x[1].get('precision', 0))
        
        DUAL_SYSTEM_CONFIG = {
            'available': True,
            'risk_specialist': {
                'name': risk_specialist[0],
                'recall': risk_specialist[1].get('recall', 0),
                'precision': risk_specialist[1].get('precision', 0),
                'f1': risk_specialist[1].get('f1', 0)
            },
            'safe_specialist': {
                'name': safe_specialist[0],
                'recall': safe_specialist[1].get('recall', 0),
                'precision': safe_specialist[1].get('precision', 0),
                'f1': safe_specialist[1].get('f1', 0)
            },
            'metadata_date': metadata.get('timestamp', 'unknown'),
            'metadata_path': str(metadata_path)
        }
        
        return DUAL_SYSTEM_CONFIG
        
    except Exception as e:
        print(f"Error loading dual system config: {e}")
        return None

def get_optimized_threshold(model_name):
    """Obtiene el threshold optimizado para un modelo DL"""
    global OPTIMIZED_THRESHOLDS
    
    if OPTIMIZED_THRESHOLDS is None:
        load_dual_system_config()
    
    if OPTIMIZED_THRESHOLDS and model_name in OPTIMIZED_THRESHOLDS:
        return OPTIMIZED_THRESHOLDS[model_name]
    
    return 0.5

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/models', methods=['GET'])
def get_models():
    ml_models = []
    dl_models = []
    
    if MODELS_DIR.exists():
        for pkl in MODELS_DIR.glob('*.pkl'):
            if pkl.stem not in ['scaler', 'config']:
                ml_models.append(pkl.stem)
        
        # Primero buscar .keras, luego .h5 (para evitar duplicados)
        for keras_file in MODELS_DIR.glob('*.keras'):
            dl_models.append(keras_file.stem)
        for h5 in MODELS_DIR.glob('*.h5'):
            if h5.stem not in dl_models:  # Solo agregar si no existe versión .keras
                dl_models.append(h5.stem)
    
    return jsonify({
        'ml_models': sorted(ml_models),
        'dl_models': sorted(dl_models)
    })

@app.route('/api/dual-config', methods=['GET'])
def get_dual_config():
    config = load_dual_system_config()
    
    if config is None:
        return jsonify({
            'available': False,
            'message': 'Sistema dual no disponible'
        })
    
    return jsonify(config)

@app.route('/api/signals', methods=['GET'])
def get_signals():
    signals = []
    
    if EXTERNAL_SIGNALS_DIR.exists():
        for mat_file in EXTERNAL_SIGNALS_DIR.glob('*.mat'):
            signals.append(mat_file.name)
    
    if UPLOADS_DIR.exists():
        for mat_file in UPLOADS_DIR.glob('*.mat'):
            if mat_file.name not in signals:
                signals.append(mat_file.name)
    
    return jsonify({'signals': sorted(signals)})

@app.route('/api/upload', methods=['POST'])
def upload_files():
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
    p1 = UPLOADS_DIR / filename
    if p1.exists():
        return p1
    p2 = EXTERNAL_SIGNALS_DIR / filename
    if p2.exists():
        return p2
    return None

@app.route('/api/predict', methods=['POST'])
def predict():
    """Clasificación: modo single o dual"""
    try:
        data = request.get_json()
        mode = data.get('mode', 'single')
        signal_files = data.get('signal_files', [])
        
        if not signal_files:
            return jsonify({'error': 'No se proporcionaron archivos'}), 400
        
        # MODO DUAL
        if mode == 'dual':
            dual_config = load_dual_system_config()
            if not dual_config or not dual_config.get('available'):
                return jsonify({'error': 'Sistema dual no disponible'}), 400
            
            risk_model_name = dual_config['risk_specialist']['name']
            safe_model_name = dual_config['safe_specialist']['name']
            
            scaler_path = MODELS_DIR / 'scaler.pkl'
            if not scaler_path.exists():
                return jsonify({'error': 'scaler.pkl no encontrado'}), 404
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            safe_model_path = MODELS_DIR / f'{safe_model_name}.pkl'
            if not safe_model_path.exists():
                return jsonify({'error': f'{safe_model_name} no encontrado'}), 404
            with open(safe_model_path, 'rb') as f:
                safe_model = pickle.load(f)
            
            risk_model_path = MODELS_DIR / f'{risk_model_name}.pkl'
            if not risk_model_path.exists():
                return jsonify({'error': f'{risk_model_name} no encontrado'}), 404
            with open(risk_model_path, 'rb') as f:
                risk_model = pickle.load(f)
            
            results = []
            
            for signal_file in signal_files:
                signal_path = _resolve_signal_path(signal_file)
                if not signal_path:
                    results.append({'signal': signal_file, 'error': 'Archivo no encontrado'})
                    continue
                
                try:
                    mat_data = loadmat(str(signal_path))
                except:
                    results.append({'signal': signal_file, 'error': 'Error al cargar .mat'})
                    continue
                
                emg = mat_data.get('emg')
                if emg is None:
                    for key, value in mat_data.items():
                        if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                            emg = value
                            break
                
                if emg is None or emg.ndim != 2:
                    results.append({'signal': signal_file, 'error': 'EMG inválido'})
                    continue
                
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
                
                try:
                    features, windows = preprocess_signal_ml(emg, scaler)
                    if len(features) == 0:
                        results.append({'signal': signal_file, 'error': 'Sin ventanas válidas'})
                        continue
                    
                    # LÓGICA DE CASCADA
                    safe_predictions = safe_model.predict(features)
                    safe_probas = safe_model.predict_proba(features) if hasattr(safe_model, 'predict_proba') else None
                    safe_decision = int(np.round(np.mean(safe_predictions)))
                    
                    if safe_decision == 1:
                        # RF dice RISK → confiar
                        final_pred = 1
                        confidence = 'alta'
                        decision_by = safe_model_name
                        risk_predictions = None
                        risk_probas = None
                    else:
                        # RF dice SAFE → segunda opinión KNN
                        risk_predictions = risk_model.predict(features)
                        risk_probas = risk_model.predict_proba(features) if hasattr(risk_model, 'predict_proba') else None
                        risk_decision = int(np.round(np.mean(risk_predictions)))
                        
                        if risk_decision == 1:
                            final_pred = 1
                            confidence = 'media'
                            decision_by = f'{safe_model_name}+{risk_model_name}'
                        else:
                            final_pred = 0
                            confidence = 'alta'
                            decision_by = f'{safe_model_name}+{risk_model_name}'
                    
                    n_windows = len(safe_predictions)
                    safe_risk_windows = int(np.sum(safe_predictions == 1))
                    safe_safe_windows = int(np.sum(safe_predictions == 0))
                    
                    if risk_predictions is not None:
                        risk_risk_windows = int(np.sum(risk_predictions == 1))
                        risk_safe_windows = int(np.sum(risk_predictions == 0))
                    else:
                        risk_risk_windows = None
                        risk_safe_windows = None
                    
                    safe_avg_proba = np.mean(safe_probas, axis=0).tolist() if safe_probas is not None else None
                    risk_avg_proba = np.mean(risk_probas, axis=0).tolist() if risk_probas is not None else None
                    
                    is_risk = bool(final_pred == 1)
                    
                    results.append({
                        'signal': signal_file,
                        'metadata': metadata,
                        'prediction': final_pred,
                        'risk_label': 'RIESGO' if is_risk else 'SEGURO',
                        'is_risk': is_risk,
                        'confidence': confidence,
                        'decision_by': decision_by,
                        'dual_details': {
                            safe_model_name: {
                                'prediction': safe_decision,
                                'risk_windows': safe_risk_windows,
                                'safe_windows': safe_safe_windows,
                                'probability': safe_avg_proba
                            },
                            risk_model_name: {
                                'prediction': risk_decision if risk_predictions is not None else None,
                                'risk_windows': risk_risk_windows,
                                'safe_windows': risk_safe_windows,
                                'probability': risk_avg_proba
                            } if risk_predictions is not None else None
                        },
                        'n_windows': n_windows,
                        'signal_shape': list(emg.shape),
                        'signal_data': emg[::10, :].tolist()
                    })
                    
                except Exception as e:
                    import traceback
                    results.append({
                        'signal': signal_file,
                        'error': f'Error: {str(e)}',
                        'traceback': traceback.format_exc()
                    })
            
            return jsonify({
                'mode': 'dual',
                'config': dual_config,
                'results': results
            })
        
        # MODO INDEPENDIENTE
        else:
            model_name = data.get('model_name')
            model_type = data.get('model_type')
            
            if not model_name or not model_type:
                return jsonify({'error': 'Faltan parámetros'}), 400
            
            if model_type == 'ml':
                model_path = MODELS_DIR / f'{model_name}.pkl'
                if not model_path.exists():
                    return jsonify({'error': f'{model_name}.pkl no encontrado'}), 404
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                scaler_path = MODELS_DIR / 'scaler.pkl'
                if not scaler_path.exists():
                    return jsonify({'error': 'scaler.pkl no encontrado'}), 404
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            elif model_type == 'dl':
                # Buscar archivo .keras primero, luego .h5
                model_path = MODELS_DIR / f'{model_name}.keras'
                if not model_path.exists():
                    model_path = MODELS_DIR / f'{model_name}.h5'
                if not model_path.exists():
                    return jsonify({'error': f'{model_name} no encontrado'}), 404
                
                try:
                    # Cargar con custom_objects
                    model = keras.models.load_model(
                        str(model_path),
                        custom_objects=CUSTOM_OBJECTS,  # Usar el dict global definido arriba
                        compile=False
                    )
                    print(f"✅ {model_name} cargado exitosamente desde {model_path.suffix}")
                    
                except Exception as e:
                    print(f"❌ Error cargando {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({
                        'error': f'No se pudo cargar {model_name}',
                        'detail': str(e)
                    }), 500
                
                scaler_path = MODELS_DIR / 'scaler.pkl'
                if not scaler_path.exists():
                    return jsonify({'error': 'scaler.pkl no encontrado'}), 404
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                return jsonify({'error': 'Tipo inválido'}), 400
            
            results = []
            
            for signal_file in signal_files:
                signal_path = _resolve_signal_path(signal_file)
                if not signal_path:
                    results.append({'signal': signal_file, 'error': 'Archivo no encontrado'})
                    continue
                
                try:
                    mat_data = loadmat(str(signal_path))
                except:
                    results.append({'signal': signal_file, 'error': 'Error al cargar'})
                    continue
                
                emg = mat_data.get('emg')
                if emg is None:
                    for key, value in mat_data.items():
                        if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                            emg = value
                            break
                
                if emg is None or emg.ndim != 2:
                    results.append({'signal': signal_file, 'error': 'EMG inválido'})
                    continue
                
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
                
                try:
                    if model_type == 'ml':
                        features, windows = preprocess_signal_ml(emg, scaler)
                        if len(features) == 0:
                            results.append({'signal': signal_file, 'error': 'Sin ventanas'})
                            continue
                        
                        predictions = model.predict(features)
                        avg_proba = np.mean(model.predict_proba(features), axis=0).tolist() if hasattr(model, 'predict_proba') else None
                        final_pred = int(np.round(np.mean(predictions)))
                        
                        n_windows = len(predictions)
                        risk_windows = int(np.sum(predictions == 1))
                        safe_windows = int(np.sum(predictions == 0))
                        risk_pct = 100.0 * risk_windows / n_windows if n_windows > 0 else 0.0
                    
                    else:
                        windows = preprocess_signal_dl(emg, scaler)
                        if len(windows) == 0:
                            results.append({'signal': signal_file, 'error': 'Sin ventanas'})
                            continue
                        
                        # Obtener threshold optimizado
                        threshold = get_optimized_threshold(model_name)
                        probas = model.predict(windows, verbose=0)
                        predictions = (probas > threshold).astype(int).flatten()

                        avg_proba_risk = float(np.mean(probas))
                        avg_proba = [1.0 - avg_proba_risk, avg_proba_risk]
                        final_pred = int(np.round(np.mean(predictions)))
                        
                        n_windows = len(predictions)
                        risk_windows = int(np.sum(predictions == 1))
                        safe_windows = int(np.sum(predictions == 0))
                        risk_pct = 100.0 * risk_windows / n_windows if n_windows > 0 else 0.0
                    
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
                        'signal_data': emg[::10, :].tolist()
                    })
                
                except Exception as e:
                    import traceback
                    results.append({
                        'signal': signal_file,
                        'error': f'Error: {str(e)}',
                        'traceback': traceback.format_exc()
                    })
            
            return jsonify({
                'mode': 'single',
                'model': model_name,
                'model_type': model_type,
                'results': results
            })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/')
def index():
    index_path = FRONTEND_DIR / 'index.html'
    if index_path.exists():
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    return jsonify({'status': 'API activa'})

@app.route('/<path:path>')
def serve_static(path):
    file_path = FRONTEND_DIR / path
    if file_path.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return jsonify({'error': 'No encontrado'}), 404

if __name__ == '__main__':
    MODELS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    FRONTEND_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 SERVIDOR CLASIFICADOR STC - CON SISTEMA DUAL")
    print("="*70)
    print(f"\n📂 Rutas: {MODELS_DIR}")
    print(f"⚙️  Window: {cfg.WINDOW_SIZE}ms, Overlap: {cfg.OVERLAP*100:.0f}%")
    
    dual_config = load_dual_system_config()
    if dual_config and dual_config.get('available'):
        print(f"\n🔥 Sistema Dual DISPONIBLE:")
        print(f"   SAFE: {dual_config['safe_specialist']['name']} (precision {dual_config['safe_specialist']['precision']:.3f})")
        print(f"   RISK: {dual_config['risk_specialist']['name']} (recall {dual_config['risk_specialist']['recall']:.3f})")
    else:
        print(f"\n⚠️  Sistema Dual NO disponible")
    
    print(f"\n🌐 http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
