"""
Script para organizar datos de Ninapro DB2 Exercise B (E1) en carpetas train y test.

Este script:
1. Copia archivos completos a la carpeta 'train' para entrenamiento
2. Extrae repeticiones específicas de test a archivos individuales en carpeta 'test'
3. Cada archivo de test contiene una sola repetición de un movimiento de un sujeto

Autor: Karen Nicolle Arango Valencia
Fecha: 2025
"""

import os
import shutil
import scipy.io as sio
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN - MODIFICAR SEGÚN NECESIDADES
# ============================================================================

# Ruta donde están los archivos originales de Ninapro DB2
SOURCE_DIR = r"C:\Users\karen\Desktop\ninaproV2\DB2_E1_only"

# Ruta donde se crearán las carpetas train y test
OUTPUT_DIR = r"C:\Users\karen\Desktop\ninaproV5\DB2_E1_only"

# Sujetos a procesar (1-40 disponibles en DB2)
SUBJECTSTRAIN = range(1,41) # Modificar según necesidad

# Sujetos para separar
SUBJECTSTEST = [20]  # Modificar según necesidad

# Repeticiones que se usarán como TEST (según protocolo Ninapro estándar es [2, 5])
TEST_REPETITIONS = [5]  # Modificar si se desea otro split

# Movimientos a extraer (1-17 para Exercise B, 0 es rest)
MOVEMENTS = list(range(1, 18))  # [1, 2, 3, ..., 17] - Todos los movimientos

# Nombre del ejercicio (E1 corresponde a Exercise B en DB2)
EXERCISE = "E1"


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def create_directories(output_dir):
    """
    Crea las carpetas train y test si no existen.
    
    Args:
        output_dir (str): Ruta base donde crear las carpetas
        
    Returns:
        tuple: Rutas de carpetas (train_dir, test_dir)
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    # Crear carpetas si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"✓ Carpeta train creada/verificada: {train_dir}")
    print(f"✓ Carpeta test creada/verificada: {test_dir}")
    
    return train_dir, test_dir


def copy_train_files(source_dir, train_dir, subjects, exercise):
    """
    Copia los archivos completos a la carpeta train.
    
    Args:
        source_dir (str): Directorio con archivos originales
        train_dir (str): Directorio destino para train
        subjects (list): Lista de IDs de sujetos a copiar
        exercise (str): Nombre del ejercicio (ej: 'E1')
    """
    print("\n" + "="*70)
    print("COPIANDO ARCHIVOS COMPLETOS A CARPETA TRAIN")
    print("="*70)
    
    copied_count = 0
    
    for subject_id in subjects:
        # Formato del nombre: S{ID}_E1_A1.mat
        filename = f"S{subject_id}_{exercise}_A1.mat"
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(train_dir, filename)
        
        # Verificar si el archivo existe
        if not os.path.exists(source_path):
            print(f"⚠ ADVERTENCIA: Archivo no encontrado - {filename}")
            continue
        
        # Copiar archivo
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"  ✓ Copiado: {filename}")
        except Exception as e:
            print(f"  ✗ Error copiando {filename}: {str(e)}")
    
    print(f"\n✓ Total de archivos copiados a train: {copied_count}/{len(subjects)}")


def extract_test_samples(source_dir, test_dir, subjects, exercise, 
                         test_repetitions, movements):
    """
    Extrae repeticiones específicas de test a archivos individuales.
    
    Cada archivo contendrá:
    - emg: señal EMG (n_samples x 12)
    - subject: ID del sujeto
    - exercise: número de ejercicio
    - frequency: frecuencia de muestreo
    - rerepetition: número de repetición
    - movement: número de movimiento (guardado separadamente para referencia)
    
    Nota: NO se incluye 'restimulus' para evitar "trampas" durante evaluación.
    
    Args:
        source_dir (str): Directorio con archivos originales
        test_dir (str): Directorio destino para test
        subjects (list): Lista de IDs de sujetos
        exercise (str): Nombre del ejercicio
        test_repetitions (list): Lista de repeticiones a extraer
        movements (list): Lista de movimientos a extraer
    """
    print("\n" + "="*70)
    print("EXTRAYENDO MUESTRAS DE TEST A ARCHIVOS INDIVIDUALES")
    print("="*70)
    
    total_extracted = 0
    
    for subject_id in subjects:
        filename = f"S{subject_id}_{exercise}_A1.mat"
        source_path = os.path.join(source_dir, filename)
        
        # Verificar si el archivo existe
        if not os.path.exists(source_path):
            print(f"\n⚠ Sujeto {subject_id}: Archivo no encontrado")
            continue
        
        print(f"\n--- Procesando Sujeto {subject_id} ---")
        
        try:
            # Cargar archivo MATLAB
            data = sio.loadmat(source_path)
            
            # Extraer variables necesarias
            emg = data['emg']  # (n_samples, 12)
            restimulus = data['restimulus'].flatten()  # Etiquetas de movimiento
            rerepetition = data['rerepetition'].flatten()  # Número de repetición
            subject = data['subject'].flatten()[0]
            exercise_num = data['exercise'].flatten()[0]
            frequency = data['frequency'].flatten()[0]
            
            subject_extracted = 0
            
            # Iterar sobre movimientos de interés
            for movement in movements:
                # Iterar sobre repeticiones de test
                for rep in test_repetitions:
                    # Crear máscara para seleccionar datos específicos
                    # Condición: movimiento actual Y repetición actual
                    mask = (restimulus == movement) & (rerepetition == rep)
                    
                    # Verificar si hay datos para esta combinación
                    if not np.any(mask):
                        print(f"  ⚠ Mov {movement:2d}, Rep {rep}: Sin datos")
                        continue
                    
                    # Extraer señal EMG correspondiente
                    emg_sample = emg[mask, :]  # (n_samples_movement, 12)
                    rerepetition_sample = rerepetition[mask]
                    
                    # Verificar que todos los valores de repetición son iguales
                    if not np.all(rerepetition_sample == rep):
                        print(f"  ⚠ Mov {movement:2d}, Rep {rep}: Inconsistencia en repetición")
                        continue
                    
                    # Crear nombre de archivo: S{ID}_mov{MOV}_rep{REP}.mat
                    test_filename = f"S{subject_id}_mov{movement:02d}_rep{rep}.mat"
                    test_path = os.path.join(test_dir, test_filename)
                    
                    # Crear diccionario con datos a guardar
                    # IMPORTANTE: NO incluimos 'restimulus' para evitar trampas
                    test_data = {
                        'emg': emg_sample,
                        'subject': subject,
                        'exercise': exercise_num,
                        'frequency': frequency,
                        'rerepetition': rep,
                        'movement': movement,  # Guardado para referencia externa
                    }
                    
                    # Guardar archivo individual
                    sio.savemat(test_path, test_data)
                    
                    # Calcular duración en segundos
                    duration = emg_sample.shape[0] / frequency
                    
                    print(f"  ✓ Mov {movement:2d}, Rep {rep}: "
                          f"{emg_sample.shape[0]:6d} muestras ({duration:.2f}s) "
                          f"→ {test_filename}")
                    
                    subject_extracted += 1
                    total_extracted += 1
            
            print(f"  Total extraído para sujeto {subject_id}: {subject_extracted} archivos")
            
        except Exception as e:
            print(f"  ✗ Error procesando sujeto {subject_id}: {str(e)}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ TOTAL DE ARCHIVOS DE TEST EXTRAÍDOS: {total_extracted}")
    print(f"{'='*70}")


def print_summary(train_dir, test_dir):
    """
    Imprime un resumen de los archivos generados.
    
    Args:
        train_dir (str): Directorio de train
        test_dir (str): Directorio de test
    """
    # Contar archivos en cada carpeta
    train_files = len([f for f in os.listdir(train_dir) if f.endswith('.mat')])
    test_files = len([f for f in os.listdir(test_dir) if f.endswith('.mat')])
    
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Carpeta TRAIN: {train_files} archivos")
    print(f"Carpeta TEST:  {test_files} archivos")
    print(f"\nUbicación:")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")
    print("="*70)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta el proceso completo de organización de datos.
    """
    print("="*70)
    print("ORGANIZACIÓN DE DATOS NINAPRO DB2 PARA ENTRENAMIENTO Y TEST")
    print("="*70)
    print(f"\nConfiguración:")
    print(f"  - Directorio origen: {SOURCE_DIR}")
    print(f"  - Directorio destino: {OUTPUT_DIR}")
    print(f"  - Sujetos seleccionados: Train: {SUBJECTSTRAIN} y test: {SUBJECTSTEST}")
    print(f"  - Repeticiones de test: {TEST_REPETITIONS}")
    print(f"  - Movimientos a extraer: {len(MOVEMENTS)} movimientos")
    print(f"  - Ejercicio: {EXERCISE}")
    
    # Verificar que el directorio origen existe
    if not os.path.exists(SOURCE_DIR):
        print(f"\n✗ ERROR: Directorio origen no encontrado: {SOURCE_DIR}")
        return
    
    # Paso 1: Crear carpetas train y test
    train_dir, test_dir = create_directories(OUTPUT_DIR)
    
    # Paso 2: Copiar archivos completos a train
    copy_train_files(SOURCE_DIR, train_dir, SUBJECTSTRAIN, EXERCISE)
    
    # Paso 3: Extraer muestras de test
    extract_test_samples(SOURCE_DIR, test_dir, SUBJECTSTEST, EXERCISE, 
                         TEST_REPETITIONS, MOVEMENTS)
    
    # Paso 4: Mostrar resumen
    print_summary(train_dir, test_dir)
    
    print("\n✓ Proceso completado exitosamente!")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()