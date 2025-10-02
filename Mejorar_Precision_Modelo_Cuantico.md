# Plan de Mejora de Precisión para QESN (Quantum Energy State Network)

## Objetivo

Elevar la precisión global del modelo QESN del ~93% actual hacia un rango 90%-95% de forma consistente sobre el nuevo dataset de exposición, manteniendo la interpretabilidad de la arquitectura y la velocidad de inferencia.

---

## 1. Alineación con el Checkpoint Base (≈92.6% accuracy)

- **Reutilizar el export oficial** (`model_weights.bin`, `model_config.json`) como punto de partida inicial.
- Verificar con `examples/quick_demo.py` que las probabilidades en el entorno V2 coinciden con las del repositorio GPU (`QESN_GPU_REPO`).
- Confirmar que todos los demos y notebooks consumen la física correcta (`window_size=60`, `stride=30`, `dt=0.002`, `energy_injection=0.05`, `coupling_strength=diffusion_rate=0.5`, `decay_rate=0.001`).

---

## 2. Limpieza y Preparación de Datos

- **Curar pistas de keypoints**: eliminar frames con tracking defectuoso (<0.3 confianza media) y reemplazarlos con interpolaciones suaves antes de inyectar energía.
- **Balanceo temporal**: asegurar que cada ventana de 60 frames mantenga la proporción real de clases minoritarias; usar sobre-muestreo controlado para acciones poco frecuentes.
- **Normalización adaptativa**: ajustar escalas de coordenadas por cámara y sesión para evitar desvíos en la codificación espacial.

---

## 3. Ajustes de Física Cuántica

- **Explorar `dt` dinámico**: reducir a 0.0015 en ventanas con movimientos rápidos; mantener 0.002 para acciones lentas.
- **Acoplamiento adaptativo**: permitir que `coupling_strength` oscile entre 0.45 y 0.52 dependiendo de la entropía por ventana (aplicar en entrenamiento con `updatePhysicsParameters`).
- **Refinar `energy_injection`**: experimentar con valores 0.04–0.06 según el conteo de keypoints válidos; usar normalización por frame para evitar saturaciones.

---

## 4. Clasificador Final (Capa Lineal + Softmax)

- **Regularización L2**: incrementar `weight_decay` de 1e-5 a 2e-5 para reducir oscilaciones en clases ruidosas.
- **Temperatura Softmax**: en inferencia, aplicar temperatura 0.95 para suavizar distribuciones y mejorar ranking top-k.
- **Calibración post-hoc**: entrenar un modelo de calibración (Platt scaling o isotonic) sobre el conjunto de validación para ajustar probabilidades finales.

---

## 5. Entrenamiento Incremental y Fine-Tuning

1. **Warm-start**: cargar pesos del export y continuar entrenamiento con tasa de aprendizaje reducida (`lr=5e-4`).
2. **Curriculum temporal**: comenzar con ventanas de 40 frames y aumentar a 60 tras estabilizar métricas; reduce ruido inicial.
3. **Validación cruzada**: emplear 3 folds estratificados por sesión para medir robustez; objetivo ≥90% en todos los folds.
4. **Early stopping**: Monitorizar accuracy y macro-F1; detener si no hay mejora tras 5 épocas.

---

## 6. Evaluación y Métricas

- **Macro-F1**: asegurarse de que las clases minoritarias no caen por debajo de 0.15.
- **Confusion matrices**: analizar pares de clases con más confusión (p. ej. `approach` vs `follow`) y ajustar reglas de posprocesado si es necesario.
- **Pipelines de reporting**: usar los notebooks `QESN_Complete_Classification_Demo.ipynb` y `QESN_Professional_Quantum_Demo.ipynb` para generar informes comparativos antes/después de los ajustes.

---

## 7. Recomendaciones de Producción

- Versionar checkpoints y configs (`kaggle_model/`), documentando fecha, hiperparámetros y métricas de validación.
- Automatizar una prueba rápida con `examples/quick_demo.py` y datos sintéticos para detectar regressiones en inferencia.
- Mantener scripts de conversión y validación sincronizados entre el repositorio GPU y V2 para evitar divergencias físicas.

---

## Resumen de Próximos Pasos

1. Copiar `model_weights.bin` / `model_config.json` al repositorio V2 y validar paridad de inferencia.
2. Ejecutar una tanda de entrenamiento fine-tuning con los ajustes de física y regularización propuestos.
3. Medir accuracy y F1 en validación cruzada; iterar sobre parámetros (`dt`, `energy_injection`, `weight_decay`).
4. Documentar resultados y preparar material de presentación con los nuevos gráficos y métricas.

Con estas acciones se espera consolidar la precisión global en el rango 90%-95%, manteniendo la filosofía física del modelo y la reproducibilidad del pipeline.
