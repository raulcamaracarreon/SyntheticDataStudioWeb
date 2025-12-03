# 🧬 Synthetic Data Studio (Web)

> **Advanced Synthetic Data Generation & Augmentation Tool**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![Engine](https://img.shields.io/badge/Engine-SDV%20(Synthetic%20Data%20Vault)-purple)
![Status](https://img.shields.io/badge/Privacy-GDPR%20Compliant-orange)

Una plataforma web integral para la generación de datos sintéticos. Diseñada para Científicos de Datos y equipos de QA que necesitan datasets realistas para pruebas, entrenamiento de modelos o aumentación de datos (Data Augmentation), sin comprometer la privacidad de la información sensible.

---

## 🚀 Capacidades Principales

La aplicación opera en dos modos distintos para cubrir diferentes casos de uso:

### 1. 🛠️ Modo Diseño Manual (Schema Builder)
Ideal para crear datos desde cero (Mock Data) definiendo reglas estadísticas específicas.
* **Control de Distribuciones:** Soporte para distribuciones Normal, Uniforme, LogNormal, Gamma, Weibull, Poisson, Binomial, entre otras.
* **Matriz de Correlación:** Definición manual de correlaciones entre variables utilizando **Cópulas Gaussianas** para mantener la coherencia matemática.
* **Restricciones:** Configuración de límites (min/max), redondeo de decimales y manejo de valores atípicos (outliers).
* **Gestión de Esquemas:** Guardar y cargar configuraciones complejas en formato `.json`.

### 2. 🧠 Modo Data-Driven (SDV AI)
Utiliza Machine Learning para "aprender" la estructura de un CSV real y generar nuevos datos que imitan sus propiedades estadísticas.
* **Motor:** Basado en `GaussianCopulaSynthesizer` de la librería SDV.
* **Metadata Detection:** Detección automática de tipos de datos y relaciones.
* **Reporte de Calidad:** Generación de métricas de fidelidad (`QualityReport`) para comparar la similitud entre los datos reales y los sintéticos.
---

## 🛠️ Arquitectura Técnica

El proyecto utiliza un stack robusto de Python para el procesamiento estadístico y Flask para la interfaz web.

| Componente | Tecnología | Uso |
| :--- | :--- | :--- |
| **Backend** | `Flask` | Servidor web y enrutamiento. |
| **Core Estadístico** | `Scipy` + `Numpy` | Generación de números aleatorios y distribuciones complejas. |
| **Generative AI** | `SDV` (Synthetic Data Vault) | Modelado de datos tabulares y aprendizaje de estructura. |
| **Visualización** | `Matplotlib` | Generación de histogramas y gráficos de barras en tiempo real. |
| **Data Handling** | `Pandas` | Manipulación, exportación CSV y análisis exploratorio. |

---

## 📦 Instalación y Despliegue

### Requisitos Previos
* Python 3.8+
* Pip

### 1. Clonar el repositorio
```bash
git clone [https://github.com/raul-camara-20416b379/SyntheticDataStudioWeb.git](https://github.com/raul-camara-20416b379/SyntheticDataStudioWeb.git)
cd SyntheticDataStudioWeb
2. Configurar entorno virtual
Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Instalar dependencias
Este proyecto requiere librerías científicas pesadas.

Bash

pip install -r requirements.txt
4. Ejecutar localmente
Bash

flask run
Accede a http://127.0.0.1:5000

☁️ Despliegue en Producción (Render/Docker)
Para desplegar en servicios como Render, se recomienda usar Gunicorn como servidor WSGI.

Comando de arranque (Start Command):

Bash

gunicorn app:app
Nota sobre Memoria: El uso de SDV puede ser intensivo en memoria RAM dependiendo del tamaño del dataset de entrenamiento. Se recomienda un entorno con al menos 1GB de RAM para datasets pequeños/medianos.

📄 Estructura del Proyecto
Plaintext

SyntheticDataStudioWeb/
├── app.py                # Controlador principal (Flask Routes)
├── generator.py          # Lógica de generación manual y Schema
├── templates/            # Plantillas HTML (Jinja2)
│   ├── index.html        # Dashboard principal
│   ├── sdv/              # Plantillas específicas para modo SDV
│   └── ...
├── static/               # Estilos CSS y Scripts JS
├── requirements.txt      # Dependencias
└── README.md             # Documentación
Autor: Raúl Héctor Cámara Carreón
