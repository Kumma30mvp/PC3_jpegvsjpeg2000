# Análisis de compresión JPEG vs JPEG2000 sobre el dataset Kodak

Este proyecto en Python permite comparar, de manera reproducible, la compresión **JPEG** —basada en DCT— y **JPEG2000** —basada en wavelets— sobre el *Kodak Lossless True Color Image Dataset*, compuesto por 24 imágenes PNG en formato RGB de 24 bits. Para cada imagen y cada nivel de compresión, el pipeline codifica, decodifica, mide el tamaño del archivo resultante y calcula métricas de fidelidad visual, como PSNR y SSIM. Además, genera gráficos comparativos, mapas de error y recortes visuales para analizar artefactos de compresión.

El proyecto está diseñado para ejecutarse desde Visual Studio Code en Windows, y para que pueda ser clonado y reproducido fácilmente por otros compañeros.

---

## Contexto académico

La compresión de imágenes reduce el espacio de almacenamiento requerido por una imagen, idealmente sin alterar de forma perceptible su contenido visual. En este proyecto se estudian dos métodos fundamentales de compresión con pérdida:

- **JPEG**: utiliza la **Transformada Discreta del Coseno (DCT)** aplicada por bloques, cuantización escalar sobre cada bloque de 8×8 píxeles y codificación entrópica. Fue estandarizado en 1992.
- **JPEG2000**: utiliza la **Transformada Wavelet Discreta (DWT)** aplicada sobre la imagen completa, junto con codificación embebida por planos de bits (EBCOT). Fue estandarizado en el año 2000 y fue diseñado para mitigar los artefactos de bloques característicos de JPEG, además de permitir decodificación progresiva.

El objetivo de este proyecto es evaluar, sobre el mismo conjunto de imágenes:

1. **Eficiencia de compresión**: qué tan pequeños resultan los archivos comprimidos.
2. **Calidad de imagen**: cuánta información se pierde respecto a la imagen original.
3. **Artefactos visuales**: qué tipo de degradación introduce cada método.

---

## Conceptos clave

| Término | Definición breve |
|---|---|
| **Compresión de imágenes** | Proceso de codificar una imagen usando menos bytes que su representación directa en píxeles. |
| **Compresión con pérdida** | La imagen decodificada es visualmente similar a la original, pero no idéntica bit a bit. Parte de la información se descarta de forma permanente para reducir el tamaño del archivo. |
| **JPEG** | Códec de compresión con pérdida basado en DCT por bloques de 8×8, cuantización y codificación entrópica. |
| **DCT** | Transformada Discreta del Coseno. Expresa un bloque de 8×8 píxeles como una combinación de funciones base cosenoidales. En imágenes naturales, la mayor parte de la energía suele concentrarse en bajas frecuencias. |
| **JPEG2000** | Códec con pérdida o sin pérdida basado en una transformada wavelet multirresolución. Sustituye la estructura por bloques de la DCT por una degradación más suave y localizada en frecuencia. |
| **Transformada wavelet** | Descomposición multiescala que separa la imagen en subbandas de aproximación (LL) y detalle (LH, HL, HH). JPEG2000 utiliza la wavelet irreversible 9/7 para compresión con pérdida. |
| **Ratio de Compresión (CR)** | `tamaño_archivo_original / tamaño_archivo_comprimido`. Un valor más alto indica una compresión más fuerte. |
| **MSE** | Error cuadrático medio entre los píxeles de la imagen original y los de la imagen reconstruida. Un valor menor indica mayor similitud. |
| **PSNR** | `10·log10(255² / MSE)`. Mide la fidelidad a nivel de píxel en decibelios (dB). Un valor más alto indica mejor calidad. |
| **SSIM** | Índice de Similitud Estructural, con valores entre 0 y 1. Modela similitudes perceptuales en luminancia, contraste y estructura. Un valor más alto indica mayor similitud estructural. |
| **Mapa de error** | Representación de `\|original − reconstruida\|`, normalmente amplificada, que permite visualizar en qué zonas el códec perdió más información. |
| **Artefactos de bloque** | Patrón visible en forma de cuadrícula de 8×8 producido por JPEG cuando la compresión es alta. |
| **Desenfoque** | Pérdida de detalles de alta frecuencia. Puede aparecer en ambos códecs cuando la compresión es muy elevada. |
| **Ringing** | Oscilaciones o halos cerca de bordes pronunciados, especialmente frecuentes en imágenes codificadas mediante wavelets. |
| **Pérdida de textura fina** | Suavizado de patrones pequeños o repetitivos, como tela, follaje o poros de la piel. |

---

## Dataset

Coloque las 24 imágenes Kodak en la siguiente ruta:

```text
data/kodak/raw/kodim01.png
data/kodak/raw/kodim02.png
...
data/kodak/raw/kodim24.png
