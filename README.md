# Atlas de la Mente

Proyecto abierto y exploratorio para ver cómo piensan por dentro las IAs: páginas interactivas, cuadernos y datos.

**Estado:** exploratorio, sin revisión externa. Las versiones 0 y 1 reproducen resultados ya conocidos. Las versiones 2 a 4 son un estudio pequeño sobre cómo deciden los modelos de lenguaje el género de «nueva» o «nuevo» en frases como «Mi casa es…».

| Versión | Qué muestra | Contenido |
|---|---|---|
| v0 | Una mini-IA aprende a sumar módulo 43 y descubre un algoritmo de ondas | train.py, datos, página |
| v1 | Una mini-IA aprende a copiar patrones (cabezas de inducción) | train.py (JAX), datos, página |
| v2 | Género en Qwen2.5-1.5B: dónde vive y cuándo se transmite | cuaderno de Colab, script, datos, página |
| v3 | Dos modelos, más palabras y qué cabezas de atención hacen el traspaso | cuaderno de Colab, script, datos, página |
| v4 | ¿Son imprescindibles esas cabezas? Apagarlas, frases con distractor, tres modelos | cuaderno de Colab, script, datos, página |

## Cómo reproducir
- v0: `pip install numpy` y `python train.py 12000`.
- v1: `pip install jax numpy` y `python train.py 5000`.
- v2 a v4: abre el cuaderno en Google Colab con GPU T4 y ejecuta todo; al final descarga un JSON con los resultados.
- Las páginas `pagina.html` son autocontenidas: se abren en el navegador.

## Resultado principal (v4)
En Qwen2.5-1.5B y Qwen2.5-0.5B, el género del sustantivo llega a la última palabra a través de 2 y 3 cabezas de atención del último cuarto de la red. Apagarlas elimina cerca del 80 % de la preferencia por el adjetivo correcto y apagar cabezas al azar no cambia nada. En BLOOM-1b1 el traspaso está repartido y las cabezas que más pesan explican solo el 23 %.

## Límites
Tres modelos (dos de la misma familia), una plantilla base, ablación solo en la última palabra y sin análisis de los patrones de atención. Falta repetirlo en más familias y fenómenos.

## Licencia y autoría
Licencia: MIT (ver LICENSE). Autoría: Santiago, con ayuda de Claude (Anthropic).»

Nota técnica: NOTA.md

/v0_suma/pagina.html


/v1_induccion/pagina.html


/v2_genero_qwen/pagina.html


/v3_dos_modelos/pagina.html


/v4_cabezas_imprescindibles/pagina.html
