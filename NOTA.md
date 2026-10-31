# Cómo deciden dos familias de modelos el género de «Mi casa es…»

*Nota técnica preliminar del Atlas de la Mente. Santiago, escrita con ayuda de Claude (Anthropic). Estado: exploratorio, sin revisión externa.*

## Resumen

En tres modelos de lenguaje (Qwen2.5-1.5B, Qwen2.5-0.5B y BLOOM-1b1) estudiamos cómo se decide el género del adjetivo en frases como «Mi casa es…» (¿«nueva» o «nuevo»?). En los dos Qwen, el género del sustantivo viaja a la última palabra a través de muy pocas cabezas de atención del último cuarto de la red: 2 en el modelo de 1,5B y 3 en el de 0,5B. Apagarlas elimina alrededor del 80 % de la preferencia por el adjetivo correcto, mientras que apagar el mismo número de cabezas al azar no cambia nada. Además, copiar solo esas cabezas desde la frase contraria recupera alrededor del 80 % de la decisión. En BLOOM-1b1, que concuerda casi siempre bien (99,5 %), el traspaso está repartido: las cuatro cabezas que más pesan explican solo entre el 23 % y el 37 %.

## Pregunta y método

- **Frases:** «Mi {sustantivo} es» más un adjetivo. El posesivo «mi» no tiene género, así que el género solo puede venir del sustantivo. La medida es la diferencia de logits entre la forma femenina y la masculina de varios adjetivos (6 pares en Qwen, 20 en BLOOM), promediada.
- **Palabras:** 107 sustantivos (Qwen) y 187 (BLOOM) de un solo token: regulares (-a femenino, -o masculino), sin sufijo marcado y «trampa», donde la terminación engaña (mano, foto; día, mapa, problema).
- **Parcheo de activaciones:** se copia lo que hay en una capa y una posición de la frase con un sustantivo del género contrario y se mide qué fracción de la diferencia se recupera. Solo se usan los pares que el modelo acierta con margen suficiente (76, 178 y 74 pares).
- **Cabezas:** parcheo cabeza a cabeza en la última palabra, ablación sustituyendo su salida por la media y control con el mismo número de cabezas al azar (30 sorteos).
- **Sondas:** clasificador lineal entrenado solo con palabras regulares y probado con regulares nuevas, trampas y palabras sin sufijo.
- **Frases con distractor:** «Mi casa del libro es…» y «Mi libro de la casa es…».

## Resultados

### Comportamiento

Qwen2.5-1.5B acierta el 89,7 % (intervalo del 95 %: 82,5 a 94,2) y las 12 palabras trampa; sus 11 fallos son todos femeninos. El margen medio es de 2,33 en masculinos frente a 1,10 en femeninos. Qwen2.5-0.5B acierta el 86,9 %. BLOOM-1b1 acierta el 99,5 % y solo falla «vez».

Con el artículo contradictorio («El casa es…»), el modelo sigue al sustantivo en el 47,7 % de los casos (Qwen-1.5B), el 72,9 % (Qwen-0.5B) y el 20,3 % (BLOOM): BLOOM hace mucho más caso al artículo.

### Dónde y cuándo

Hasta la capa 6 (Qwen) o la 4 (BLOOM) la decisión vive entera en el sustantivo. En Qwen-1.5B el mayor paso está en la capa 21 (del 32 % al 94 % de la decisión en la última palabra); en Qwen-0.5B, en la capa 19 (del 23 % al 95 %); en BLOOM es gradual, con su mayor paso en la capa 14 (del 44 % al 68 %). En Qwen-1.5B, el clasificador acierta de media solo el 42 % con palabras trampa en las cinco primeras capas (lee la terminación) y el 90 % o más desde la capa 20. En BLOOM lo hace desde la capa 3.

### Quién y si es imprescindible

| Modelo | Cabezas (capa.cabeza) | Recuperan al parchearlas | Margen que se pierde al apagarlas | Control al azar (margen) |
|---|---|---|---|---|
| Qwen2.5-1.5B | 21.8, 21.7 | 85 % | 81 % (1,65 → 0,32; acierto 89,7 → 69,2 %) | 1,62 ± 0,11 |
| Qwen2.5-0.5B | 19.4, 19.0, 20.13 | 83 % | 78 % (1,28 → 0,28; acierto 86,9 → 72,0 %) | 1,27 ± 0,03 |
| BLOOM-1b1 | 19.3, 22.1, 8.0, 14.4 | 37 % | 23 % (3,48 → 2,68; acierto 99,5 → 98,9 %) | 3,46 ± 0,04 |

Apagar todas las cabezas en la última palabra deja una respuesta constante, lo que confirma que la información del sustantivo llega por la atención.

### Frases con distractor

En «Mi casa del libro es…» (sustantivo femenino, distractor masculino), Qwen-1.5B y Qwen-0.5B aciertan el 58,3 % y el 50,8 % (cerca del azar), y con las cabezas clave apagadas el 25,8 % y el 20,8 %. BLOOM pasa del 97,5 % al 96,7 %. En «Mi libro de la casa es…», Qwen-1.5B baja del 96,7 % al 92,5 % (margen de 1,54 a 0,53; al azar, 1,53 ± 0,01), Qwen-0.5B del 79,2 % al 66,7 % y BLOOM mantiene el 98,3 % con el margen en 2,65 → 1,62. Las mismas cabezas siguen importando en otras frases, con la salvedad de que el masculino por defecto se mezcla con el efecto del distractor.

## Qué no sabemos

- Solo hay tres modelos, dos de la misma familia, y una plantilla base de español neutro.
- No hemos mirado a qué atienden esas cabezas. Que «muevan el género» es una interpretación funcional apoyada en parcheo y ablación, no en patrones de atención.
- La ablación solo se hace en la última palabra.
- Usar sustantivos de un solo token sesga hacia palabras frecuentes.
- Todos los fallos de Qwen son femeninos, lo que sugiere un masculino por defecto, pero no lo hemos aislado.

## Trabajo relacionado

Concordancia sujeto-verbo con parcheo en Gemma 2B en inglés y español (Ferrando y Costa-jussà, 2024); circuitos de concordancia en 29 lenguas y cinco familias de modelos (Gidi y otros, 2026); sondas causales para género y número en español (Naturalistic Causal Probing for Morpho-Syntax). No hemos encontrado un estudio equivalente sobre concordancia sustantivo-adjetivo a nivel de cabezas, pero no podemos descartar que exista.

## Reproducibilidad

Los cuadernos de Colab (v2 a v4), los datos en JSON y las páginas interactivas del Atlas acompañan a esta nota. Los experimentos usan semillas fijas.
