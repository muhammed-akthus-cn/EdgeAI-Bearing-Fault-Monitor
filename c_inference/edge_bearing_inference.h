/* Auto-generated Embedded Edge-AI Bearing Fault Inference Header */
/* Model Footprint: 307 Bytes (INT8 Quantized) | SQNR: 41.27 dB */

#ifndef EDGE_BEARING_INFERENCE_H
#define EDGE_BEARING_INFERENCE_H

#include <stdint.h>

#define NUM_FEATURES 8
#define HIDDEN1_SIZE 16
#define HIDDEN2_SIZE 8
#define NUM_CLASSES  3

/* Predicted Fault Class IDs */
#define CLASS_HEALTHY          0
#define CLASS_INNER_RACE_FAULT 1
#define CLASS_OUTER_RACE_FAULT 2

/* Runs INT8 quantized inference on 8 extracted vibration features.
 * Returns predicted class ID (0, 1, or 2) and writes raw logits to out_scores[3]. */
int8_t predict_bearing_fault(const float raw_features[NUM_FEATURES], float out_scores[NUM_CLASSES]);

#endif /* EDGE_BEARING_INFERENCE_H */
