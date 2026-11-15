#include <stdio.h>
#include "edge_bearing_inference.h"

int main(void) {
    const char* labels[3] = {"Healthy", "InnerRaceFault", "OuterRaceFault"};
    float sample_features[NUM_FEATURES] = {0.718403f, 4.262895f, 8.073343f, 5.933848f, 0.837023f, 0.718211f, 154.477227f, 601.805569f};
    float scores[NUM_CLASSES];
    int8_t pred = predict_bearing_fault(sample_features, scores);
    printf("Ground Truth : InnerRaceFault\n", "    printf("C Prediction : %s (Class %d)\n", labels[pred], pred);
    printf("Logit Scores : [%.3f, %.3f, %.3f]\n", scores[0], scores[1], scores[2]);
    return 0;
}
