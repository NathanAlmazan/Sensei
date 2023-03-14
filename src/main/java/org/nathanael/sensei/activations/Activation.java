package org.nathanael.sensei.activations;

public interface Activation {
    float[] forward(float[] input);
    float[] backward(float[] product, float[] activation, float[] error);
}
