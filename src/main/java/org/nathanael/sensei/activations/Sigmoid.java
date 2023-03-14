package org.nathanael.sensei.activations;

public class Sigmoid implements Activation {

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];

        for (int x = 0; x < input.length; x++) {
            output[x] = (float) (1 / (1 + Math.exp(-1 * input[x])));
        }

        return output;
    }

    @Override
    public float[] backward(float[] product, float[] activation, float[] error) {
        float[] gradient = new float[activation.length];

        for (int x = 0; x < activation.length; x++) {
            float derivative = activation[x] * (1 - activation[x]);
            gradient[x] = derivative * error[x];
        }

        return gradient;
    }
}
