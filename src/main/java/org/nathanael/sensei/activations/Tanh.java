package org.nathanael.sensei.activations;

public class Tanh implements Activation {
    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];

        for (int x = 0; x < input.length; x++) {
            output[x] = (float) ((Math.exp(input[x]) - Math.exp(-1 * input[x])) / (Math.exp(input[x]) + Math.exp(-1 * input[x])));
        }

        return output;
    }

    @Override
    public float[] backward(float[] product, float[] activation, float[] error) {
        float[] gradient = new float[activation.length];

        for (int x = 0; x < activation.length; x++) {
            gradient[x] = (float) (1 - Math.pow(activation[x], 2.0));
        }

        return gradient;
    }
}
