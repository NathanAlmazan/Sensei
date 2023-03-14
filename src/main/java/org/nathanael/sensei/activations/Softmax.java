package org.nathanael.sensei.activations;

public class Softmax implements Activation {

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];

        float summation = 0;
        for (float v : input)
            summation += (float) Math.exp(v);

        for (int x = 0; x < input.length; x++) {
            output[x] = (float) (Math.exp(input[x]) / summation);
        }

        return output;
    }

    @Override
    public float[] backward(float[] product, float[] activation, float[] error) {
        float[] gradient = new float[activation.length];
        float[][] derivatives = new float[activation.length][activation.length];

        for (int x = 0; x < activation.length; x++) {
            for (int y = 0; y < activation.length; y++){
                if (x == y) derivatives[x][y] = activation[x] * (1 - activation[x]);
                else derivatives[x][y] = -1 * activation[x] * activation[y];
            }
        }

        for (int x = 0; x < activation.length; x++) {
            float summation = 0;
            for (int y = 0; y < activation.length; y++)
                summation += derivatives[x][y] * error[y];

            gradient[x] = summation;
        }

        return gradient;
    }
}
