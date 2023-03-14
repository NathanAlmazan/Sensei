package org.nathanael.sensei.activations;

public class LeakyReLu implements Activation {

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];

        /*
            Leaky Relu is an activation function that
            outputs the input if it is greater than zero
            but outputs input multiplied by a small value
            close to zero if less than or equal to zero.
        */
        for (int x = 0; x < input.length; x++) {
            if (input[x] > 0) output[x] = input[x];
            else output[x] = input[x] * 0.1f;
        }

        return output;
    }

    @Override
    public float[] backward(float[] product, float[] activation, float[] error) {
        float[] gradient = new float[product.length];

        /*
            Leaky Relu has the derivative of 1 for inputs greate
            than 0 but alpha otherwise
        */
        for (int x = 0; x < product.length; x++) {
            if (product[x] > 0) gradient[x] = 1 * error[x];
            else gradient[x] = 0.1f * error[x];
        }

        return gradient;
    }
}
