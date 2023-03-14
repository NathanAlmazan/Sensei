package org.nathanael.sensei.loss;

public class MeanSquareError implements LossFunction {

    @Override
    public float forward(float[] output, float[] answer) {

        float summation = 0;
        for (int x = 0; x < output.length; x++) {
            summation += Math.pow((answer[x] - output[x]), 2.0);
        }

        return summation / output.length;
    }

    @Override
    public float[] backward(float[] output, float[] answer) {
        float[] gradient = new float[output.length];

        for (int x = 0; x < output.length; x++) {
            gradient[x] = (-2 * (answer[x] - output[x])) / output.length;
        }

        return gradient;
    }
}
