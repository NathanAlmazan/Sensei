package org.nathanael.sensei.loss;

public class BinaryCrossEntropy implements LossFunction {
    @Override
    public float forward(float[] output, float[] answer) {
        float loss = 0;

        for (int x = 0; x < output.length; x++) {
            if (answer[x] == 1.0)
                loss = (float) (-1 * Math.log(output[x]));
        }

        return loss;
    }

    @Override
    public float[] backward(float[] output, float[] answer) {
        float[] gradient = new float[output.length];

        for (int x = 0; x < output.length; x++) {
            gradient[x] = (output[x] - answer[x]) / (output[x] * (1 - output[x]));
        }

        return gradient;
    }
}
