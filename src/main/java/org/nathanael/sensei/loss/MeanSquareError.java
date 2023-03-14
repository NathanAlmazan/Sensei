package org.nathanael.sensei.loss;

import java.util.Arrays;

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
    public float[] backward(float[] output, float[] answer) throws Exception {
        float[] gradient = new float[output.length];

        for (int x = 0; x < output.length; x++) {
            float grad = (-2 * (answer[x] - output[x])) / output.length;

            if (Float.isInfinite(grad) || Float.isNaN(grad))
                throw new Exception("Gradient is infinite or not a number. Last output is " + Arrays.toString(output));

            gradient[x] = grad;
        }

        return gradient;
    }
}
