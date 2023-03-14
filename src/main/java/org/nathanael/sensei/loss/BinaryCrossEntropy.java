package org.nathanael.sensei.loss;

import java.util.Arrays;

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
    public float[] backward(float[] output, float[] answer) throws Exception {
        float[] gradient = new float[output.length];

        for (int x = 0; x < output.length; x++) {
            float grad = (output[x] - answer[x]) / (output[x] * (1 - output[x]));

            if (Float.isInfinite(grad) || Float.isNaN(grad))
                throw new Exception("Gradient is infinite or not a number. Last output is " + Arrays.toString(output));

            gradient[x] = grad;
        }

        return gradient;
    }
}
