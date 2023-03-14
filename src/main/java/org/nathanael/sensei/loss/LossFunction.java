package org.nathanael.sensei.loss;

public interface LossFunction {
    float forward(float[] output, float[] answer);
    float[] backward(float[] output, float[] answer) throws Exception;
}
