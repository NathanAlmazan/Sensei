package org.nathanael.sensei;

public class TrainingData {
    private final float[] input;
    private final float[] output;


    public TrainingData(float[] input, float[] output) {
        this.input = input;
        this.output = output;
    }

    public float[] getInput() {
        return input;
    }

    public float[] getOutput() {
        return output;
    }
}
