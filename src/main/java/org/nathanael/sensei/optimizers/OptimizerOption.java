package org.nathanael.sensei.optimizers;

public abstract class OptimizerOption {
    public static Optimizer getOptimizer(String type, float learningRate, int inputSize, int outputSize) {
        if (type.equals("Adam")) return new Adam(learningRate, inputSize, outputSize);
        else if (type.equals("RMSProp")) return new RMSProp(learningRate, inputSize, outputSize);
        else return new Momentum(learningRate, inputSize, outputSize);
    }
}
