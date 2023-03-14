package org.nathanael.sensei.optimizers;

public class RMSProp extends Optimizer {

    public RMSProp(float alpha, int inputSize, int outputSize) {
        super(alpha, inputSize, outputSize);
    }

    public RMSProp(float beta, float alpha, int inputSize, int outputSize) {
        super(beta, alpha, inputSize, outputSize);
    }

    @Override
    public float weightDescent(int x, int y, float gradient) {
        weightVectors[x][y] = (float) (beta1 * weightVectors[x][y] + (1 - beta1) * Math.pow(gradient, 2.0));

        return (float) (alpha * (gradient / (Math.sqrt(weightVectors[x][y]) + 0.00000001)));
    }

    @Override
    public float biasDescent(int index, float gradient) {
        biasVectors[index] = (float) (beta1 * biasVectors[index] + (1 - beta1) * Math.pow(gradient, 2.0));

        return (float) (alpha * (gradient / (Math.sqrt(biasVectors[index]) + 0.00000001)));
    }
}
