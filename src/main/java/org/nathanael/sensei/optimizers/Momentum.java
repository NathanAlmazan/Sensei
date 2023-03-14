package org.nathanael.sensei.optimizers;

public class Momentum extends Optimizer {

    public Momentum(float alpha, int inputSize, int outputSize) {
        super(alpha, inputSize, outputSize);
    }

    public Momentum(float beta, float alpha, int inputSize, int outputSize) {
        super(beta, alpha, inputSize, outputSize);
    }

    @Override
    public float weightDescent(int x, int y, float gradient) {
        weightVectors[x][y] = beta1 * weightVectors[x][y] + (1 - beta1) * gradient;

        return alpha * weightVectors[x][y];
    }

    @Override
    public float biasDescent(int index, float gradient) {
        biasVectors[index] = beta1 * biasVectors[index] + (1 - beta1) * gradient;

        return alpha * biasVectors[index];
    }
}
