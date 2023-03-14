package org.nathanael.sensei.optimizers;

public class Adam extends Optimizer {
    private final float[][] weightMomentum;
    private final float[] biasMomentum;

    public Adam(float alpha, int inputSize, int outputSize) {
        super(alpha, inputSize, outputSize);

        weightMomentum = new float[outputSize][inputSize];
        biasMomentum = new float[outputSize];
    }

    public Adam(float beta1, float beta2, float alpha, int inputSize, int outputSize) {
        super(beta1, beta2, alpha, inputSize, outputSize);

        weightMomentum = new float[outputSize][inputSize];
        biasMomentum = new float[outputSize];
    }

    @Override
    public float weightDescent(int x, int y, float gradient) {
        weightMomentum[x][y] = beta1 * weightMomentum[x][y] + (1 - beta1) * gradient;
        weightVectors[x][y] = (float) (beta2 * weightVectors[x][y] + (1 - beta2) * Math.pow(gradient, 2.0));

        float momentum = (float) (weightMomentum[x][y] / (1 - Math.pow(beta1, counter + 1)));
        float rms = (float) (weightVectors[x][y] / (1 - Math.pow(beta2, counter + 1)));

        return (float) (alpha * (momentum / (Math.sqrt(rms) + 0.00000001)));
    }

    @Override
    public float biasDescent(int index, float gradient) {
        biasMomentum[index] = beta1 * biasMomentum[index] + (1 - beta1) * gradient;
        biasVectors[index] = (float) (beta2 * biasVectors[index] + (1 - beta2) * Math.pow(gradient, 2.0));

        float momentum = (float) (biasMomentum[index] / (1 - Math.pow(beta1, counter + 1)));
        float rms = (float) (biasVectors[index] / (1 - Math.pow(beta2, counter + 1)));

        return (float) (alpha * (momentum / (Math.sqrt(rms) + 0.00000001)));
    }
}
