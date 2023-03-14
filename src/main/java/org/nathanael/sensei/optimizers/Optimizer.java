package org.nathanael.sensei.optimizers;

public class Optimizer {
    protected final float beta1;
    protected final float beta2;
    protected final float alpha;
    protected final float[][] weightVectors;
    protected final float[] biasVectors;
    protected int counter;

    public Optimizer(float alpha, int inputSize, int outputSize) {
        this.beta1 = 0.9f;
        this.beta2 = 0.999f;
        this.alpha = alpha;
        this.weightVectors = new float[outputSize][inputSize];
        this.biasVectors = new float[outputSize];
        this.counter = 0;
    }

    public Optimizer(float beta, float alpha, int inputSize, int outputSize) {
        this.beta1 = beta;
        this.beta2 = 0.999f;
        this.alpha = alpha;
        this.weightVectors = new float[outputSize][inputSize];
        this.biasVectors = new float[outputSize];
        this.counter = 0;
    }

    public Optimizer(float beta1, float beta2, float alpha, int inputSize, int outputSize) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.alpha = alpha;
        this.weightVectors = new float[outputSize][inputSize];
        this.biasVectors = new float[outputSize];
        this.counter = 0;
    }

    public float weightDescent(int x, int y, float gradient) {
        return 0;
    }

    public float biasDescent(int index, float gradient) {
        return 0;
    }

    public void updateCounter()  {
        counter++;
    }
}
