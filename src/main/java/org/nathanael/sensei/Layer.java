package org.nathanael.sensei;

import org.nathanael.sensei.activations.Activation;
import org.nathanael.sensei.initialization.WeightInitialization;
import org.nathanael.sensei.optimizers.Optimizer;

import java.util.Arrays;

public class Layer {
    private final float[][] weights;
    private final float[] biases;
    private final float[][] weightGradients;
    private final float[] biasGradients;
    private final Activation activation;
    private final Optimizer optimizer;
    private float[] inputCache;
    private float[] productCache;
    private float[] activationCache;


    public Layer(int input, int output, Activation activation, WeightInitialization initialization, Optimizer optimizer) {
        this.weights = initialization.findInitialWeights(input, output);
        this.biases = new float[output];
        this.weightGradients = new float[output][input];
        this.biasGradients = new float[output];
        this.activation = activation;
        this.optimizer = optimizer;
    }

    public Layer(float[][] weights, float[] biases, Activation activation, Optimizer optimizer) {
        this.weights = weights;
        this.biases = biases;
        this.weightGradients = new float[weights.length][weights[0].length];
        this.biasGradients = new float[biases.length];
        this.activation = activation;
        this.optimizer = optimizer;
    }

    public float[] forwardPropagation(float[] input) {
        inputCache = input; // cache input
        float[] product = new float[weights.length]; // matrix product vector

        // get the matrix product of input and weights
        for (int x = 0; x < weights.length; x++) {
            float summation = 0;
            for (int y = 0; y < weights[x].length; y++)
                summation += input[y] * weights[x][y];

            // add the product to biases
            product[x] = summation + biases[x];
        }

        float[] activations = activation.forward(product); // use the non-linear activation formula

        // cache matrix product and activation
        productCache = product;
        activationCache = activations;

        return activations;
    }

    public float[] backwardPropagation(float[] lastLayer, boolean inputLayer) {
        // matrix product of derivative of last layer and current layer's activation
        float[] gradients = activation.backward(productCache, activationCache, lastLayer);

        // cache weight and bias gradients
        for (int x = 0; x < weights.length; x++) {
            for (int y = 0; y < weights[x].length; y++)
                weightGradients[x][y] += inputCache[y] * gradients[x]; // weights derivative

            biasGradients[x] += gradients[x]; // bias derivative
        }

        if (inputLayer) return null;

        // compute derivative of last layer neurons (aL - 1)
        float[] neurons = new float[inputCache.length];
        for (int x = 0; x < inputCache.length; x++) {
            float summation = 0;
            for (int y = 0; y < weights.length; y++)
                summation += weights[y][x] * gradients[y];

            neurons[x] = summation;
        }

        return neurons;
    }

    public void updateWeightAndBiases(int iterations) {
        // update weights and biases
        for (int x = 0; x < weights.length; x++) {
            for (int y = 0; y < weights[x].length; y++) {
                weights[x][y] -= optimizer.weightDescent(x, y, (weightGradients[x][y] / iterations));
                weightGradients[x][y] = 0;
            }

            biases[x] -= optimizer.biasDescent(x, (biasGradients[x] / iterations));
            biasGradients[x] = 0;
        }

        optimizer.updateCounter();
    }

    public float[][] getWeights() {
        return weights;
    }

    public float[] getBiases() {
        return biases;
    }

    public Activation getActivation() {
        return activation;
    }
}
