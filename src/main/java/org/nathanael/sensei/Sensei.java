package org.nathanael.sensei;

import org.nathanael.sensei.loss.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class Sensei {
    private final List<Layer> layers;
    private final LossFunction lossFunction;

    public Sensei(List<Layer> layers, LossFunction lossFunction) {
        this.layers = layers;
        this.lossFunction = lossFunction;
    }

    public void trainModel(List<TrainingData> trainingData, int epoch) {
        for (int e = 0; e < epoch; e++) {
            System.out.println("Epoch " + (e + 1));

            float error = 0;
            for (TrainingData data : trainingData) {
                // forward propagation
                float[] activation = data.getInput();
                for (Layer layer : layers)
                    activation = layer.forwardPropagation(activation);

                error += lossFunction.forward(activation, data.getOutput()); // save loss

                // backward propagation
                float[] gradients = lossFunction.backward(activation, data.getOutput());
                for (int z = layers.size() - 1; z >= 0 ; z--)
                    gradients = layers.get(z).backwardPropagation(gradients, z == 0);
            }

            // update weights and biases
            for (Layer layer : layers)
                layer.updateWeightAndBiases(trainingData.size());

            System.out.println("Error: " + error / trainingData.size());
        }
    }

    public void trainModel(List<TrainingData> trainingData, int batchSize, int epoch) {
        int stepPerEpoch = trainingData.size() / batchSize;
        for (int e = 0; e < epoch; e++) {
            System.out.println("Epoch " + (e + 1));

            float error = 0;
            for (int s = 0; s < stepPerEpoch; s++) {
                List<TrainingData> batch = trainingData.subList((batchSize * s), ((batchSize * s) + batchSize));

                for (TrainingData data : batch) {
                    // forward propagation
                    float[] activation = data.getInput();
                    for (Layer layer : layers)
                        activation = layer.forwardPropagation(activation);

                    error += lossFunction.forward(activation, data.getOutput()); // save loss

                    // backward propagation
                    float[] gradients = lossFunction.backward(activation, data.getOutput());
                    for (int z = layers.size() - 1; z >= 0 ; z--)
                        gradients = layers.get(z).backwardPropagation(gradients, z == 0);
                }

                // update weights and biases
                for (Layer layer : layers)
                    layer.updateWeightAndBiases(batchSize);
            }

            System.out.println("Loss: " + error / trainingData.size());
        }
    }

    public List<TrainingData> runModel(List<TrainingData> trainingData) {
        List<TrainingData> outputs = new ArrayList<>();

        for (TrainingData data : trainingData) {
            float[] activations = data.getInput();
            for (Layer layer : layers)
                activations = layer.forwardPropagation(activations);

            outputs.add(new TrainingData(data.getOutput(), activations));
        }

        return outputs;
    }

    public List<Layer> getLayers() {
        return layers;
    }
}
