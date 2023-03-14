package org.nathanael.sensei;

import org.junit.jupiter.api.Test;
import org.nathanael.sensei.activations.LeakyReLu;
import org.nathanael.sensei.activations.ReLu;
import org.nathanael.sensei.activations.Softmax;
import org.nathanael.sensei.dataset.CSV;
import org.nathanael.sensei.dataset.Dataset;
import org.nathanael.sensei.dataset.Normalization;
import org.nathanael.sensei.initialization.HeInitial;
import org.nathanael.sensei.initialization.NormalizedXavier;
import org.nathanael.sensei.loss.BinaryCrossEntropy;
import org.nathanael.sensei.optimizers.Adam;
import org.nathanael.sensei.store.JsonStorage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class SenseiTest {

    @Test
    void testModel() throws Exception {
        CSV csv = new CSV();
        Dataset dataset = csv.loadDataset("E:\\ML\\datasets\\diabetes2.csv");

        HashMap<String, List<Float>> normalizeData = new HashMap<>();
        Normalization norm = new Normalization();

        for (String column : dataset.getColumns()) {
            List<Float> values = new ArrayList<>();

            for (List<String> row : dataset.getRows()) {
                Float value = Float.parseFloat(row.get(dataset.getColumns().indexOf(column)));
                values.add(value);
            }

            if (column.equals("Outcome")) normalizeData.put(column, values);
            else normalizeData.put(column, norm.normalizeColumn(values));
        }

        List<TrainingData> trainingData = new ArrayList<>();

        for (int x = 0; x < dataset.getRows().size(); x++) {
            float[] input = new float[8];
            float[] output = new float[2];

            for (int y = 0; y < dataset.getColumns().size(); y++) {
                if (y == 8) {
                    if (normalizeData.get(dataset.getColumns().get(y)).get(x) > 0) output[0] = 1.0f;
                    else output[1] = 1.0f;
                }
                else input[y] = normalizeData.get(dataset.getColumns().get(y)).get(x);
            }

            trainingData.add(new TrainingData(input, output));
        }

        float learningRate = 0.01f;

        List<Layer> layers = new ArrayList<>();
        layers.add(new Layer(8, 4, new ReLu(), new HeInitial(), new Adam(learningRate, 8, 4)));
        layers.add(new Layer(4, 4, new ReLu(), new HeInitial(), new Adam(learningRate, 4, 4)));
        layers.add(new Layer(4, 2, new Softmax(), new NormalizedXavier(), new Adam(learningRate, 4, 2)));

        JsonStorage storage = new JsonStorage();
        Sensei nnet = new Sensei(layers, new BinaryCrossEntropy());
//        Sensei nnet = storage.loadModel("E:\\ML\\datasets\\sample.json", 0.02f);
        nnet.trainModel(trainingData.subList(0, 600), 100, 500);

        List<TrainingData> data = nnet.runModel(trainingData.subList(600, 768));
        int correct = 0;
        for (TrainingData d : data) {
            if (d.getInput()[0] > d.getInput()[1] && d.getOutput()[0] > d.getOutput()[1]) correct++;
            else if (d.getInput()[0] < d.getInput()[1] && d.getOutput()[0] < d.getOutput()[1]) correct++;
        }
        System.out.println(correct);

        storage.saveModel(nnet, "E:\\ML\\datasets\\sample.json");
    }

    @Test
    void sevenSegmentDisplay() throws Exception {
        float[][] input = {
                { 1, 1, 1, 1, 1, 1, 0 },
                { 0, 0, 0, 0, 1, 1, 0 },
                { 1, 0, 1, 1, 0, 1, 1 },
                { 1, 0, 0, 1, 1, 1, 1 },
                { 0, 1, 0, 0, 1, 1, 1 },
                { 1, 1, 0, 1, 1, 0, 1 },
                { 1, 1, 1, 1, 1, 0, 1 },
                { 1, 0, 0, 0, 1, 1, 0 },
                { 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 0, 1, 1, 1, 1 }
        };

        float[][] output = {
                { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }
        };

        List<TrainingData> trainingData = new ArrayList<>();
        for (int x = 0; x < 10; x++) {
            trainingData.add(new TrainingData(input[x], output[x]));
        }

        float learningRate = 0.02f;

        List<Layer> layers = new ArrayList<>();
        layers.add(new Layer(7, 8, new LeakyReLu(), new HeInitial(), new Adam(learningRate, 7, 8)));
        layers.add(new Layer(8, 8, new LeakyReLu(), new HeInitial(), new Adam(learningRate, 8, 8)));
        layers.add(new Layer(8, 10, new Softmax(), new NormalizedXavier(), new Adam(learningRate, 8, 10)));

        Sensei model = new Sensei(layers, new BinaryCrossEntropy());
        model.trainModel(trainingData, 1000);

        List<TrainingData> samples = model.runModel(trainingData);

        int correct = 0;
        for (TrainingData data : samples) {
            if (findHighest(data.getInput()) == findHighest(data.getOutput()))
                correct++;
        }

        for (TrainingData data : samples) {
            System.out.println(Arrays.toString(data.getInput()));
            System.out.println(Arrays.toString(data.getOutput()));
        }

        System.out.println("Score: " + correct);
    }

    public static int findHighest(float[] output) {
        int index = 0;
        float highest = 0;

        for (int x = 0; x < output.length; x++) {
            if (output[x] > highest) {
                highest = output[x];
                index = x;
            }
        }

        return index;
    }
}
