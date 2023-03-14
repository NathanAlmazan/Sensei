package org.nathanael.sensei.store;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.nathanael.sensei.Layer;
import org.nathanael.sensei.Sensei;
import org.nathanael.sensei.activations.Activation;
import org.nathanael.sensei.loss.LossFunction;
import org.nathanael.sensei.optimizers.Optimizer;
import org.nathanael.sensei.optimizers.OptimizerOption;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class JsonStorage implements ModelStorage {

    @Override
    public Sensei loadModel(String location, Float learningRate) throws Exception {
        JSONParser parser = new JSONParser();

        FileReader reader = new FileReader(location);
        JSONObject model = (JSONObject)  parser.parse(reader);

        LossFunction lossFunction = StringToObject.LOSS_FUNCTION.get((String) model.get("lossFunction"));

        JSONArray listOfLayers = (JSONArray) model.get("layers");
        List<Layer> layers = new ArrayList<>();

        for (Object layerObj : listOfLayers) {
            JSONObject layer = (JSONObject) layerObj;

            JSONObject optimizerObj = (JSONObject) layer.get("optimizer");
            String type = (String) optimizerObj.get("type");
            Long inputSize = (Long) optimizerObj.get("inputSize");
            Long outputSize = (Long) optimizerObj.get("outputSize");
            Double savedRate = (Double) optimizerObj.get("learningRate");

            float lr = savedRate.floatValue();
            if (learningRate != null) lr = learningRate;

            Optimizer optimizer = OptimizerOption.getOptimizer(type, lr, inputSize.intValue(), outputSize.intValue());

            JSONArray neurons = (JSONArray) layer.get("weights");
            float[][] layerWeights = new float[outputSize.intValue()][inputSize.intValue()];

            for (int i = 0; i < outputSize.intValue(); i++) {
                JSONArray weights = (JSONArray) neurons.get(i);
                float[] layerWeight = new float[inputSize.intValue()];

                for (int j = 0; j < inputSize.intValue(); j++)
                    layerWeight[j] = ((Double) weights.get(j)).floatValue();

                layerWeights[i] = layerWeight;
            }

            JSONArray biases = (JSONArray) layer.get("biases");
            float[] layerBiases = new float[outputSize.intValue()];

            for (int i = 0; i < outputSize.intValue(); i++)
                layerBiases[i] = ((Double) biases.get(i)).floatValue();

            Activation activation = StringToObject.ACTIVATION.get((String) layer.get("activation"));

            layers.add(new Layer(layerWeights, layerBiases, activation, optimizer));
        }


        return new Sensei(layers, lossFunction);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void saveModel(Sensei model, String location) throws Exception {
        JSONObject modelObj = new JSONObject();

        modelObj.put("lossFunction", model.getLossFunction().getClass().getSimpleName());

        JSONArray modelLayers = new JSONArray();
        for (Layer layer : model.getLayers()) {
            JSONObject layerObj = new JSONObject();

            // save weights
            JSONArray weights = new JSONArray();
            for (float[] weight : layer.getWeights()) {
                JSONArray neuron = new JSONArray();
                neuron.addAll(floatArrayToList(weight));

                weights.add(neuron);
            }
            layerObj.put("weights", weights);

            // save biases
            JSONArray biases = new JSONArray();
            biases.addAll(floatArrayToList(layer.getBiases()));
            layerObj.put("biases", biases);

            JSONObject optimizerObj = new JSONObject();
            optimizerObj.put("type", layer.getOptimizer().getClass().getSimpleName());
            optimizerObj.put("inputSize", layer.getInputSize());
            optimizerObj.put("outputSize", layer.getOutputSize());
            optimizerObj.put("learningRate", layer.getOptimizer().getAlpha());

            layerObj.put("activation", layer.getActivation().getClass().getSimpleName());
            layerObj.put("optimizer", optimizerObj);

            modelLayers.add(layerObj);
        }

        modelObj.put("layers", modelLayers);

        // save model
        FileWriter file = new FileWriter(location);
        file.write(modelObj.toJSONString());
        file.flush();
    }

    private List<Float> floatArrayToList(float[] array) {
        List<Float> list = new ArrayList<>();

        for (float num : array) list.add(num);

        return list;
    }
}
