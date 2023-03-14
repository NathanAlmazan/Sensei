package org.nathanael.sensei.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CSV implements DatasetLoader {

    private static final String COMMA_DELIMITER = ",";

    @Override
    public Dataset loadDataset(String location) throws Exception {
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(location))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                records.add(Arrays.asList(values));
            }
        }

        List<String> header = records.remove(0);

        for (int x = 0; x < records.size(); x++) {
            List<String> row = records.get(x);

            for (String value : row) {
                if (value.length() < 1) records.remove(x);
            }
        }

        for (int x = 0; x < records.size(); x++) {
            List<String> row = records.get(x);

            for (String value : row) {
                if (value.length() < 1) records.remove(x);
            }
        }

        return new Dataset(header, records);
    }
}
