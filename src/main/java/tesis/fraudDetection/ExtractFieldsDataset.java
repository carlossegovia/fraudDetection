package tesis.fraudDetection;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by carlitos on 08/09/18
 */


public class ExtractFieldsDataset {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("KmeasExample").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = null;
        if (args != null && args.length > 0) {
            path = args[0];
        } else {
            path = "src/main/resources/llamadasPrueba";
        }
        int[] columsTypeString = {2, 4, 11, 12, 13, 15, 20, 21};
        int[] columsTypeDate = {5, 8, 9};

        JavaRDD<String> inputData = jsc.textFile(path);
        JavaRDD<String> inputParseado = inputData.map(line -> {
            String[] parts = line.split(",");
            StringBuilder result = new StringBuilder();
            SimpleDateFormat formatDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
            //Convertir las fechas a su equivalente en timestamp (long)
            for (int i : columsTypeDate) {
                Date date = formatDate.parse(parts[i]);
                Timestamp timestamp = new Timestamp(date.getTime());
                parts[i] = String.valueOf(timestamp.getTime());
            }

            for (int j: columsTypeString) {
                // Esta función de hash debería de optimizarse (retorna por ejemplo un hash negativo)
                parts[j] = String.valueOf(parts[j].hashCode());
            }
            result.append(String.join(",", parts));
            return result.toString();
        });

        inputParseado.coalesce(1).saveAsTextFile("src/main/resources/tmp/prueba");
        jsc.stop();

    }


}
