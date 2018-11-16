package tesis.fraudDetection;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Array;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

/**
 * Created by carlitos on 08/09/18
 */



public class ExtractFields {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[2]").set("spark.hadoop.validateOutputSpecs", "false");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = null;
        if (args != null && args.length > 0) {
            path = args[0];
        } else {
            path = "datasets/llamadas_062018.csv";
        }
        int[] columsTypeString = {2, 4, 11, 12, 13, 15, 20, 21};
        int[] columsTypeDate = {5, 8, 9};
//        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
        JavaRDD<String> inputData = null;
        File file = new File(path);

        inputData = jsc.textFile(file.getPath()).cache();
        // Se parten los rdd's y los que no tienen 22 columnas se obvian
        JavaRDD<String> inputStringType = inputData.filter(line -> line.split(",").length == 22).map(line -> {
            String[] parts = line.split(",");
            // result es el string formado por todas las columnas de tipo string en el dataset original
            StringBuilder result = new StringBuilder();
            for (int i : columsTypeString) {
                if (result.length() > 0)
                    result.append(",");
                result.append(parts[i]);
            }
            return result.toString();
        }).cache();

        // Se forma un Array con todos los diferentes strings encontrados que se van a mapear a un número
        ArrayList<List<String>> replacements = new ArrayList<>();
        for (int j = 0; j < columsTypeString.length; j++) {
            int finalJ = j;
            // Se recorre el rdd generado anteriormente que solo contiene las columnas de tipo string
            JavaRDD<String> rddTemp = inputStringType.map(line -> {
                String[] temp = line.split(",");
                return temp[finalJ];
            }).cache();

            // Se libera de la memoria el rdd inputStringType
            inputStringType.unpersist();

            // El rddTemp contiene los strings que no se repiten, eso se devuelve como lista y se almacena en replacements
            replacements.add(rddTemp.distinct().collect());

            // Se libera de la memoria el rdd rddTemp
            rddTemp.unpersist();
        }
        try {

        int d = 0;
        for (List<String> list : replacements) {
            FileWriter writer = null;
            writer = new FileWriter("datasets/columnas" + d);
            int i = 0;
            for (String dato : list) {
                writer.write(dato + "," + i + "\n");
                i++;
            }
            writer.close();
            d++;
        }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Se recorre el dataset para sustituir las columnas de tipo string por el id generado anteriormente y
        // también sustituir las columnas de tipo Date. De vuelta se ignoran las filas que no tienen todos los campos (22)
        JavaRDD<String> rddFinal = inputData.filter(line -> line.split(",").length == 22).map(line -> {
            //Se sustituye cada string contenido en replacements con su correspondiente id que es un number
            for (List<String> list : replacements) {
                Integer i = 0;

                for (String dato : list) {
                    line = line.replaceAll("^" + dato + ",", i.toString() + ",")
                            .replaceAll("," + dato + ",", "," + i.toString() + ",")
                            .replaceAll("," + dato + "$", "," + i.toString());
                    i++;
                }
            }
            // line ya es equivalente a una fila sin columnas con tipos de datos de string.
            String[] parts = line.split(",");
            StringBuilder result = new StringBuilder();
            SimpleDateFormat formatDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
            Date date;
            Timestamp timestamp;
            //Convertir las fechas a su equivalente en timestamp (long)
            try {
                for (int i : columsTypeDate) {
                    date = formatDate.parse(parts[i]);
                    timestamp = new Timestamp(date.getTime());
                    parts[i] = String.valueOf(timestamp.getTime());
                }
            } catch (Exception e) {
                System.out.println(line);
            }
            // result contiene las filas con todos las columnas con tipos de datos númericos.
            result.append(String.join(",", parts));
            return result.toString();
        });
        // Se libera de la memoria el rdd inputData
        inputData.unpersist();
        rddFinal.coalesce(1).saveAsTextFile("tmp/result");
        jsc.stop();

    }



}
