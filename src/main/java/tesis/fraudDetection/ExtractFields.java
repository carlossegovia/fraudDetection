package tesis.fraudDetection;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Array;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by carlitos on 08/09/18
 */



public class ExtractFields {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[2]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = null;
        if (args != null && args.length > 0) {
            path = args[0];
        } else {
            path = "datasets/kddcup.data";
        }
        int[] columnas = {1,2,3,41};
//        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
        JavaRDD<String> inputData = null;
        File file = new File(path);

        inputData = jsc.textFile(file.getPath());
        JavaRDD<String> inputParseado = inputData.map( line -> {
            String[] parts = line.split(",");
            StringBuilder result = new StringBuilder();
            for (int i: columnas){
                if(result.length() > 0)
                    result.append(",");
                result.append(parts[i]);
            }
            return result.toString();
        });
        ArrayList<List<String>> replacements = new ArrayList<>();
        for (int j=0; j<columnas.length; j++){
            int finalJ = j;
            JavaRDD<String> rddTemp = inputParseado.map(line -> {
                String[] temp = line.split(",");
                return temp[finalJ];
            });
            JavaRDD<String> rddDistinct = rddTemp.distinct();
            replacements.add(rddDistinct.collect());
        }
        JavaRDD<String> rddFinal = inputData.map(line -> {
            for (List<String> list: replacements){
                Integer i = 0;
                for(String dato: list){
                    line = line.replaceAll(dato+ "(?!_)", i.toString());
                    i++;
                }
            }
            return line;
        });
        rddFinal.coalesce(1).saveAsTextFile("tmp/result");
        jsc.stop();

    }



}
