package ca.queensu;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class AlzheimersML {
	public static void main(String[] args){
		SparkConf conf = new SparkConf().setAppName("AlzheimersML");
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	    
	    JavaRDD<String> csvFile = jsc.textFile("hdfs://bi-hadoop-prod-4157.bi.services.us-south.bluemix.net:8020/tmp/data_transposed.csv");
	    
	    final String header = csvFile.first();
	    JavaRDD<String> csvFileWithoutHeader = csvFile.filter(new Function<String, Boolean>(){
	    	
	    	private static final long serialVersionUID = 1L;
	    	
	    	@Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header);
            }
	    });
	    JavaRDD<Vector> mat = csvFileWithoutHeader.map(mapFunction);
	    
	    mat.cache();
	    
	    AlzheimersClustering alzKmeans = new AlzheimersClustering(2, 20, mat);
	    
	    JavaRDD<String> predictedResults = alzKmeans.getPredictedResults().map(new Function<Integer, String>(){
	    	private static final long serialVersionUID = 1L;
	    	
	    	@Override
	    	public String call(Integer i){
	    		return Integer.toString(i);
	    	}
	    });
	    
	    JavaRDD<String> predictedMatrix = csvFileWithoutHeader.union(predictedResults);
	    
	    JavaRDD<LabeledPoint> labeledmat = predictedMatrix.map(labeledMapFunction);
	    
	    AlzheimersClassification alzLogistic = new AlzheimersClassification(labeledmat);
	    
	    jsc.stop();
	    jsc.close();
	}
	
	static Function<String, Vector> mapFunction = 
	      new Function<String, Vector>() {
			private static final long serialVersionUID = 1L;
	        public Vector call(String s) {
	            String[] sarray = s.split(" ");
	            double[] values = new double[sarray.length - 1]; 
	            // Ignore 1st token, it is survival months and not needed here.
	            for (int i = 0; i < sarray.length - 1; i++)
	                values[i] = Double.parseDouble(sarray[i + 1]);
	            return Vectors.dense(values);
	        }
	    };
	
    static Function<String, LabeledPoint> labeledMapFunction = 
    	new Function<String, LabeledPoint>(){
	    	private static final long serialVersionUID = 1L;
			@Override
			public LabeledPoint call(String arg0) throws Exception {
	            String[] attributes = arg0.split(",");
	            
	            double[] values = new double[attributes.length];
	            for (int i = 0; i < attributes.length-1; i++) {
	        		values[i] = Double.parseDouble(attributes[i]);
	            }
	            return new LabeledPoint(Double.parseDouble(attributes[attributes.length-1]), Vectors.dense(values));  
			}
	    	
	    };

}
