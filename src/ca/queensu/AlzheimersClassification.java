package ca.queensu;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class AlzheimersClassification {
	public static void main(String[] args){
		SparkConf conf = new SparkConf().setAppName("AlzheimersClassification");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		
		JavaRDD<String> csvFile =  jsc.textFile("hdfs://bi-hadoop-prod-4157.bi.services.us-south.bluemix.net:8020/tmp/data_transposed.csv");
		final String header = csvFile.first();
	    JavaRDD<String> csvFileWithoutHeader = csvFile.filter(new Function<String, Boolean>(){
	    	
	    	private static final long serialVersionUID = 1L;
	    	
	    	@Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header);
            }
	    });
	    
	    JavaRDD<LabeledPoint> labeledmat = csvFileWithoutHeader.map(new Function<String, LabeledPoint>(){

			private static final long serialVersionUID = 1L;

			@Override
			public LabeledPoint call(String arg0) throws Exception {
//				System.out.println("arg0"+arg0);
	            String[] attributes = arg0.split(",");
	            
	            double[] values = new double[attributes.length];
	            for (int i = 0; i < attributes.length-1; i++) {
	        		values[i] = Double.parseDouble(attributes[i]);
//	        		System.out.println(values[i]);
	            }
	            return new LabeledPoint(Double.parseDouble(attributes[attributes.length-1]), Vectors.dense(values));  
			}
	    	
	    });

	    // Split initial RDD into two... [70% training data, 30% testing data].
	    JavaRDD<LabeledPoint>[] splits = labeledmat.randomSplit(new double[] {0.7, 0.3}, 11L);
	    JavaRDD<LabeledPoint> training = splits[0].cache();
	    JavaRDD<LabeledPoint> test = splits[1];
	    
	    // Run training algorithm to build the model.
	    LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(4).run(training.rdd());

	    // Compute raw scores on the test set.
	    JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(record -> new Tuple2<>(model.predict(record.features()), record.label()));

	    // Get evaluation metrics.
	    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
	    double accuracy = metrics.accuracy();
	    System.out.println("\n\n\n\n\n\nAccuracy = " + accuracy);

	    // Save and load model
	    model.save(jsc.sc(), "hdfs://bi-hadoop-prod-4218.bi.services.us-south.bluemix.net:8020/tmp/Model");
	    System.out.println("Model Saved Successfully!!!"+model.toString()); 
	    LogisticRegressionModel sameModel = LogisticRegressionModel.load(jsc.sc(),"hdfs://bi-hadoop-prod-4218.bi.services.us-south.bluemix.net:8020/tmp/Model");
	    System.out.println("Model Loaded Successfully!!!"+sameModel.toString()); 
	    
	    jsc.stop();
	    jsc.close();
	}

}
