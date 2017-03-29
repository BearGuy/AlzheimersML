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
	JavaRDD<LabeledPoint> labeledMat;
	double accuracy;
	
	
	public AlzheimersClassification(JavaRDD<LabeledPoint> matLabeled){
		this.labeledMat = matLabeled;
		this.accuracy = calculateAccuracy(labeledMat);
	}
	
	public double getAccuracy() {
		return this.accuracy;
	}
	
	public double calculateAccuracy(JavaRDD<LabeledPoint> labeledMat){

	    // Split initial RDD into two... [70% training data, 30% testing data].
	    JavaRDD<LabeledPoint>[] splits = labeledMat.randomSplit(new double[] {0.7, 0.3}, 11L);
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
	    
	    return accuracy;

	    // Save and load model
//	    model.save(jsc.sc(), "hdfs://bi-hadoop-prod-4218.bi.services.us-south.bluemix.net:8020/tmp/Model");
//	    System.out.println("Model Saved Successfully!!!"+model.toString()); 
//	    LogisticRegressionModel sameModel = LogisticRegressionModel.load(jsc.sc(),"hdfs://bi-hadoop-prod-4218.bi.services.us-south.bluemix.net:8020/tmp/Model");
//	    System.out.println("Model Loaded Successfully!!!"+sameModel.toString()); 
	    
	}

}
