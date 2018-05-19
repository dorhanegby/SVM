package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

public class MainHW5 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {

		Instances trainingData = loadData("HomeWork5/Data/training.arff");
		Instances testData = loadData("HomeWork5/Data/test.arff");

		SVM svm = new SVM();

		PolyKernel polyKernel = new PolyKernel();
		polyKernel.setExponent(2);
		svm.setKernel(polyKernel);
		svm.buildClassifier(trainingData);
		svm.calcConfusion(testData);

	}
}
