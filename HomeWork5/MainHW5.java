package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import javafx.util.Pair;
import weka.core.Instances;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

class KernelResult {
	private Kernel kernelType;
	private double kernelParameter;
	private double power;
	private double TPR;
	private double FPR;

	public KernelResult(double kernelParameter, double power, double TPR, double FPR, Kernel kernelType) {
		this.kernelType = kernelType;
		this.kernelParameter = kernelParameter;
		this.power = power;
		this.TPR = TPR;
		this.FPR = FPR;
	}

	public Kernel getKernelType() {
		return kernelType;
	}

	public void setKernelType(Kernel kernelType) {
		this.kernelType = kernelType;
	}

	public double getPower() {
		return power;
	}

	public void setPower(double power) {
		this.power = power;
	}

	public double getKernelParameter() {
		return kernelParameter;
	}

	public void setKernelParameter(double kernelParameter) {
		this.kernelParameter = kernelParameter;
	}

	public double getTPR() {
		return TPR;
	}

	public void setTPR(double TPR) {
		this.TPR = TPR;
	}

	public double getFPR() {
		return FPR;
	}

	public void setFPR(double FPR) {
		this.FPR = FPR;
	}

	public enum Kernel {
		RBF,
		POLY
	}
}

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

		Instances data = loadData("HomeWork5/Data/cancer.arff");
		Instances trainingData = data.trainCV(5, 1);
		Instances testData = data.testCV(5, 1);

		SVM svm = new SVM();
		double alpha = 1.5;

		KernelResult maxPoly = polyKernelMaximize(svm, trainingData, testData, alpha);
		KernelResult maxRBF = rbfKernelMaximize(svm, trainingData, testData, alpha);

		if (maxPoly.getPower() > maxRBF.getPower()) {
			System.out.println("The best kernel is PolyKernel with degree " + maxPoly.getKernelParameter()
					+ " rates: " + maxPoly.getTPR() + " / " + maxPoly.getFPR());
			slackValueMaximize(svm, maxPoly, trainingData, testData);
		} else {
			System.out.println("The best kernel is RBFKernel with gamma " + maxRBF.getKernelParameter()
					+ " rates : " + maxRBF.getTPR() + " / " + maxPoly.getFPR());
			slackValueMaximize(svm, maxRBF, trainingData, testData);
		}

	}

	private static void slackValueMaximize(SVM svm, KernelResult bestKernel, Instances trainingData, Instances testData) throws Exception{
		int[] confusions;
		double TPR = 0.0;
		double FPR = 0.0;
		KernelResult.Kernel kernelType = bestKernel.getKernelType();
		if(kernelType == KernelResult.Kernel.POLY) {
			PolyKernel polyKernel = new PolyKernel();
			polyKernel.setExponent(bestKernel.getKernelParameter());
			svm.setKernel(polyKernel);
		}
		else {
			RBFKernel rbfKernel = new RBFKernel();
			rbfKernel.setGamma(bestKernel.getKernelParameter());
			svm.setKernel(rbfKernel);
		}

		int[] exponentValues = new int[] {1, 0 , -1, -2, -3, -4};
		int[] linearValues = new int[] {3, 2 ,1 };

		for (int i : exponentValues) {
			for (int j: linearValues) {
				double c = Math.pow(10, i) * ((double) j / 3);
				svm.setC(c);
				svm.buildClassifier(trainingData);
				confusions = svm.calcConfusion(testData);
				TPR = getTPR(confusions);
				FPR = getFPR(confusions);
				System.out.println("For C " + c + " the rates are: ");
				printRates(TPR, FPR);
				System.out.println();
			}
		}
	}

	private static KernelResult rbfKernelMaximize(SVM svm, Instances trainingData, Instances testData, double alpha) throws Exception {
		int[] confusions;
		double TPR = 0.0;
		double FPR = 0.0;
		double maxGamma = 0.0;
		double maxPower = 0.0;

		RBFKernel rbfKernel = new RBFKernel();

		double[] gammaValues = new double[]{0.005, 0.05, 0.5};


		for (double gamma : gammaValues) {
			rbfKernel.setGamma(gamma);
			svm.setKernel(rbfKernel);
			svm.buildClassifier(trainingData);

			confusions = svm.calcConfusion(testData);
			TPR = getTPR(confusions);
			FPR = getFPR(confusions);

			System.out.println("For RBFKernel with gamma " + gamma + " the rates are: ");
			printRates(TPR, FPR);
			System.out.println();

			double power = alpha * TPR - FPR;
			if (power > maxPower) {
				maxPower = power;
				maxGamma = gamma;
			}
		}

		return new KernelResult(maxGamma, maxPower, TPR, FPR, KernelResult.Kernel.RBF);
	}

	private static KernelResult polyKernelMaximize(SVM svm, Instances trainingData, Instances testData, double alpha) throws Exception {
		int[] confusions;
		double TPR = 0.0;
		double FPR = 0.0;
		double maxDeg = 0;
		double maxPower = 0.0;

		PolyKernel polyKernel = new PolyKernel();
		int[] degValues = new int[]{2, 3, 4};

		for (int deg : degValues) {
			polyKernel.setExponent(deg);
			svm.setKernel(polyKernel);
			svm.buildClassifier(trainingData);

			confusions = svm.calcConfusion(testData);
			TPR = getTPR(confusions);
			FPR = getFPR(confusions);

			System.out.println("For PolyKernel with degree " + deg + " the rates are: ");
			printRates(TPR, FPR);
			System.out.println();
			double power = alpha * TPR - FPR;
			if (power > maxPower) {
				maxPower = power;
				maxDeg = (double) deg;
			}
		}

		return new KernelResult(maxDeg, maxPower, TPR, FPR, KernelResult.Kernel.POLY);

	}

	private static double getTPR(int[] confusions) {
		int tp = confusions[SVM.Confusion.TP.ordinal()];
		int fn = confusions[SVM.Confusion.FN.ordinal()];
		return (double) tp / (tp + fn);
	}

	private static double getFPR(int[] confusions) {
		int fp = confusions[SVM.Confusion.FP.ordinal()];
		int tn = confusions[SVM.Confusion.TN.ordinal()];
		return (double) fp / (fp + tn);
	}

	private static void printRates(double TPR, double FPR) {
		System.out.println("TPR = " + TPR);
		System.out.println("FPR = " + FPR);
	}
}
