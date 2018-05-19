package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instance;
import weka.core.Instances;

public class SVM {
	public SMO m_smo;

	public SVM() {
		this.m_smo = new SMO();
	}
	
	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}
	
	public int[] calcConfusion(Instances instances) throws Exception {

        int [] confusions = new int[4]; // [TP, FP, TN, FN]

        for (int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            Confusion confusion = getConfusion(this.m_smo.classifyInstance(instance), instance.classValue());
            confusions[confusion.ordinal()]++;
        }

        return confusions;

    }

	private Confusion getConfusion(double predicted, double actual) {
        if(predicted == 1.0) {
            if(actual == 1.0) {
                return Confusion.TN;
            }
            else {
                return Confusion.FN;
            }
        }
        else {
            if(actual == 1.0) {
                return Confusion.FP;
            }
            else {
                return Confusion.TP;
            }
        }
    }


	public void setKernel(Kernel kernel) {
		this.m_smo.setKernel(kernel);
	}

	public void setC(double value) {
		this.m_smo.setC(value);
	}

	public double getC() {
		return this.m_smo.getC();
	}

	public enum Confusion {
		TP,
		FP,
		TN,
		FN
	}

}
