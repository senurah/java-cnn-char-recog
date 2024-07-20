package layers;

import java.util.List;

public class MaxPoolLayer extends Layer{

    //Giving the step size and the window size
    private int _stepSize;
    private int _windowSize;

    //Input length
    private int _inLength;
    private int inRows;
    private int _inCols;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this.inRows = inRows;
        this._inCols = _inCols;
    }

    //Max pool forward pass
    //public List<double[][]> maxPoolForwardPass(List<double[][]> input){}


    @Override
    public double[] getOutput(List<double[][]> input) {
        return new double[0];
    }

    @Override
    public double[] getOutput(double[] input) {
        return new double[0];
    }

    @Override
    public void backPropagation(double[] dLdO) {

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return 0;
    }
}
