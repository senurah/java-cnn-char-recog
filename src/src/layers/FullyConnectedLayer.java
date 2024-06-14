package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{

    //Adding the weight and the RELU function:
    private double[][] _weights;
    //to track the input length and the output length
    private int _inLength;
    private int _outLength;
    //using Random SEEDs to give random values to the weights
    private long SEED;


    public FullyConnectedLayer(int _inLength, int _outLength,long SEED) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;

        //Setting the weights array
        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    //forward pass in the fullyConnected layer
    public double[] fullyConnectedForwardPass(double[] input){
        double[] z = new double[_outLength];
        double[] out= new double[_outLength];

        //Moving through each input nodes
        for(int i = 0; i<_inLength ; i++){
            for(int j =0; j<_outLength;j++){
                //calculation - multiplying them with the weight
                z[j] += input[i] * _weights[i][j];
            }
        }

        //Running the result through the activation function
        for(int i = 0; i<_inLength ; i++){
            for(int j =0; j<_outLength;j++){
                //Activating the RELU function
                out[j] = reLu(z[j]);
            }
        }

        return out;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        //converting
        double[] vector = matrixToVector(input);
        return  getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {

        //Taking the forward pass results
        double[]  forwardPass = fullyConnectedForwardPass(input);

        if(_nextLayer != null){
            //passing to the forward pass to get the output
            return _nextLayer.getOutput(forwardPass);
        } else {
            //There is no node to pass , therefor returning the final forward pass
            return forwardPass;
        }

    }

    //back propagation build

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

    public void setRandomWeights(){
        //Using the random number generator
        Random random = new Random(SEED);

        for(int i =0; i< _inLength ;i++){
            for(int j = 0; j < _outLength;j++){
                //Gaussian - Numbers which are equally distributed around zero
                _weights[i][j] = random.nextGaussian();
            }
        }
    }

    //RELU activation method
    public double reLu(double input){
        if(input <=0){
            return 0;
        }else{
            return input;
        }
    }


}
