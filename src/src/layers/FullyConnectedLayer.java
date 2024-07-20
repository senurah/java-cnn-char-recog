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
    //To keep track of the z value
    private double[] lastZ;
    //addressing the value changing error in the derivative ReLu method
    private final double leak = 0.01;
    //Keeping track of the last input
    private double[] lastX;
    //Learning rate
    private double _learningRate;


    public FullyConnectedLayer(int _inLength, int _outLength,long SEED,double _learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = _learningRate;

        //Setting the weights array
        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    //forward pass in the fullyConnected layer
    public double[] fullyConnectedForwardPass(double[] input){
        double[] z = new double[_outLength];
        double[] out= new double[_outLength];

        // Debugging: Print the lengths of input and weights
        System.out.println("Input length: " + input.length);
        System.out.println("Weights dimensions: " + _weights.length + "x" + _weights[0].length);

        //Keeping track of the last input
        lastX = input;

        //Moving through each input nodes
        for(int i = 0; i <_inLength ; i++){
            for(int j = 0; j <_outLength;j++){
                //calculation - multiplying them with the weight
                z[j] += input[i] * _weights[i][j];
            }
        }

        //Storing the last z in order to use in the back Propagation
        lastZ = z;

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
        //converting matrix to vector
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

        double[] dLdX = new double[_inLength];

        //Derivative of the ReLu function
        double dOdz;
        //Xi = the last input
        double dzdw;
        //weight loss per each weight
        double dLdw;
        // dLdw = dLdO * dOdz * dzdw;
        //For back propagation
        double dzdx;

        //setting the values
        for(int k=0; k< _inLength ; k++){

            //In order to find the error in the previous layer for passing the backPropagation
            double dLdX_sum = 0;

            for(int j=0; j <_outLength; j++){

                //Should input the last z value to the derivativeReLu method
                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                //Chain rule
                dLdw = dLdO[j] * dOdz * dzdw;

                //Multiplying it by the learning rate and updating the weight
                _weights[k][j] -= dLdw*_learningRate;

                //Chain rule for finding the error in the previous layer
                dLdX_sum += dLdO[j]*dOdz* dzdx;

            }

            dLdX[k] = dLdX_sum;
        }


        //Passing to the previous layer
        if(_previousLayer != null){
            _previousLayer.backPropagation(dLdX);
        }

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        //covering matrix to vector
        double[] vector = matrixToVector(dLdO);
        //then back propagation
        backPropagation(vector);


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
        return _outLength;
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

    //Derivative of the ReLu - dOdz
    public double derivativeReLu(double input){
        if(input <=0){
            //without returning 0 as it causes dead area returning the leak
            return leak;
        }else{
            return 1;
        }
    }


}
