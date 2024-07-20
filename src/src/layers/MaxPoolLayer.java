package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    //Giving the step size and the window size
    private int _stepSize;
    private int _windowSize;

    //Input length
    private int _inLength;
    private int _inRows;
    private int _inCols;

    //To store the location of the max values
    List <int[][]> _lastMaxRow;
    List <int[][]> _lastMaxCol;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }

    //Max pool forward pass
    public List<double[][]> maxPoolForwardPass(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();

        //Should initialize the lastMaxRow and lastMaxCol
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();


        for(int l=0;l < input.size();l++){
            //adding the result of each pool to the output
            output.add(pool(input.get(l)));
        }


        return output;
    }

    //Pooling method - Getting max values from one image
    public double[][] pool(double[][] input){

        //making the output of the pool
        double output[][] = new double[getOutputRows()][getOutputCols()];
        //Output size is calculate by the getOutputElement()

        //Tracking the max value cords
        int [][] maxRows = new int[getOutputRows()][getOutputCols()];
        int [][] maxCols = new int[getOutputRows()][getOutputCols()];



        //Incrementing it with the step size
        for(int r =0 ; r<getOutputRows(); r+= _stepSize){
            for(int c =0 ; c<getOutputCols(); c+=_stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                //getting the max value
                for(int x =0; x< _windowSize; x++){
                    for(int y =0 ; y< _windowSize; y++){
                        //Checking the maximum value of the window
                        if(max < input[r+x][c+y]) {
                            max = input[r+x][c+y];
                            //Updating the max cords to locate
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }

                output[r][c] = max;
            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);

        return output;

    }

    //For back propagation to locate the max value , creating the two arrays to store the rows and cols of the max value


    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);

        //Should have a next layer as the last layer is the fully connected layer.
         return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input,_inLength,_inRows,_inCols);
        //Passing to the getOutput()
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        //Converting vector to matrix
        List<double[][]> matrixList = vectorToMatrix(dLdO ,getOutputLength(),getOutputRows(),getOutputCols());
        //Passing to the backpropagation()
        backPropagation(matrixList);

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        //finding the dX/DL of the previous layer
        List <double[][]> dXdL = new ArrayList<>();

        int l =0;
        for(double[][] array :dLdO){
            double[][] error = new double[_inRows][_inCols];

            for(int r =0; r< getOutputRows();r++){
                for(int c =0; c< getOutputCols();c++){
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if(max_i != -1){
                        //if not the default value adding to the previous array
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }
            //Adding to the dxdl
            dXdL.add(error);
            l++;
        }

        //Passing to the previous layer
        if(_previousLayer != null){
            _previousLayer.backPropagation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    //To get the size of the output to the pool layer
    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize)/_stepSize +1 ;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _windowSize)/_stepSize +1 ;
    }

    @Override
    public int getOutputElements() {
        return _inRows * getOutputCols() *getOutputRows();
    }
}
