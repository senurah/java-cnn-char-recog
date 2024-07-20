package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer{

    //To generate the filers we are using a seed
    private long SEED;

    //Filter matrix list to store filter matrices
    private List<double[][]> _filters;
    //By assuming the filter matrices are square
    private int _filterSize;
    private int _stepSize;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    public ConvolutionLayer(int _filterSize, int _stepSize, int _inLength, int _inRows, int _inCols,long SEED, int numFilters) {
        this.SEED = SEED;
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;

        generateRandomFilters(numFilters);
    }

    //method to generate filters
    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        //Using java Random num generator
        Random random = new Random(SEED);

        for(int n =0; n< numFilters; n++){
            //Storing the new Filter
            double[][] newFilter = new double[_filterSize][_filterSize];

            //Random num between -1 -- 1 on gaussian distribution
            for(int i = 0 ; i < _filterSize ;i++){
                for(int j = 0 ; j < _filterSize ;j++){

                    double value = random.nextGaussian();
                    newFilter[i][j] = value;

                }

            }

            filters.add(newFilter);

        }
        _filters = filters;
    }

    //Convolution forward pass method
    /*
        This method is going through each input layer convolve each one with filters
    * */
    public List<double[][]> convolutionForwardPass(List<double[][]> list){
        List<double[][]> output = new ArrayList<>();

        //Convolution process with each filter
        for(int n=0; n< list.size();n++){
            for(double[][] filter : _filters){
                output.add(convolve(list.get(n), filter ,_stepSize));
            }


        }

        return output;

    }

    public double[][] convolve(double[][] input, double[][] filter, int stepSize){
        //Dimensions of the output matrix
        int outRows = (input.length - filter.length)/stepSize +1;
        int outCols = (input[0].length - filter[0].length)/ stepSize + 1;

        //Dimensions of the input matrix
        int inRows = input.length;
        int inCols = input[0].length;

        //Dimensions of the filter matrix
        int fRows = filter.length;
        int fCols = filter[0].length;

        //Creating an output matrix
        double[][] output = new double[outRows][outCols];

        //To keep track of the output matrix coords
        int outRow = 0;
        int outCol;

        //Moving the filter across the input layer
        for(int i = 0; i <= inRows- fRows; i+= stepSize){

            outCol = 0;
            //Moving the filter down in the input layer
            for(int j = 0; j <= inCols -fCols ; j+= stepSize){

                //Summing the multiplied values
                double sum = 0.0;

                //This position indicates the upper right corner of the filter applied on the input layer
                //Should apply other positions as a filter around this spot

                //Applying the filter around this position
                for(int x =0; x< fRows ; x++){
                    for(int y=0; y< fCols; y++){
                        //Now should multiply the input value and the filter value
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum += value;


                    }
                }
                //Putting the values to the output matrix
                output[outRow][outCol] = sum;
                //Incrementing the output col
                outCol++;

            }
            //Incrementing the output row
            outRow++;

        }

        return output;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);

        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        //Converting vector to matrix
        List<double[][]> matrixInput = vectorToMatrix(input, _inLength,_inRows,_inCols);

        return  getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] dLdO) {

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

    }

    @Override
    public int getOutputLength() {
        //Number of inputs * filters
        return _filters.size()*_inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows-_filterSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols-_filterSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        //OutputRows * OutputCols * OutputLength
        return getOutputLength()*getOutputRows()*getOutputCols();
    }
}
