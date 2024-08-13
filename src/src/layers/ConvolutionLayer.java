package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

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
    private double _learningRate;

    //Storing the values in the forward pass
    private List<double[][]> _lastInput;

    public ConvolutionLayer(int _filterSize, int _stepSize, int _inLength, int _inRows, int _inCols,long SEED, int numFilters,double _learningRate ) {
        this.SEED = SEED;
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this._learningRate = _learningRate;

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

    /**
     * Convolution forward pass method
     * This method is going through each input layer convolve each one with filters
     * @param list :input layer
     * @return : output layer
     */
    public List<double[][]> convolutionForwardPass(List<double[][]> list){

        //Saving the last input for the backward pass
        _lastInput = list;

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

    //Method to space out the output loss matrix to use it in the convolution method
    public double[][] spaceArray(double[][] input){
        if(_stepSize == 1){
            return input;
        }

        //Size of the output
        int outRows = (input.length-1)*_stepSize+1;
        int outCols = (input[0].length-1)*_stepSize +1;

        double[][] output = new double[outRows][outCols];

        for(int i = 0; i< input.length ; i++){
            for(int j =0; j< input[0].length; j++){
                //Setting the output value : and stretching it using the stepSize
                output[i*_stepSize][j*_stepSize] = input[i][j];
            }
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
        //converting vector to matrix
        List<double[][]> matrixInput = vectorToMatrix(dLdO, _inLength,_inRows,_inCols);
        backPropagation(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        //Finding the filter loss
        //Initializing the filter matrix list
        List<double[][]> filterDelta = new ArrayList<>();

        //Making a list to keep track of the errors of the previous layer
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();

        for(int f= 0; f < _filters.size(); f++){
            //Creating new matrix for each filter update
            filterDelta.add(new double[_filterSize][_filterSize]);
        }

        //Finding each filter contributed to the loss
        for(int i =0; i <_lastInput.size(); i++){

            //Tracking the error for each input
            double[][] errorForInput = new double[_inRows][_inCols];

            for(int f=0; f<_filters.size(); f++){

                double[][] currFilter = _filters.get(f);
                //getting the current dL/dO value
                double[][] error = dLdO.get(i*_filters.size()+f);

                //Spacing out the error to pass it in to the convolution method
                double[][] spacedError = spaceArray(error);

                //Doing the convolution process
                double[][] dLdF = convolve(_lastInput.get(i),spacedError,1);

                //Updating the filters : to do that need a learning rate
                // How much we should multiply before adding or subtracting the current filter.
                //Error * learning rate and add that to the filter = new filter

                double[][] delta = multiply(dLdF,_learningRate*-1);
                //Using -1 to shrink?

                //should sum up all the errors
                double[][] newTotalDelta = add(filterDelta.get(f),delta);
                filterDelta.set(f,newTotalDelta);

                //full convolution
                //Flipping horizontally and vertically
                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));

                //Full convolution and adding it to the existing error for this input
                errorForInput  = add(errorForInput,fullConvolution(currFilter,flippedError));

            }

            //adding to the sum total across the all filters
            dLdOPreviousLayer.add(errorForInput);

        }

        //Updating the filters : learning section
        for(int f= 0; f< _filters.size(); f++){
            //new filter = current filter + total delta(error)
            double[][] modified = add(filterDelta.get(f),_filters.get(f));
            _filters.set(f,modified);
        }


        //Finally propagating to the previous layer
        if(_previousLayer != null){
            _previousLayer.backPropagation(dLdOPreviousLayer);
        }

    }

    //Creating a method to flip the matrix values horizontally and vertically to make the (dL/dO)*
    public double[][] flipArrayHorizontal(double[][] array){

        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        //flipping
        for(int i =0; i< rows; i++){
            for(int j=0; j< cols; j++){
                //Writing the array backwards
                output[rows-i-1][j] = array[i][j];
            }
        }

        return output;
    }

    public double[][] flipArrayVertical(double[][] array){

        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        //flipping
        for(int i =0; i< rows; i++){
            for(int j=0; j< cols; j++){
                //Writing the array backwards
                output[cols-j-1][j] = array[i][j];
            }
        }

        return output;
    }


    //Method for full convolution
    public double[][] fullConvolution(double[][] input, double[][] filter){
        //Should add the filter lengths
        //As the filter is sometimes bigger than the inputs
        int outRows = (input.length + filter.length) +1;
        int outCols = (input[0].length + filter[0].length)+1;

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

        //Moving the output loss layer across the filter layer
        //This should start from a negative value as the layer should pass the right bottom corner first

        for(int i = -fRows+1; i <= inRows; i++){

            outCol = 0;
            //Moving the filter down
            for(int j = -fCols +1 ; j < inCols  ; j++){

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

                        //Checking the row and the column indexing are valid
                        if(inputRowIndex >= 0 && inputColIndex >=0 && inputRowIndex < inRows && inputColIndex < inCols){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }

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
