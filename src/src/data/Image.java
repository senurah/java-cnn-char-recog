package data;

public class Image {

    private double[][] data;
    private int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    //Testing
    public String toString(){

        String s = label + ", \n";

        for(int i = 0; i<data.length;i++){
            for(int j=0;j<data[0].length;j++){
                s+= data[i][j]+",";
            }
            s+= "\n";
        }

        return s;
    }

}
