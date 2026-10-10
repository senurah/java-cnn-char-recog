package cli;

import java.util.Objects;

/**
 * Immutable configuration model encapsulating all hyperparameters, dataset paths,
 * runtime limits, and CLI execution modes for the CNN character recognition system.
 */
public final class AppConfig {

    // Fixed MNIST input specifications
    public static final int INPUT_ROWS = 28;
    public static final int INPUT_COLS = 28;

    // Default configuration values
    public static final String DEFAULT_TRAIN_PATH = "data/mnist_train.csv";
    public static final String DEFAULT_TEST_PATH = "data/mnist_test.csv";
    public static final int DEFAULT_TRAIN_LIMIT = 0;       // 0 = full dataset
    public static final int DEFAULT_TEST_LIMIT = 0;        // 0 = full dataset
    public static final int DEFAULT_EPOCHS = 3;
    public static final long DEFAULT_SEED = 123L;
    public static final double DEFAULT_SCALE_FACTOR = 256.0 * 100.0;

    public static final int DEFAULT_NUM_FILTERS = 8;
    public static final int DEFAULT_FILTER_SIZE = 5;
    public static final int DEFAULT_CONV_STEP_SIZE = 1;
    public static final double DEFAULT_CONV_LEARNING_RATE = 0.1;

    public static final int DEFAULT_POOL_WINDOW_SIZE = 3;
    public static final int DEFAULT_POOL_STEP_SIZE = 2;

    public static final int DEFAULT_NUM_CLASSES = 10;
    public static final double DEFAULT_FC_LEARNING_RATE = 0.1;

    public static final boolean DEFAULT_VERBOSE = false;
    public static final boolean DEFAULT_INTERACTIVE = false;
    public static final String DEFAULT_SAVE_PATH = null;
    public static final String DEFAULT_LOAD_PATH = null;

    // Dataset & Execution settings
    private final String trainPath;
    private final String testPath;
    private final int trainLimit;
    private final int testLimit;
    private final int epochs;
    private final long seed;
    private final double scaleFactor;

    // Model Persistence settings
    private final String savePath;
    private final String loadPath;

    // Convolution Layer settings
    private final int numFilters;
    private final int filterSize;
    private final int convStepSize;
    private final double convLearningRate;

    // Max Pooling Layer settings
    private final int poolWindowSize;
    private final int poolStepSize;

    // Fully Connected Layer settings
    private final int numClasses;
    private final double fcLearningRate;

    // CLI execution flags
    private final boolean verbose;
    private final boolean interactive;

    private AppConfig(Builder builder) {
        this.trainPath = builder.trainPath;
        this.testPath = builder.testPath;
        this.trainLimit = builder.trainLimit;
        this.testLimit = builder.testLimit;
        this.epochs = builder.epochs;
        this.seed = builder.seed;
        this.scaleFactor = builder.scaleFactor;
        this.savePath = builder.savePath;
        this.loadPath = builder.loadPath;

        this.numFilters = builder.numFilters;
        this.filterSize = builder.filterSize;
        this.convStepSize = builder.convStepSize;
        this.convLearningRate = builder.convLearningRate;

        this.poolWindowSize = builder.poolWindowSize;
        this.poolStepSize = builder.poolStepSize;

        this.numClasses = builder.numClasses;
        this.fcLearningRate = builder.fcLearningRate;

        this.verbose = builder.verbose;
        this.interactive = builder.interactive;
    }

    public static AppConfig getDefault() {
        return new Builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder(this);
    }

    // Getters
    public String getTrainPath() { return trainPath; }
    public String getTestPath() { return testPath; }
    public int getTrainLimit() { return trainLimit; }
    public int getTestLimit() { return testLimit; }
    public int getEpochs() { return epochs; }
    public long getSeed() { return seed; }
    public double getScaleFactor() { return scaleFactor; }
    public String getSavePath() { return savePath; }
    public String getLoadPath() { return loadPath; }

    public int getNumFilters() { return numFilters; }
    public int getFilterSize() { return filterSize; }
    public int getConvStepSize() { return convStepSize; }
    public double getConvLearningRate() { return convLearningRate; }

    public int getPoolWindowSize() { return poolWindowSize; }
    public int getPoolStepSize() { return poolStepSize; }

    public int getNumClasses() { return numClasses; }
    public double getFcLearningRate() { return fcLearningRate; }

    public boolean isVerbose() { return verbose; }
    public boolean isInteractive() { return interactive; }

    // Derived CNN dimension calculations
    public int getConvOutputRows() {
        if (convStepSize <= 0) return 0;
        return (INPUT_ROWS - filterSize) / convStepSize + 1;
    }

    public int getConvOutputCols() {
        if (convStepSize <= 0) return 0;
        return (INPUT_COLS - filterSize) / convStepSize + 1;
    }

    public int getPoolOutputRows() {
        if (poolStepSize <= 0) return 0;
        return (getConvOutputRows() - poolWindowSize) / poolStepSize + 1;
    }

    public int getPoolOutputCols() {
        if (poolStepSize <= 0) return 0;
        return (getConvOutputCols() - poolWindowSize) / poolStepSize + 1;
    }

    public int getFcInputFeatures() {
        return numFilters * getPoolOutputRows() * getPoolOutputCols();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AppConfig appConfig = (AppConfig) o;
        return trainLimit == appConfig.trainLimit &&
                testLimit == appConfig.testLimit &&
                epochs == appConfig.epochs &&
                seed == appConfig.seed &&
                Double.compare(appConfig.scaleFactor, scaleFactor) == 0 &&
                numFilters == appConfig.numFilters &&
                filterSize == appConfig.filterSize &&
                convStepSize == appConfig.convStepSize &&
                Double.compare(appConfig.convLearningRate, convLearningRate) == 0 &&
                poolWindowSize == appConfig.poolWindowSize &&
                poolStepSize == appConfig.poolStepSize &&
                numClasses == appConfig.numClasses &&
                Double.compare(appConfig.fcLearningRate, fcLearningRate) == 0 &&
                verbose == appConfig.verbose &&
                interactive == appConfig.interactive &&
                Objects.equals(trainPath, appConfig.trainPath) &&
                Objects.equals(testPath, appConfig.testPath) &&
                Objects.equals(savePath, appConfig.savePath) &&
                Objects.equals(loadPath, appConfig.loadPath);
    }

    @Override
    public int hashCode() {
        return Objects.hash(trainPath, testPath, trainLimit, testLimit, epochs, seed,
                scaleFactor, savePath, loadPath, numFilters, filterSize, convStepSize, convLearningRate,
                poolWindowSize, poolStepSize, numClasses, fcLearningRate, verbose, interactive);
    }

    @Override
    public String toString() {
        return "AppConfig{" +
                "trainPath='" + trainPath + '\'' +
                ", testPath='" + testPath + '\'' +
                ", trainLimit=" + trainLimit +
                ", testLimit=" + testLimit +
                ", epochs=" + epochs +
                ", seed=" + seed +
                ", scaleFactor=" + scaleFactor +
                ", savePath='" + savePath + '\'' +
                ", loadPath='" + loadPath + '\'' +
                ", numFilters=" + numFilters +
                ", filterSize=" + filterSize +
                ", convStepSize=" + convStepSize +
                ", convLearningRate=" + convLearningRate +
                ", poolWindowSize=" + poolWindowSize +
                ", poolStepSize=" + poolStepSize +
                ", numClasses=" + numClasses +
                ", fcLearningRate=" + fcLearningRate +
                ", verbose=" + verbose +
                ", interactive=" + interactive +
                '}';
    }

    /**
     * Fluent builder for {@link AppConfig}.
     */
    public static final class Builder {
        private String trainPath = DEFAULT_TRAIN_PATH;
        private String testPath = DEFAULT_TEST_PATH;
        private int trainLimit = DEFAULT_TRAIN_LIMIT;
        private int testLimit = DEFAULT_TEST_LIMIT;
        private int epochs = DEFAULT_EPOCHS;
        private long seed = DEFAULT_SEED;
        private double scaleFactor = DEFAULT_SCALE_FACTOR;
        private String savePath = DEFAULT_SAVE_PATH;
        private String loadPath = DEFAULT_LOAD_PATH;

        private int numFilters = DEFAULT_NUM_FILTERS;
        private int filterSize = DEFAULT_FILTER_SIZE;
        private int convStepSize = DEFAULT_CONV_STEP_SIZE;
        private double convLearningRate = DEFAULT_CONV_LEARNING_RATE;

        private int poolWindowSize = DEFAULT_POOL_WINDOW_SIZE;
        private int poolStepSize = DEFAULT_POOL_STEP_SIZE;

        private int numClasses = DEFAULT_NUM_CLASSES;
        private double fcLearningRate = DEFAULT_FC_LEARNING_RATE;

        private boolean verbose = DEFAULT_VERBOSE;
        private boolean interactive = DEFAULT_INTERACTIVE;

        public Builder() {}

        public Builder(AppConfig config) {
            this.trainPath = config.trainPath;
            this.testPath = config.testPath;
            this.trainLimit = config.trainLimit;
            this.testLimit = config.testLimit;
            this.epochs = config.epochs;
            this.seed = config.seed;
            this.scaleFactor = config.scaleFactor;
            this.savePath = config.savePath;
            this.loadPath = config.loadPath;
            this.numFilters = config.numFilters;
            this.filterSize = config.filterSize;
            this.convStepSize = config.convStepSize;
            this.convLearningRate = config.convLearningRate;
            this.poolWindowSize = config.poolWindowSize;
            this.poolStepSize = config.poolStepSize;
            this.numClasses = config.numClasses;
            this.fcLearningRate = config.fcLearningRate;
            this.verbose = config.verbose;
            this.interactive = config.interactive;
        }

        public Builder trainPath(String trainPath) {
            this.trainPath = trainPath;
            return this;
        }

        public Builder testPath(String testPath) {
            this.testPath = testPath;
            return this;
        }

        public Builder trainLimit(int trainLimit) {
            this.trainLimit = trainLimit;
            return this;
        }

        public Builder testLimit(int testLimit) {
            this.testLimit = testLimit;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder scaleFactor(double scaleFactor) {
            this.scaleFactor = scaleFactor;
            return this;
        }

        public Builder savePath(String savePath) {
            this.savePath = savePath;
            return this;
        }

        public Builder loadPath(String loadPath) {
            this.loadPath = loadPath;
            return this;
        }

        public Builder numFilters(int numFilters) {
            this.numFilters = numFilters;
            return this;
        }

        public Builder filterSize(int filterSize) {
            this.filterSize = filterSize;
            return this;
        }

        public Builder convStepSize(int convStepSize) {
            this.convStepSize = convStepSize;
            return this;
        }

        public Builder convLearningRate(double convLearningRate) {
            this.convLearningRate = convLearningRate;
            return this;
        }

        public Builder poolWindowSize(int poolWindowSize) {
            this.poolWindowSize = poolWindowSize;
            return this;
        }

        public Builder poolStepSize(int poolStepSize) {
            this.poolStepSize = poolStepSize;
            return this;
        }

        public Builder numClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        public Builder fcLearningRate(double fcLearningRate) {
            this.fcLearningRate = fcLearningRate;
            return this;
        }

        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        public Builder interactive(boolean interactive) {
            this.interactive = interactive;
            return this;
        }

        public AppConfig build() {
            return new AppConfig(this);
        }
    }
}
