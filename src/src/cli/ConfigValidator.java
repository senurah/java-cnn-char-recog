package cli;

/**
 * Validates {@link AppConfig} hyperparameter and runtime configurations against
 * numerical constraints, parameter bounds, and CNN mathematical viability.
 */
public final class ConfigValidator {

    private ConfigValidator() {}

    /**
     * Validates an entire {@link AppConfig} object.
     *
     * @param config The application configuration to validate
     * @throws ConfigValidationException If any constraint or mathematical viability check fails
     */
    public static void validate(AppConfig config) throws ConfigValidationException {
        if (config == null) {
            throw new ConfigValidationException("Configuration instance cannot be null");
        }

        // Dataset paths
        if (config.getTrainPath() == null || config.getTrainPath().trim().isEmpty()) {
            throw new ConfigValidationException("trainPath", "Training dataset path cannot be null or empty");
        }
        if (config.getTestPath() == null || config.getTestPath().trim().isEmpty()) {
            throw new ConfigValidationException("testPath", "Testing dataset path cannot be null or empty");
        }

        // Subsample limits
        validateTrainLimit(config.getTrainLimit());
        validateTestLimit(config.getTestLimit());

        // Epochs
        validateEpochs(config.getEpochs());

        // Scale factor
        validateScaleFactor(config.getScaleFactor());

        // Convolution Layer constraints
        validateNumFilters(config.getNumFilters());
        validateFilterSize(config.getFilterSize());
        validateConvStride(config.getFilterSize(), config.getConvStepSize());
        validateConvLearningRate(config.getConvLearningRate());

        int convOutRows = config.getConvOutputRows();
        int convOutCols = config.getConvOutputCols();
        if (convOutRows <= 0 || convOutCols <= 0) {
            throw new ConfigValidationException("filterSize/convStepSize",
                    String.format("Convolution output dimensions must be positive, but calculated (%dx%d) for input (%dx%d)",
                            convOutRows, convOutCols, AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS));
        }

        // Max Pooling Layer constraints
        validatePoolWindow(convOutRows, config.getPoolWindowSize());
        validatePoolStride(convOutRows, config.getPoolWindowSize(), config.getPoolStepSize());

        int poolOutRows = config.getPoolOutputRows();
        int poolOutCols = config.getPoolOutputCols();
        if (poolOutRows <= 0 || poolOutCols <= 0) {
            throw new ConfigValidationException("poolWindow/poolStepSize",
                    String.format("Pooling output dimensions must be positive, but calculated (%dx%d) from conv map (%dx%d)",
                            poolOutRows, poolOutCols, convOutRows, convOutCols));
        }

        // Fully Connected Layer constraints
        validateNumClasses(config.getNumClasses());
        validateFcLearningRate(config.getFcLearningRate());

        // Overall FC feature check
        if (config.getFcInputFeatures() <= 0) {
            throw new ConfigValidationException("networkArchitecture",
                    "Calculated fully connected input features must be positive, got: " + config.getFcInputFeatures());
        }
    }

    public static void validateEpochs(int epochs) throws ConfigValidationException {
        if (epochs < 1 || epochs > 1000) {
            throw new ConfigValidationException("epochs",
                    "Epochs must be between 1 and 1000, got: " + epochs);
        }
    }

    public static void validateScaleFactor(double scaleFactor) throws ConfigValidationException {
        if (scaleFactor <= 0.0 || Double.isNaN(scaleFactor) || Double.isInfinite(scaleFactor)) {
            throw new ConfigValidationException("scaleFactor",
                    "Scale factor must be a positive finite number, got: " + scaleFactor);
        }
    }

    public static void validateTrainLimit(int trainLimit) throws ConfigValidationException {
        if (trainLimit < 0) {
            throw new ConfigValidationException("trainLimit",
                    "Train limit must be >= 0 (0 indicates no limit), got: " + trainLimit);
        }
    }

    public static void validateTestLimit(int testLimit) throws ConfigValidationException {
        if (testLimit < 0) {
            throw new ConfigValidationException("testLimit",
                    "Test limit must be >= 0 (0 indicates no limit), got: " + testLimit);
        }
    }

    public static void validateNumFilters(int numFilters) throws ConfigValidationException {
        if (numFilters < 1) {
            throw new ConfigValidationException("numFilters",
                    "Number of convolution filters must be at least 1, got: " + numFilters);
        }
    }

    public static void validateFilterSize(int filterSize) throws ConfigValidationException {
        if (filterSize < 1) {
            throw new ConfigValidationException("filterSize",
                    "Filter size must be at least 1, got: " + filterSize);
        }
        if (filterSize > AppConfig.INPUT_ROWS) {
            throw new ConfigValidationException("filterSize",
                    String.format("Filter size (%d) cannot exceed input image dimension (%dx%d)",
                            filterSize, AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS));
        }
    }

    public static void validateConvStride(int filterSize, int convStride) throws ConfigValidationException {
        if (convStride < 1) {
            throw new ConfigValidationException("convStride",
                    "Convolution stride must be at least 1, got: " + convStride);
        }
        int outRows = (AppConfig.INPUT_ROWS - filterSize) / convStride + 1;
        if (outRows <= 0) {
            throw new ConfigValidationException("convStride",
                    String.format("Filter size %d with stride %d produces non-positive output dimension: %d",
                            filterSize, convStride, outRows));
        }
    }

    public static void validateConvLearningRate(double convLearningRate) throws ConfigValidationException {
        if (convLearningRate <= 0.0 || convLearningRate > 1.0 || Double.isNaN(convLearningRate)) {
            throw new ConfigValidationException("convLearningRate",
                    "Convolution learning rate must be in range (0.0, 1.0], got: " + convLearningRate);
        }
    }

    public static void validatePoolWindow(int convOutRows, int poolWindow) throws ConfigValidationException {
        if (poolWindow < 1) {
            throw new ConfigValidationException("poolWindow",
                    "Pooling window size must be at least 1, got: " + poolWindow);
        }
        if (poolWindow > convOutRows) {
            throw new ConfigValidationException("poolWindow",
                    String.format("Pooling window size (%d) cannot exceed previous convolution dimension (%dx%d)",
                            poolWindow, convOutRows, convOutRows));
        }
    }

    public static void validatePoolStride(int convOutRows, int poolWindow, int poolStride) throws ConfigValidationException {
        if (poolStride < 1) {
            throw new ConfigValidationException("poolStride",
                    "Pooling stride must be at least 1, got: " + poolStride);
        }
        int outRows = (convOutRows - poolWindow) / poolStride + 1;
        if (outRows <= 0) {
            throw new ConfigValidationException("poolStride",
                    String.format("Conv dim %d, pool window %d, and stride %d produce non-positive pooling dimension: %d",
                            convOutRows, poolWindow, poolStride, outRows));
        }
    }

    public static void validateNumClasses(int numClasses) throws ConfigValidationException {
        if (numClasses < 2) {
            throw new ConfigValidationException("numClasses",
                    "Number of output classes must be at least 2, got: " + numClasses);
        }
    }

    public static void validateFcLearningRate(double fcLearningRate) throws ConfigValidationException {
        if (fcLearningRate <= 0.0 || fcLearningRate > 1.0 || Double.isNaN(fcLearningRate)) {
            throw new ConfigValidationException("fcLearningRate",
                    "Fully connected learning rate must be in range (0.0, 1.0], got: " + fcLearningRate);
        }
    }
}
