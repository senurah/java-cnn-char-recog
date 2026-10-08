package cli;

import java.io.PrintStream;

/**
 * Renders an aesthetic ASCII summary table of the active {@link AppConfig}
 * hyperparameters, tensor dimensions, and runtime constraints.
 */
public final class ConfigRenderer {

    private ConfigRenderer() {}

    /**
     * Formats the configuration into a cleanly aligned ASCII summary card.
     *
     * @param config The application configuration to display
     * @return Formatted ASCII summary string
     */
    public static String render(AppConfig config) {
        StringBuilder sb = new StringBuilder();

        String trainLimitStr = config.getTrainLimit() == 0
                ? "All 60000"
                : String.valueOf(config.getTrainLimit());
        String testLimitStr = config.getTestLimit() == 0
                ? "All 10000"
                : String.valueOf(config.getTestLimit());

        int convOutRows = config.getConvOutputRows();
        int convOutCols = config.getConvOutputCols();
        int poolOutRows = config.getPoolOutputRows();
        int poolOutCols = config.getPoolOutputCols();
        int totalPoolElements = config.getFcInputFeatures();

        sb.append("========================================================================\n");
        sb.append("                  CNN CHARACTER RECOGNITION CONFIGURATION                \n");
        sb.append("========================================================================\n");

        sb.append(" [Dataset & Runtime]\n");
        sb.append(String.format("   Train Data Path  : %-20s (Limit: %s)\n", config.getTrainPath(), trainLimitStr));
        sb.append(String.format("   Test Data Path   : %-20s (Limit: %s)\n", config.getTestPath(), testLimitStr));
        sb.append(String.format("   Epochs           : %d\n", config.getEpochs()));
        sb.append(String.format("   Random Seed      : %d\n", config.getSeed()));
        sb.append(String.format("   Scale Factor     : %.1f (Input: %dx%d)\n\n",
                config.getScaleFactor(), AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS));

        sb.append(" [Convolution Layer 1]\n");
        sb.append(String.format("   Filters          : %d\n", config.getNumFilters()));
        sb.append(String.format("   Kernel Size      : %dx%d (Step: %d)\n",
                config.getFilterSize(), config.getFilterSize(), config.getConvStepSize()));
        sb.append(String.format("   Output Shape     : %d x %dx%d\n",
                config.getNumFilters(), convOutRows, convOutCols));
        sb.append(String.format("   Learning Rate    : %.4f\n\n", config.getConvLearningRate()));

        sb.append(" [Max Pooling Layer 1]\n");
        sb.append(String.format("   Window Size      : %dx%d (Step: %d)\n",
                config.getPoolWindowSize(), config.getPoolWindowSize(), config.getPoolStepSize()));
        sb.append(String.format("   Output Shape     : %d x %dx%d (Total Elements: %d)\n\n",
                config.getNumFilters(), poolOutRows, poolOutCols, totalPoolElements));

        sb.append(" [Fully Connected Layer]\n");
        sb.append(String.format("   Input Features   : %d\n", totalPoolElements));
        sb.append(String.format("   Classes (Output) : %d\n", config.getNumClasses()));
        sb.append(String.format("   Learning Rate    : %.4f\n", config.getFcLearningRate()));

        sb.append("========================================================================\n");

        return sb.toString();
    }

    /**
     * Prints the formatted ASCII summary to System.out.
     *
     * @param config The application configuration to display
     */
    public static void print(AppConfig config) {
        print(config, System.out);
    }

    /**
     * Prints the formatted ASCII summary to the specified PrintStream.
     *
     * @param config The application configuration to display
     * @param out The PrintStream destination
     */
    public static void print(AppConfig config, PrintStream out) {
        out.print(render(config));
    }
}
