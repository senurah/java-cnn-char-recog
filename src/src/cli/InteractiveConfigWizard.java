package cli;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;

/**
 * Guided step-by-step interactive CLI wizard for custom CNN configuration.
 * Validates each parameter interactively on input and provides educational
 * explanations when verbose mode is active.
 */
public class InteractiveConfigWizard {

    private final BufferedReader reader;
    private final PrintStream out;
    private final boolean verbose;

    public InteractiveConfigWizard() {
        this(new BufferedReader(new InputStreamReader(System.in)), System.out, false);
    }

    public InteractiveConfigWizard(InputStream in, PrintStream out, boolean verbose) {
        this(new BufferedReader(new InputStreamReader(in)), out, verbose);
    }

    public InteractiveConfigWizard(BufferedReader reader, PrintStream out, boolean verbose) {
        this.reader = reader;
        this.out = out;
        this.verbose = verbose;
    }

    /**
     * Executes the interactive configuration wizard starting from default values.
     *
     * @return Validated {@link AppConfig} instance reflecting user selections
     * @throws IOException If terminal I/O fails
     */
    public AppConfig runWizard() throws IOException {
        return runWizard(AppConfig.getDefault());
    }

    /**
     * Executes the interactive configuration wizard starting from an initial configuration.
     *
     * @param base Starting configuration providing defaults
     * @return Validated {@link AppConfig} instance reflecting user selections
     * @throws IOException If terminal I/O fails
     */
    public AppConfig runWizard(AppConfig base) throws IOException {
        out.println("========================================================================");
        out.println("                  INTERACTIVE CNN CONFIGURATION WIZARD                  ");
        out.println("========================================================================");
        out.println("Configure CNN hyperparameters step-by-step.");
        out.println("Press [Enter] to keep the default value displayed in brackets.");
        out.println("========================================================================\n");

        AppConfig.Builder builder = base.toBuilder();
        builder.interactive(true);
        if (verbose) {
            builder.verbose(true);
        }

        // 1. Epochs
        int epochs = promptInt(
                ParameterExplainer.EPOCHS,
                "Training Epochs",
                base.getEpochs(),
                ConfigValidator::validateEpochs
        );
        builder.epochs(epochs);

        // 2. Convolution Filters
        int numFilters = promptInt(
                ParameterExplainer.NUM_FILTERS,
                "Convolution Filters",
                base.getNumFilters(),
                ConfigValidator::validateNumFilters
        );
        builder.numFilters(numFilters);

        // 3. Kernel / Filter Size
        int filterSize = promptInt(
                ParameterExplainer.FILTER_SIZE,
                "Kernel / Filter Size (NxN)",
                base.getFilterSize(),
                ConfigValidator::validateFilterSize
        );
        builder.filterSize(filterSize);

        // 4. Convolution Stride
        int convStride = promptInt(
                ParameterExplainer.CONV_STRIDE,
                "Convolution Stride",
                base.getConvStepSize(),
                val -> ConfigValidator.validateConvStride(filterSize, val)
        );
        builder.convStepSize(convStride);

        int convOutRows = (AppConfig.INPUT_ROWS - filterSize) / convStride + 1;

        // 5. Convolution Learning Rate
        double convLr = promptDouble(
                ParameterExplainer.CONV_LR,
                "Convolution Learning Rate",
                base.getConvLearningRate(),
                ConfigValidator::validateConvLearningRate
        );
        builder.convLearningRate(convLr);

        // 6. Max Pooling Window Size
        int poolWindow = promptInt(
                ParameterExplainer.POOL_WINDOW,
                "Max Pooling Window Size (NxN)",
                base.getPoolWindowSize(),
                val -> ConfigValidator.validatePoolWindow(convOutRows, val)
        );
        builder.poolWindowSize(poolWindow);

        // 7. Max Pooling Stride
        int poolStride = promptInt(
                ParameterExplainer.POOL_STRIDE,
                "Max Pooling Stride",
                base.getPoolStepSize(),
                val -> ConfigValidator.validatePoolStride(convOutRows, poolWindow, val)
        );
        builder.poolStepSize(poolStride);

        // 8. Output Classes
        int numClasses = promptInt(
                ParameterExplainer.NUM_CLASSES,
                "Target Classes",
                base.getNumClasses(),
                ConfigValidator::validateNumClasses
        );
        builder.numClasses(numClasses);

        // 9. Fully Connected Learning Rate
        double fcLr = promptDouble(
                ParameterExplainer.FC_LR,
                "Fully Connected Learning Rate",
                base.getFcLearningRate(),
                ConfigValidator::validateFcLearningRate
        );
        builder.fcLearningRate(fcLr);

        // 10. Scale Factor
        double scaleFactor = promptDouble(
                ParameterExplainer.SCALE_FACTOR,
                "Pixel Normalization Scale Factor",
                base.getScaleFactor(),
                ConfigValidator::validateScaleFactor
        );
        builder.scaleFactor(scaleFactor);

        // 11. Random Seed
        long seed = promptLong(
                ParameterExplainer.SEED,
                "Random Seed",
                base.getSeed(),
                null
        );
        builder.seed(seed);

        // 12. Training Sample Limit
        int trainLimit = promptInt(
                ParameterExplainer.TRAIN_LIMIT,
                "Training Sample Limit (0 for all 60,000)",
                base.getTrainLimit(),
                ConfigValidator::validateTrainLimit
        );
        builder.trainLimit(trainLimit);

        // 13. Testing Sample Limit
        int testLimit = promptInt(
                ParameterExplainer.TEST_LIMIT,
                "Testing Sample Limit (0 for all 10,000)",
                base.getTestLimit(),
                ConfigValidator::validateTestLimit
        );
        builder.testLimit(testLimit);

        AppConfig finalConfig = builder.build();

        try {
            ConfigValidator.validate(finalConfig);
        } catch (ConfigValidationException e) {
            out.println("\n[Critical Validation Error] " + e.getMessage());
            throw new IllegalStateException("Interactive configuration produced invalid state: " + e.getMessage(), e);
        }

        out.println("\n========================================================================");
        out.println("         Configuration successfully completed and validated!            ");
        out.println("========================================================================\n");

        return finalConfig;
    }

    @FunctionalInterface
    private interface IntValidator {
        void validate(int value) throws ConfigValidationException;
    }

    @FunctionalInterface
    private interface DoubleValidator {
        void validate(double value) throws ConfigValidationException;
    }

    @FunctionalInterface
    private interface LongValidator {
        void validate(long value) throws ConfigValidationException;
    }

    private int promptInt(ParameterExplainer.ParameterDoc doc, String label, int defaultVal, IntValidator validator)
            throws IOException {
        if (verbose && doc != null) {
            ParameterExplainer.printDoc(out, doc);
        }

        while (true) {
            out.printf("%s [default: %d]: ", label, defaultVal);
            out.flush();
            String line = reader.readLine();
            if (line == null || line.trim().isEmpty()) {
                return defaultVal;
            }

            try {
                int val = Integer.parseInt(line.trim());
                if (validator != null) {
                    validator.validate(val);
                }
                return val;
            } catch (NumberFormatException e) {
                out.println("  [Error] Invalid input: please enter a valid integer.");
            } catch (ConfigValidationException e) {
                out.println("  [Error] " + e.getMessage());
            }
        }
    }

    private long promptLong(ParameterExplainer.ParameterDoc doc, String label, long defaultVal, LongValidator validator)
            throws IOException {
        if (verbose && doc != null) {
            ParameterExplainer.printDoc(out, doc);
        }

        while (true) {
            out.printf("%s [default: %d]: ", label, defaultVal);
            out.flush();
            String line = reader.readLine();
            if (line == null || line.trim().isEmpty()) {
                return defaultVal;
            }

            try {
                long val = Long.parseLong(line.trim());
                if (validator != null) {
                    validator.validate(val);
                }
                return val;
            } catch (NumberFormatException e) {
                out.println("  [Error] Invalid input: please enter a valid integer/long value.");
            } catch (ConfigValidationException e) {
                out.println("  [Error] " + e.getMessage());
            }
        }
    }

    private double promptDouble(ParameterExplainer.ParameterDoc doc, String label, double defaultVal, DoubleValidator validator)
            throws IOException {
        if (verbose && doc != null) {
            ParameterExplainer.printDoc(out, doc);
        }

        while (true) {
            out.printf("%s [default: %.4f]: ", label, defaultVal);
            out.flush();
            String line = reader.readLine();
            if (line == null || line.trim().isEmpty()) {
                return defaultVal;
            }

            try {
                double val = Double.parseDouble(line.trim());
                if (validator != null) {
                    validator.validate(val);
                }
                return val;
            } catch (NumberFormatException e) {
                out.println("  [Error] Invalid input: please enter a valid floating-point number.");
            } catch (ConfigValidationException e) {
                out.println("  [Error] " + e.getMessage());
            }
        }
    }
}
