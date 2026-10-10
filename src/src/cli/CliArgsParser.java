package cli;

import java.io.PrintStream;

/**
 * Pure Core Java parser for POSIX/GNU-style command-line arguments.
 * Supports flags, options with arguments (--key val or --key=val),
 * case-insensitive matching, synonyms, and partial overrides.
 */
public final class CliArgsParser {

    public static final class ParseResult {
        private final AppConfig config;
        private final boolean helpRequested;
        private final boolean interactiveRequested;

        public ParseResult(AppConfig config, boolean helpRequested, boolean interactiveRequested) {
            this.config = config;
            this.helpRequested = helpRequested;
            this.interactiveRequested = interactiveRequested;
        }

        public AppConfig getConfig() {
            return config;
        }

        public boolean isHelpRequested() {
            return helpRequested;
        }

        public boolean isInteractiveRequested() {
            return interactiveRequested;
        }
    }

    private CliArgsParser() {}

    /**
     * Parses the command line arguments starting from the default configuration.
     *
     * @param args Command-line arguments array
     * @return ParseResult containing the built config and control flags
     * @throws CliParseException If syntax is invalid, unknown flags are passed, or values cannot be parsed
     */
    public static ParseResult parse(String[] args) throws CliParseException {
        return parse(args, AppConfig.getDefault());
    }

    /**
     * Parses the command line arguments starting from a provided base configuration.
     *
     * @param args Command-line arguments array
     * @param base Base configuration to apply overrides to
     * @return ParseResult containing the built config and control flags
     * @throws CliParseException If syntax is invalid, unknown flags are passed, or values cannot be parsed
     */
    public static ParseResult parse(String[] args, AppConfig base) throws CliParseException {
        if (args == null || args.length == 0) {
            return new ParseResult(base, false, false);
        }

        AppConfig.Builder builder = base.toBuilder();
        boolean helpRequested = false;
        boolean interactiveRequested = false;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i].trim();
            if (arg.isEmpty()) {
                continue;
            }

            String flag = arg;
            String inlineVal = null;

            int eqIdx = arg.indexOf('=');
            if (eqIdx != -1) {
                flag = arg.substring(0, eqIdx);
                inlineVal = arg.substring(eqIdx + 1);
            }

            String lowerFlag = flag.toLowerCase();

            // Boolean flags (no argument required)
            if (lowerFlag.equals("--help") || lowerFlag.equals("-h")) {
                helpRequested = true;
                continue;
            }
            if (lowerFlag.equals("--custom-configuration") || lowerFlag.equals("-c")) {
                interactiveRequested = true;
                builder.interactive(true);
                continue;
            }
            if (lowerFlag.equals("--verbose") || lowerFlag.equals("-v")) {
                builder.verbose(true);
                continue;
            }
            if (lowerFlag.equals("--quick") || lowerFlag.equals("-q")) {
                builder.trainLimit(200);
                builder.testLimit(100);
                continue;
            }

            // Options requiring an argument
            String val;
            if (inlineVal != null) {
                val = inlineVal;
            } else {
                if (i + 1 >= args.length || (args[i + 1].startsWith("-") && !isNegativeNumber(args[i + 1]))) {
                    throw new CliParseException(
                            String.format("Option '%s' requires an argument. See --help for usage details.", flag));
                }
                i++;
                val = args[i].trim();
            }

            try {
                switch (lowerFlag) {
                    case "--epochs":
                    case "-e":
                        builder.epochs(parseInt(val, flag));
                        break;
                    case "--filters":
                        builder.numFilters(parseInt(val, flag));
                        break;
                    case "--filter-size":
                        builder.filterSize(parseInt(val, flag));
                        break;
                    case "--conv-stride":
                        builder.convStepSize(parseInt(val, flag));
                        break;
                    case "--conv-lr":
                        builder.convLearningRate(parseDouble(val, flag));
                        break;
                    case "--pool-window":
                        builder.poolWindowSize(parseInt(val, flag));
                        break;
                    case "--pool-stride":
                        builder.poolStepSize(parseInt(val, flag));
                        break;
                    case "--classes":
                        builder.numClasses(parseInt(val, flag));
                        break;
                    case "--fc-lr":
                        builder.fcLearningRate(parseDouble(val, flag));
                        break;
                    case "--scale-factor":
                        builder.scaleFactor(parseDouble(val, flag));
                        break;
                    case "--seed":
                        builder.seed(parseLong(val, flag));
                        break;
                    case "--train-limit":
                        builder.trainLimit(parseInt(val, flag));
                        break;
                    case "--test-limit":
                        builder.testLimit(parseInt(val, flag));
                        break;
                    case "--train-path":
                        builder.trainPath(val);
                        break;
                    case "--test-path":
                        builder.testPath(val);
                        break;
                    case "--save":
                        builder.savePath(val);
                        break;
                    case "--load":
                        builder.loadPath(val);
                        break;
                    default:
                        throw new CliParseException(
                                String.format("Unknown option '%s'. Use --help for available options.", flag));
                }
            } catch (NumberFormatException e) {
                throw new CliParseException(
                        String.format("Invalid numeric format for option '%s': '%s'", flag, val), e);
            }
        }

        return new ParseResult(builder.build(), helpRequested, interactiveRequested);
    }

    private static int parseInt(String val, String flag) throws CliParseException {
        try {
            return Integer.parseInt(val);
        } catch (NumberFormatException e) {
            throw new CliParseException(String.format("Option '%s' requires an integer value, got '%s'", flag, val), e);
        }
    }

    private static long parseLong(String val, String flag) throws CliParseException {
        try {
            return Long.parseLong(val);
        } catch (NumberFormatException e) {
            throw new CliParseException(String.format("Option '%s' requires an integer/long value, got '%s'", flag, val), e);
        }
    }

    private static double parseDouble(String val, String flag) throws CliParseException {
        try {
            return Double.parseDouble(val);
        } catch (NumberFormatException e) {
            throw new CliParseException(String.format("Option '%s' requires a floating-point value, got '%s'", flag, val), e);
        }
    }

    private static boolean isNegativeNumber(String str) {
        if (str == null || str.length() < 2 || str.charAt(0) != '-') {
            return false;
        }
        char secondChar = str.charAt(1);
        return Character.isDigit(secondChar);
    }

    /**
     * Prints formatted help documentation to the specified PrintStream.
     */
    public static void printHelp(PrintStream out) {
        out.println(getHelpMessage());
    }

    /**
     * Generates human-readable CLI help manual.
     */
    public static String getHelpMessage() {
        return ""
                + "Usage: java Main [OPTIONS]\n"
                + "\n"
                + "A pure Core Java Convolutional Neural Network (CNN) for handwritten digit recognition.\n"
                + "\n"
                + "General Options:\n"
                + "  -h, --help                 Display this help reference and exit\n"
                + "  -c, --custom-configuration Launch interactive step-by-step configuration wizard\n"
                + "  -v, --verbose              Enable detailed educational parameter explanations\n"
                + "  -q, --quick                Quick-test mode (equivalent to --train-limit 200 --test-limit 100)\n"
                + "\n"
                + "Training & Dataset Options:\n"
                + "  -e, --epochs <N>           Number of training epochs (default: 3, range: 1..1000)\n"
                + "      --train-limit <N>      Maximum training samples to load (default: 0 = all 60,000)\n"
                + "      --test-limit <N>       Maximum testing samples to load (default: 0 = all 10,000)\n"
                + "      --train-path <PATH>    Path to training CSV file (default: data/mnist_train.csv)\n"
                + "      --test-path <PATH>     Path to testing CSV file (default: data/mnist_test.csv)\n"
                + "      --seed <N>             Random seed for filter initialization (default: 123)\n"
                + "      --scale-factor <F>     Input pixel normalization divisor (default: 25600.0)\n"
                + "\n"
                + "Convolution Layer Hyperparameters:\n"
                + "      --filters <N>          Number of convolution filters (default: 8, range: 1..64)\n"
                + "      --filter-size <N>      Square kernel dimension NxN (default: 5, range: 1..28)\n"
                + "      --conv-stride <N>      Stride step size for convolution (default: 1)\n"
                + "      --conv-lr <F>          Learning rate for convolution weights (default: 0.1)\n"
                + "\n"
                + "Max Pooling Layer Hyperparameters:\n"
                + "      --pool-window <N>      Pooling window dimension NxN (default: 3)\n"
                + "      --pool-stride <N>      Stride step size for pooling (default: 2)\n"
                + "\n"
                + "Fully Connected Layer Hyperparameters:\n"
                + "      --classes <N>          Number of target output classes (default: 10)\n"
                + "      --fc-lr <F>            Learning rate for dense layer weights (default: 0.1)\n"
                + "\n"
                + "Model Persistence Options:\n"
                + "      --save <PATH>          Export trained model weights to the specified file\n"
                + "      --load <PATH>          Load pre-trained model weights before evaluation/training\n"
                + "\n"
                + "Examples:\n"
                + "  java Main                                # Run with default hyperparameters\n"
                + "  java Main --quick                        # Fast sanity test on sample slice\n"
                + "  java Main --epochs 5 --filters 16        # Custom epochs and filter count\n"
                + "  java Main --quick --save weights.bin     # Train and save weights to file\n"
                + "  java Main --load weights.bin --epochs 0  # Load weights and evaluate without retraining\n"
                + "  java Main --custom-configuration -v      # Interactive wizard with parameter explanations\n";
    }
}
