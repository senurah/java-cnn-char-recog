package test;

import cli.AppConfig;
import cli.CliArgsParser;
import cli.CliParseException;
import cli.ConfigRenderer;
import cli.ConfigValidationException;
import cli.ConfigValidator;
import cli.InteractiveConfigWizard;
import cli.ParameterExplainer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

public class Phase1Test {

    private static int testsPassed = 0;
    private static int testsFailed = 0;

    public static void main(String[] args) {
        System.out.println("=== Running Phase 1 Test Suite ===");

        testAppConfigDefaults();
        testAppConfigBuilderOverrides();
        testAppConfigToBuilder();
        testConfigValidator();
        testCliArgsParser();
        testInteractiveConfigWizard();
        testParameterExplainer();
        testConfigRenderer();

        System.out.println("\nPhase 1 Test Results: " + testsPassed + " passed, " + testsFailed + " failed.");
        if (testsFailed > 0) {
            System.exit(1);
        }
    }

    private static void assertTrue(String message, boolean condition) {
        if (condition) {
            System.out.println("  [PASS] " + message);
            testsPassed++;
        } else {
            System.err.println("  [FAIL] " + message);
            testsFailed++;
        }
    }

    private static void testAppConfigDefaults() {
        System.out.println("\n-- Testing TASK-101: AppConfig Defaults --");
        AppConfig config = AppConfig.getDefault();

        assertTrue("Train path default", "data/mnist_train.csv".equals(config.getTrainPath()));
        assertTrue("Test path default", "data/mnist_test.csv".equals(config.getTestPath()));
        assertTrue("Train limit default is 0", config.getTrainLimit() == 0);
        assertTrue("Test limit default is 0", config.getTestLimit() == 0);
        assertTrue("Epochs default is 3", config.getEpochs() == 3);
        assertTrue("Seed default is 123", config.getSeed() == 123L);
        assertTrue("Scale factor default is 25600.0", Math.abs(config.getScaleFactor() - 25600.0) < 1e-6);
        assertTrue("Filters default is 8", config.getNumFilters() == 8);
        assertTrue("Filter size default is 5", config.getFilterSize() == 5);
        assertTrue("Conv stride default is 1", config.getConvStepSize() == 1);
        assertTrue("Conv LR default is 0.1", Math.abs(config.getConvLearningRate() - 0.1) < 1e-6);
        assertTrue("Pool window default is 3", config.getPoolWindowSize() == 3);
        assertTrue("Pool stride default is 2", config.getPoolStepSize() == 2);
        assertTrue("Num classes default is 10", config.getNumClasses() == 10);
        assertTrue("FC LR default is 0.1", Math.abs(config.getFcLearningRate() - 0.1) < 1e-6);
        assertTrue("Verbose default is false", !config.isVerbose());
        assertTrue("Interactive default is false", !config.isInteractive());

        // Derived dimensions
        assertTrue("Conv output rows is 24", config.getConvOutputRows() == 24);
        assertTrue("Conv output cols is 24", config.getConvOutputCols() == 24);
        assertTrue("Pool output rows is 11", config.getPoolOutputRows() == 11);
        assertTrue("Pool output cols is 11", config.getPoolOutputCols() == 11);
        assertTrue("FC input features is 968", config.getFcInputFeatures() == 968);
    }

    private static void testAppConfigBuilderOverrides() {
        System.out.println("\n-- Testing TASK-101: AppConfig Partial Overrides --");
        AppConfig custom = AppConfig.builder()
                .epochs(10)
                .numFilters(16)
                .trainLimit(500)
                .verbose(true)
                .build();

        assertTrue("Overridden epochs is 10", custom.getEpochs() == 10);
        assertTrue("Overridden filters is 16", custom.getNumFilters() == 16);
        assertTrue("Overridden train limit is 500", custom.getTrainLimit() == 500);
        assertTrue("Overridden verbose is true", custom.isVerbose());

        // Unmentioned fields retain defaults
        assertTrue("Retained default test path", "data/mnist_test.csv".equals(custom.getTestPath()));
        assertTrue("Retained default filter size 5", custom.getFilterSize() == 5);
        assertTrue("Retained default pool window 3", custom.getPoolWindowSize() == 3);
        assertTrue("Retained default seed 123", custom.getSeed() == 123L);
        assertTrue("FC input features scaled with 16 filters: 16 * 11 * 11 = 1936", custom.getFcInputFeatures() == 1936);
    }

    private static void testAppConfigToBuilder() {
        System.out.println("\n-- Testing TASK-101: AppConfig toBuilder --");
        AppConfig initial = AppConfig.builder().epochs(5).numFilters(12).build();
        AppConfig modified = initial.toBuilder().epochs(7).build();

        assertTrue("Initial epochs remains 5", initial.getEpochs() == 5);
        assertTrue("Modified epochs is 7", modified.getEpochs() == 7);
        assertTrue("Modified keeps filters 12", modified.getNumFilters() == 12);
    }

    private static void testConfigValidator() {
        System.out.println("\n-- Testing TASK-102: ConfigValidator --");

        // 1. Valid default passes
        try {
            ConfigValidator.validate(AppConfig.getDefault());
            assertTrue("Default configuration passes validation", true);
        } catch (ConfigValidationException e) {
            assertTrue("Default configuration passes validation: " + e.getMessage(), false);
        }

        // 2. Filter size > 28 fails
        try {
            AppConfig invalid = AppConfig.builder().filterSize(30).build();
            ConfigValidator.validate(invalid);
            assertTrue("Filter size > 28 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Filter size > 28 failed as expected: " + e.getMessage(), true);
        }

        // 3. Negative conv stride fails
        try {
            AppConfig invalid = AppConfig.builder().convStepSize(0).build();
            ConfigValidator.validate(invalid);
            assertTrue("Stride 0 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Conv stride 0 failed as expected: " + e.getMessage(), true);
        }

        // 4. Pool window larger than conv out fails
        try {
            // Conv out with filter 27, stride 1 is (28-27)/1 + 1 = 2
            // Pool window 3 > 2 => fails
            AppConfig invalid = AppConfig.builder().filterSize(27).poolWindowSize(3).build();
            ConfigValidator.validate(invalid);
            assertTrue("Pool window > conv out should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Pool window > conv out failed as expected: " + e.getMessage(), true);
        }

        // 5. Conv learning rate > 1.0 or <= 0 fails
        try {
            AppConfig invalid = AppConfig.builder().convLearningRate(1.5).build();
            ConfigValidator.validate(invalid);
            assertTrue("Conv LR > 1.0 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Conv LR > 1.0 failed as expected: " + e.getMessage(), true);
        }

        try {
            AppConfig invalid = AppConfig.builder().fcLearningRate(0.0).build();
            ConfigValidator.validate(invalid);
            assertTrue("FC LR == 0.0 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("FC LR == 0.0 failed as expected: " + e.getMessage(), true);
        }

        // 6. Epochs < 1 or > 1000 fails
        try {
            AppConfig invalid = AppConfig.builder().epochs(0).build();
            ConfigValidator.validate(invalid);
            assertTrue("Epochs 0 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Epochs 0 failed as expected: " + e.getMessage(), true);
        }

        try {
            AppConfig invalid = AppConfig.builder().epochs(1001).build();
            ConfigValidator.validate(invalid);
            assertTrue("Epochs 1001 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Epochs 1001 failed as expected: " + e.getMessage(), true);
        }

        // 7. Num filters < 1 fails
        try {
            AppConfig invalid = AppConfig.builder().numFilters(0).build();
            ConfigValidator.validate(invalid);
            assertTrue("Filters 0 should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Filters 0 failed as expected: " + e.getMessage(), true);
        }

        // 8. Scale factor <= 0 fails
        try {
            AppConfig invalid = AppConfig.builder().scaleFactor(-10.0).build();
            ConfigValidator.validate(invalid);
            assertTrue("Negative scale factor should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Negative scale factor failed as expected: " + e.getMessage(), true);
        }
    }

    private static void testCliArgsParser() {
        System.out.println("\n-- Testing TASK-103: CliArgsParser --");

        // 1. Empty args returns defaults
        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{});
            assertTrue("Empty args produces default config", AppConfig.getDefault().equals(res.getConfig()));
            assertTrue("Help not requested", !res.isHelpRequested());
            assertTrue("Interactive not requested", !res.isInteractiveRequested());
        } catch (CliParseException e) {
            assertTrue("Empty args parse failed: " + e.getMessage(), false);
        }

        // 2. Help flags
        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{"--help"});
            assertTrue("--help sets isHelpRequested", res.isHelpRequested());
            CliArgsParser.ParseResult resShort = CliArgsParser.parse(new String[]{"-h"});
            assertTrue("-h sets isHelpRequested", resShort.isHelpRequested());
        } catch (CliParseException e) {
            assertTrue("Help parse failed: " + e.getMessage(), false);
        }

        // 3. Interactive flags
        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{"--custom-configuration"});
            assertTrue("--custom-configuration sets interactiveRequested", res.isInteractiveRequested());
            assertTrue("Config has interactive=true", res.getConfig().isInteractive());
            CliArgsParser.ParseResult resShort = CliArgsParser.parse(new String[]{"-c"});
            assertTrue("-c sets interactiveRequested", resShort.isInteractiveRequested());
        } catch (CliParseException e) {
            assertTrue("Interactive parse failed: " + e.getMessage(), false);
        }

        // 4. Quick mode
        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{"--quick"});
            assertTrue("--quick sets trainLimit=200", res.getConfig().getTrainLimit() == 200);
            assertTrue("--quick sets testLimit=100", res.getConfig().getTestLimit() == 100);

            CliArgsParser.ParseResult resShort = CliArgsParser.parse(new String[]{"-q"});
            assertTrue("-q sets trainLimit=200", resShort.getConfig().getTrainLimit() == 200);
            assertTrue("-q sets testLimit=100", resShort.getConfig().getTestLimit() == 100);
        } catch (CliParseException e) {
            assertTrue("Quick mode parse failed: " + e.getMessage(), false);
        }

        // 5. Partial overrides with space and equals
        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{
                    "--epochs", "5",
                    "--filters", "16",
                    "--filter-size=3",
                    "--conv-lr=0.05",
                    "--verbose"
            });
            assertTrue("Custom epochs is 5", res.getConfig().getEpochs() == 5);
            assertTrue("Custom filters is 16", res.getConfig().getNumFilters() == 16);
            assertTrue("Custom filter size is 3", res.getConfig().getFilterSize() == 3);
            assertTrue("Custom conv lr is 0.05", Math.abs(res.getConfig().getConvLearningRate() - 0.05) < 1e-6);
            assertTrue("Verbose is true", res.getConfig().isVerbose());
            // Default unmentioned fields
            assertTrue("Pool window remains default 3", res.getConfig().getPoolWindowSize() == 3);
            assertTrue("Seed remains default 123", res.getConfig().getSeed() == 123L);
        } catch (CliParseException e) {
            assertTrue("Partial overrides parse failed: " + e.getMessage(), false);
        }

        // 6. Short aliases: -e
        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{"-e", "7"});
            assertTrue("-e 7 sets epochs to 7", res.getConfig().getEpochs() == 7);
        } catch (CliParseException e) {
            assertTrue("-e parse failed: " + e.getMessage(), false);
        }

        // 7. Missing argument error
        try {
            CliArgsParser.parse(new String[]{"--epochs"});
            assertTrue("Missing arg should throw CliParseException", false);
        } catch (CliParseException e) {
            assertTrue("Missing arg threw CliParseException: " + e.getMessage(), true);
        }

        // 8. Unknown option error
        try {
            CliArgsParser.parse(new String[]{"--unknown-param", "123"});
            assertTrue("Unknown option should throw CliParseException", false);
        } catch (CliParseException e) {
            assertTrue("Unknown option threw CliParseException: " + e.getMessage(), true);
        }

        // 9. Invalid numeric format
        try {
            CliArgsParser.parse(new String[]{"--epochs", "not_a_number"});
            assertTrue("Invalid number should throw CliParseException", false);
        } catch (CliParseException e) {
            assertTrue("Invalid number threw CliParseException: " + e.getMessage(), true);
        }

        // 10. Help text generation is non-empty
        String help = CliArgsParser.getHelpMessage();
        assertTrue("Help message contains usage", help.contains("Usage: java Main"));
        assertTrue("Help message contains --custom-configuration", help.contains("--custom-configuration"));
    }

    private static void testInteractiveConfigWizard() {
        System.out.println("\n-- Testing TASK-104: InteractiveConfigWizard --");

        // 1. All defaults (13 newlines)
        String emptyInputs = "\n\n\n\n\n\n\n\n\n\n\n\n\n";
        ByteArrayInputStream in = new ByteArrayInputStream(emptyInputs.getBytes());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        InteractiveConfigWizard wizard = new InteractiveConfigWizard(in, new PrintStream(out), false);

        try {
            AppConfig config = wizard.runWizard();
            assertTrue("Wizard accepts all defaults", config.getEpochs() == 3);
            assertTrue("Wizard default filters is 8", config.getNumFilters() == 8);
            assertTrue("Wizard sets interactive=true", config.isInteractive());
        } catch (Exception e) {
            assertTrue("Wizard with default inputs failed: " + e.getMessage(), false);
        }

        // 2. Custom values with error recovery on first invalid input
        // Parameter 1: Epochs: first enter "0" (invalid), then "invalid_str", then "5" (valid)
        // Parameter 2: Filters: "16"
        // Parameter 3: Kernel size: "3"
        // Parameter 4: Conv stride: "1"
        // Parameter 5: Conv LR: "0.05"
        // Parameter 6: Pool window: "2"
        // Parameter 7: Pool stride: "2"
        // Parameter 8: Classes: "10"
        // Parameter 9: FC LR: "0.05"
        // Parameter 10: Scale factor: "25600"
        // Parameter 11: Seed: "999"
        // Parameter 12: Train limit: "150"
        // Parameter 13: Test limit: "75"
        String customInputs = "0\ninvalid_str\n5\n16\n3\n1\n0.05\n2\n2\n10\n0.05\n25600\n999\n150\n75\n";
        in = new ByteArrayInputStream(customInputs.getBytes());
        out = new ByteArrayOutputStream();
        wizard = new InteractiveConfigWizard(in, new PrintStream(out), false);

        try {
            AppConfig config = wizard.runWizard();
            assertTrue("Wizard parsed epochs after recovery: 5", config.getEpochs() == 5);
            assertTrue("Wizard parsed custom filters: 16", config.getNumFilters() == 16);
            assertTrue("Wizard parsed custom kernel size: 3", config.getFilterSize() == 3);
            assertTrue("Wizard parsed custom conv LR: 0.05", Math.abs(config.getConvLearningRate() - 0.05) < 1e-6);
            assertTrue("Wizard parsed custom seed: 999", config.getSeed() == 999L);
            assertTrue("Wizard parsed custom train limit: 150", config.getTrainLimit() == 150);
            assertTrue("Wizard parsed custom test limit: 75", config.getTestLimit() == 75);

            String outputStr = out.toString();
            assertTrue("Output contained invalid input error message", outputStr.contains("[Error]"));
        } catch (Exception e) {
            assertTrue("Wizard with custom inputs and error recovery failed: " + e.getMessage(), false);
        }

        // 3. Verbose wizard prints educational descriptions
        in = new ByteArrayInputStream(emptyInputs.getBytes());
        out = new ByteArrayOutputStream();
        InteractiveConfigWizard verboseWizard = new InteractiveConfigWizard(in, new PrintStream(out), true);

        try {
            AppConfig config = verboseWizard.runWizard();
            assertTrue("Verbose wizard config has verbose=true", config.isVerbose());
            String outputStr = out.toString();
            assertTrue("Verbose output contains Parameter: Convolution Filters",
                    outputStr.contains("[Parameter: Convolution Filters]"));
            assertTrue("Verbose output contains Meaning", outputStr.contains("Meaning"));
            assertTrue("Verbose output contains Recommended Range", outputStr.contains("Recommended Range"));
        } catch (Exception e) {
            assertTrue("Verbose wizard failed: " + e.getMessage(), false);
        }
    }

    private static void testParameterExplainer() {
        System.out.println("\n-- Testing TASK-105: ParameterExplainer --");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ParameterExplainer.printAllDocs(new PrintStream(out));
        String docs = out.toString();

        assertTrue("Docs contain Training Epochs", docs.contains("Training Epochs"));
        assertTrue("Docs contain Convolution Filters", docs.contains("Convolution Filters"));
        assertTrue("Docs contain Kernel / Filter Size", docs.contains("Kernel / Filter Size"));
        assertTrue("Docs contain Convolution Stride", docs.contains("Convolution Stride"));
        assertTrue("Docs contain Convolution Learning Rate", docs.contains("Convolution Learning Rate"));
        assertTrue("Docs contain Max Pooling Window Size", docs.contains("Max Pooling Window Size"));
        assertTrue("Docs contain Max Pooling Stride", docs.contains("Max Pooling Stride"));
        assertTrue("Docs contain Output Classes", docs.contains("Output Classes"));
        assertTrue("Docs contain Fully Connected Learning Rate", docs.contains("Fully Connected Learning Rate"));
        assertTrue("Docs contain Input Normalization Scale Factor", docs.contains("Input Normalization Scale Factor"));
        assertTrue("Docs contain Random Seed", docs.contains("Random Seed"));
        assertTrue("Docs contain Training Sample Limit", docs.contains("Training Sample Limit"));
        assertTrue("Docs contain Testing Sample Limit", docs.contains("Testing Sample Limit"));
    }

    private static void testConfigRenderer() {
        System.out.println("\n-- Testing TASK-106: ConfigRenderer --");

        // 1. Default config rendering
        AppConfig defaultConfig = AppConfig.getDefault();
        String summary = ConfigRenderer.render(defaultConfig);

        assertTrue("Summary contains title", summary.contains("CNN CHARACTER RECOGNITION CONFIGURATION"));
        assertTrue("Summary contains train data path and all limit", summary.contains("Train Data Path  : data/mnist_train.csv (Limit: All 60000)"));
        assertTrue("Summary contains test data path and all limit", summary.contains("Test Data Path   : data/mnist_test.csv  (Limit: All 10000)"));
        assertTrue("Summary contains epochs 3", summary.contains("Epochs           : 3"));
        assertTrue("Summary contains seed 123", summary.contains("Random Seed      : 123"));
        assertTrue("Summary contains scale factor", summary.contains("Scale Factor     : 25600.0 (Input: 28x28)"));
        assertTrue("Summary contains conv filters", summary.contains("Filters          : 8"));
        assertTrue("Summary contains kernel size", summary.contains("Kernel Size      : 5x5 (Step: 1)"));
        assertTrue("Summary contains conv output shape", summary.contains("Output Shape     : 8 x 24x24"));
        assertTrue("Summary contains conv lr", summary.contains("Learning Rate    : 0.1000"));
        assertTrue("Summary contains pool window", summary.contains("Window Size      : 3x3 (Step: 2)"));
        assertTrue("Summary contains pool output shape and total", summary.contains("Output Shape     : 8 x 11x11 (Total Elements: 968)"));
        assertTrue("Summary contains fc input features", summary.contains("Input Features   : 968"));
        assertTrue("Summary contains fc classes", summary.contains("Classes (Output) : 10"));

        // 2. Custom config with limits
        AppConfig custom = AppConfig.builder()
                .trainLimit(200)
                .testLimit(100)
                .numFilters(16)
                .build();
        String customSummary = ConfigRenderer.render(custom);

        assertTrue("Custom summary has train limit 200", customSummary.contains("(Limit: 200)"));
        assertTrue("Custom summary has test limit 100", customSummary.contains("(Limit: 100)"));
        assertTrue("Custom summary has filters 16", customSummary.contains("Filters          : 16"));
        assertTrue("Custom summary has pool elements 1936", customSummary.contains("Total Elements: 1936"));

        // 3. Print to custom PrintStream
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ConfigRenderer.print(defaultConfig, new PrintStream(out));
        assertTrue("Print method wrote correctly", out.toString().equals(summary));
    }
}
