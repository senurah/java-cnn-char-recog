package cli;

/**
 * Exception thrown when hyperparameter or runtime configurations violate
 * mathematical feasibility, CNN dimension rules, or allowable ranges.
 */
public class ConfigValidationException extends Exception {

    public ConfigValidationException(String message) {
        super(message);
    }

    public ConfigValidationException(String parameterName, String reason) {
        super(String.format("Invalid configuration for '%s': %s", parameterName, reason));
    }

    public ConfigValidationException(String message, Throwable cause) {
        super(message, cause);
    }
}
