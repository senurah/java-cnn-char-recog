package cli;

/**
 * Exception thrown when CLI argument parsing encounters unknown flags,
 * missing option values, or invalid syntax.
 */
public class CliParseException extends Exception {

    public CliParseException(String message) {
        super(message);
    }

    public CliParseException(String message, Throwable cause) {
        super(message, cause);
    }
}
