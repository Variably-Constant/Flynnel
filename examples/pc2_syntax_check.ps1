# Parse a script without running it, and say which.
#
# A syntax error in a harness that only runs when a gate fires would
# otherwise surface hours later as a run that produced nothing, at the
# exact moment the box was finally quiet enough to measure on.

param([Parameter(Mandatory = $true)][string]$Path)

$errs = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $Path, [ref]$null, [ref]$errs)

if ($errs -and $errs.Count -gt 0) {
    foreach ($e in $errs) {
        "PARSE_ERROR line $($e.Extent.StartLineNumber): $($e.Message)"
    }
    exit 1
}
"parses clean: $Path"
exit 0
