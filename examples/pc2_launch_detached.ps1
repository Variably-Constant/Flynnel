# Start a script and return at once, leaving it running without the
# connection that started it.
#
# A harness run as a child of an ssh command keeps that connection's
# sshd busy for its whole life. Measured on this box, sshd holds about
# 1.0 cores while a command is in flight, against a floor of roughly
# 0.83 from lql-server and System. So a run launched over the wire sits
# near 1.83 busy cores and can never see a quiet box by its own gate,
# and awaitquiet's -SelfRoot cannot subtract it: that walks DOWN from
# the caller and sshd is the caller's ANCESTOR.
#
# Start-Process detaches the work. The launching command exits, the
# connection closes, sshd goes idle, and the harness measures a box that
# no longer carries its own transport.
#
# Prints the detached pid so a caller can follow the run by its process
# and by the log the harness writes, since nothing is streamed back.

param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$ScriptArgs = ''
)

if (-not (Test-Path $Script)) {
    "NO_SUCH_SCRIPT $Script"
    exit 2
}

# Created through WMI, so the WMI service is the parent and the new
# process is outside the ssh session's job object. Verified on this box:
# a process spawned this way was still alive when read back over a
# separate connection, after the one that made it had closed.
#
# Start-Process does not work here. It reports a pid and the process
# dies when the launching command returns, which cost three attempts at
# one measurement before the absence of a log file gave it away.
#
# The Task Scheduler does not work either from an ssh session on this
# box: a task registered and started from here never ran, reporting
# SCHED_S_TASK_HAS_NOT_RUN, while tasks registered elsewhere with an
# identical principal run normally.
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Script`" $ScriptArgs"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmd }

if ($r.ReturnValue -ne 0) {
    "LAUNCH_FAILED Win32_Process.Create returned $($r.ReturnValue) for $Script"
    exit 5
}
"LAUNCHED_DETACHED pid=$($r.ProcessId) script=$Script args=$ScriptArgs"
exit 0
