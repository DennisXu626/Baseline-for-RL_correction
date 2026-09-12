set pagination off
set height 0
set width 0
set confirm off
set verbose off
set print thread-events off
set disable-randomization off
python
import gdb, os
_log = os.environ["H2S2R_GDB_LOG"]
gdb.execute("set logging file " + _log)
gdb.execute("set logging overwrite on")
gdb.execute("set logging redirect on")
gdb.execute("set logging enabled on")
end
handle SIGSEGV stop print nopass
handle SIGABRT stop print nopass
echo ===== NATIVE DEBUG RUN START =====\n
show disable-randomization
show environment CUDA_VISIBLE_DEVICES
show environment RL_ISAAC_NO_GUARD
show environment H2S2R_DISABLE_PERIODIC_TRACEBACK
show environment H2S2R_NATIVE_DEBUG_GUARD_TARGET
show environment H2S2R_NATIVE_DEBUG_GUARD_LINE
show environment H2S2R_NATIVE_DEBUG_GUARD_MARKER
run
python
import gdb, os, signal, threading, time
def _section(name, command):
    gdb.write("\n===== " + name + " =====\n")
    try:
        gdb.execute(command)
    except BaseException as error:
        gdb.write("SECTION_ERROR " + name + ": " + repr(error) + "\n")
_section("CURRENT THREAD", "thread")
_section("FULL BACKTRACE", "bt full")
_section("REGISTERS", "info registers")
if os.environ.get("H2S2R_GDB_INJECT_COMMAND_FAILURE") == "1":
    _section("INJECTED SINGLE COMMAND FAILURE", "info symbol")
_section("CURRENT INSTRUCTIONS", "x/32i $pc-64")
_section("SHARED LIBRARIES", "info sharedlibrary")
_section("PROCESS MAPPINGS", "info proc mappings")
_section("ALL THREADS", "thread apply all bt full")
_core = os.environ.get("H2S2R_GDB_CORE")
if _core:
    def _core_watchdog():
        time.sleep(120)
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=_core_watchdog, daemon=True).start()
    _section("CORE ATTEMPT", "generate-core-file " + _core)
gdb.write("\n===== NATIVE DEBUG CAPTURE COMPLETE =====\n")
gdb.execute("set logging enabled off")
end
quit
