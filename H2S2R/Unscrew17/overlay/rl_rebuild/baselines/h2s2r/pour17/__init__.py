"""Observation helpers reused by the Unscrew17 H2S2R adapter.

The Unscrew17 handoff contains only the shared observation contract.  Keep the
package initializer side-effect free so importing ``pour17.observation`` does
not require Pour17-only input and trajectory modules.
"""
