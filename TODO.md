
# TODO

## Refactoring

1. ~~The PriorityReason from should have better details. Each strategy should have it's own. It should have some pretty short format implementation .e.g in case of the important_files strategy it should print a file name if one, if multiple it should print the number of important files~~ **DONE** - Implemented via StrategyResult classes with format_short() method.

2. Consider alternative approach for passing context to priority strategies.
   Currently StrategyContext is passed through the call chain, but this may
   need expansion as more strategies require different context. The current
   approach is clean but may need revisiting if context requirements grow
   significantly.

3. Verify the approach for empty merge_picks strategy list. Currently we
   log a warning and return no prioritized commits. Consider whether a
   default strategy (like conflict) should be used instead.