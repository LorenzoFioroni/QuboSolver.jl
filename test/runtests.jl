using TestItemRunner

const GROUP = get(ENV, "GROUP", "all")
const OUTPUT = get(ENV, "OUTPUT", "true") == "true"

println("Running tests for group: $GROUP")
println("Output verbosity: $OUTPUT")

if GROUP == "core"
    filter = ti -> !(:code_quality in ti.tags)
elseif GROUP == "code quality"
    filter = ti -> :code_quality in ti.tags
elseif GROUP == "all"
    filter = ti -> true
else
    error("Invalid GROUP: $GROUP. Use 'core', 'code quality', or 'all'.")
end

@run_package_tests filter = filter verbose = OUTPUT
