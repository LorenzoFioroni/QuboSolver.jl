@testitem "code quality > aqua" tags = [:code_quality] begin
    using Aqua
    Aqua.test_all(QuboSolver; ambiguities = false, unbound_args = false)
end

@testitem "code quality > jet" tags = [:code_quality] begin
    using JET
    JET.test_package(
        QuboSolver;
        target_defined_modules = true,
        ignore_missing_comparison = true,
    )
end
