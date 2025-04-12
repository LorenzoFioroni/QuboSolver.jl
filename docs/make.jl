#! format: off
# turns off the julia formatting of this file

using QuboSolver
using Documenter
using DocumenterVitepress
using DocumenterCitations

DocMeta.setdocmeta!(QuboSolver, :DocTestSetup, :(using QuboSolver); recursive = true)

# some options for `makedocs`
const DOCTEST = true # set `false` to skip doc tests
const DRAFT = false  # set `true`  to disable cell evaluation

# generate bibliography
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "resources", "bibliography.bib"), 
    style=:numeric,
)

const PAGES = [
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Resources" => [
        "API" => "resources/api.md",
        "Bibliography" => "resources/bibliography.md",
        "Citing" => "resources/citing.md",
    ],
]

makedocs(;
    modules = Module[
        QuboSolver, 
    ],
    authors = "Lorenzo Fioroni and Vincenzo Savona",
    repo = Remotes.GitHub("LorenzoFioroni", "QuboSolver.jl"),
    sitename = "QuboSolver.jl",
    pages = PAGES,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/LorenzoFioroni/QuboSolver.jl",
    ),
    plugins = [bib],
    warnonly = :missing_docs,
    draft = DRAFT,
    doctest = DOCTEST,
    checkdocs = :public,
)

deploydocs(;
    repo = "github.com/LorenzoFioroni/QuboSolver.jl",
    target = "build", # this is where Vitepress stores its output
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)