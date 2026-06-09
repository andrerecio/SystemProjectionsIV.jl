using Documenter
using SystemProjectionsIV

DocMeta.setdocmeta!(
    SystemProjectionsIV,
    :DocTestSetup,
    :(using SystemProjectionsIV);
    recursive = true
)

makedocs(;
    modules = [SystemProjectionsIV],
    authors = "Andrea Recine",
    sitename = "SystemProjectionsIV.jl",
    checkdocs = :exported,
    repo = Remotes.GitHub("andrerecio", "SystemProjectionsIV.jl"),
    format = Documenter.HTML(;
        canonical = "https://andrerecio.github.io/SystemProjectionsIV.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/andrerecio/SystemProjectionsIV.jl.git",
    devbranch = "main"
)
