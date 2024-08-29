using Neuroblox
using Documenter

DocMeta.setdocmeta!(Neuroblox, :DocTestSetup, :(using Neuroblox); recursive = true)

include("pages.jl")

makedocs(sitename = "Neuroblox",
    authors = "Neuroblox Inc.",
    modules = [Neuroblox],
    clean = true, doctest = false, linkcheck = false,
    warnonly = [:docs_block, :missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
    pages = pages)

repo =  "github.com/Neuroblox/NeurobloxDocsHost.git"

withenv("GITHUB_REPOSITORY" => repo) do
    deploydocs(; repo = repo, push_preview = true)
end
