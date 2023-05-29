using Documenter
using RKHSRegularization

makedocs(
    sitename = "RKHSRegularization",
    format = Documenter.HTML(),
    modules = [RKHSRegularization]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
