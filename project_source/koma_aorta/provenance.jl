module AortaProvenance

using SHA

export file_sha256, require_matching_provenance

"""Return the lowercase SHA-256 digest of a file without loading it at once."""
function file_sha256(path::AbstractString)
    isfile(path) || throw(ArgumentError("Cannot hash missing file: $path"))
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"""Require every expected provenance field before an artifact may be reused."""
function require_matching_provenance(
    stored,
    expected::AbstractDict;
    artifact::AbstractString,
)
    stored isa AbstractDict || throw(ArgumentError(
        "Refusing reuse of $artifact: provenance is missing or malformed; " *
        "the existing artifact was left unchanged.",
    ))
    normalized = Dict(string(key) => value for (key, value) in pairs(stored))
    for key in sort!(collect(keys(expected)))
        if !haskey(normalized, key)
            throw(ArgumentError(
                "Refusing reuse of $artifact: provenance key '$key' is " *
                "missing; the existing artifact was left unchanged.",
            ))
        end
        if !isequal(normalized[key], expected[key])
            throw(ArgumentError(
                "Refusing reuse of $artifact: provenance key '$key' does " *
                "not match (stored=$(repr(normalized[key])), " *
                "expected=$(repr(expected[key]))); the existing artifact " *
                "was left unchanged.",
            ))
        end
    end
    return true
end

end
