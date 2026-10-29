module Dp150Provenance

using SHA

export file_sha256, provenance_signature, require_matching_provenance

"""Return the lowercase SHA-256 digest of a file without loading it at once."""
function file_sha256(path::AbstractString)
    isfile(path) || throw(ArgumentError("Cannot hash missing file: $path"))
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function canonical_value(value)
    if value isa AbstractDict
        keys_sorted = sort!(string.(collect(keys(value))))
        normalized = Dict(string(key) => item for (key, item) in pairs(value))
        return "{" * join(
            (repr(key) * ":" * canonical_value(normalized[key]) for key in keys_sorted),
            ",",
        ) * "}"
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join((canonical_value(item) for item in value), ",") * "]"
    elseif value isa AbstractFloat
        return string(typeof(value), ":", bitstring(value))
    elseif value isa Integer || value isa Bool || value === nothing
        return string(typeof(value), ":", value)
    end
    return string(typeof(value), ":", repr(value))
end

"""Hash a provenance dictionary using a deterministic, type-aware encoding."""
function provenance_signature(fields::AbstractDict)
    payload = canonical_value(fields)
    return bytes2hex(SHA.sha256(codeunits(payload)))
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
