using KomaMRIBase
const E046_SORTED_LOOKUP_ENABLED=Ref(true)
"""Exact closed-interval lookup for Koma's sorted Float64 event vector.

Only the search is changed. Duplicates, endpoint inclusion, order and floating
subtractions are identical to the generic upstream comprehension. Installed
package files are never modified. The generic method remains callable by invoke.
"""
function KomaMRIBase.block_global_event_times(T0,block,events::Vector{Float64})
    E046_SORTED_LOOKUP_ENABLED[]||return invoke(KomaMRIBase.block_global_event_times,Tuple{Any,Any,Any},T0,block,events)
    first=searchsortedfirst(events,T0[block]);last=searchsortedlast(events,T0[block+1])
    return [events[i]-T0[block] for i in first:last]
end
