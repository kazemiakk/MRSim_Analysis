# E-057 isolated sampling rule: refine free precession only, never the RF span.
# Uses KomaMRIBase's public SamplingRule extension point; installed upstream is unchanged.
struct PrecessionMaxStepRule <: KomaMRIBase.SamplingRule
    Δt_prec::Float64
    Δt_rf::Float64
end

# Same preserved samples as the stock rule E056 used through default_sim_params.
KomaMRIBase.preserved_samples(::PrecessionMaxStepRule) = (:gradients, :rf_center)

function KomaMRIBase.additional_sampling_times(rule::PrecessionMaxStepRule, event_times, context::KomaMRIBase.BlockSamplingContext)
    # Exactly the points the E056 primary rule MaxStepSizeRule(Inf, Δt_rf) adds.
    base = KomaMRIBase.additional_sampling_times(MaxStepSizeRule(Inf, rule.Δt_rf), event_times, context)
    isfinite(rule.Δt_prec) || return base
    gradient = KomaMRIBase.additional_sampling_times(MaxStepSizeRule(rule.Δt_prec, Inf), event_times, context)
    rf_t = context.waveforms.rf.t
    if !isempty(rf_t)
        lo, hi = first(rf_t), last(rf_t)
        gradient = filter(t -> !(lo < t < hi), gradient)
    end
    isempty(gradient) && return base
    return KomaMRIBase.merge_sampling_times(base, gradient)
end
