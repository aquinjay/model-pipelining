"""Independent two-coordinate Metropolis audits of the normal/inverse-gamma law."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import math
from statistics import fmean

from model_pipelining.inference import FitEvidence
from model_pipelining.sampling import _chain_seeds, diagnostic_summary


def audit_normal_mean_variance(model, request, posterior, metadata, checks):
    import emcee
    import numpy as np
    from scipy.special import digamma, ndtr, polygamma
    from scipy.stats import t

    settings = model.mcmc
    log_center = math.log(posterior.scale) - float(digamma(posterior.shape))
    log_sd = math.sqrt(float(polygamma(1, posterior.shape)))
    if not math.isfinite(log_center) or not math.isfinite(log_sd) or log_sd <= 0:
        raise ValueError("Unrepresentable log-variance transform")
    average = fmean(item.value for item in request.history)
    scatter = math.fsum((item.value - average)**2 for item in request.history)
    count = len(request.history)

    def decode(coordinates):
        z, w = map(float, coordinates)
        log_variance = log_center + log_sd * w
        variance = math.exp(log_variance)
        conditional_sd = math.sqrt(variance / posterior.mean_precision)
        mu = posterior.mean + conditional_sd * z
        jacobian = 1.5 * log_variance - 0.5 * math.log(posterior.mean_precision) + math.log(log_sd)
        if variance <= 0 or conditional_sd <= 0 or not all(math.isfinite(x) for x in (z, w, variance, mu, jacobian)):
            raise ValueError("Unrepresentable transformed sampler state")
        return mu, variance, log_variance, jacobian

    def log_probability(coordinates):
        try:
            mu, variance, lv, jacobian = decode(coordinates)
            # Raw prior * sufficient-statistic likelihood, with the full
            # nonconstant Jacobian for (z,w) -> (mu,v). Constants are omitted.
            prior = -(model.prior_shape + 1.5) * lv - model.prior_scale / variance
            prior -= 0.5 * model.prior_mean_precision * ((mu - model.prior_mean) / math.sqrt(variance))**2
            likelihood = -0.5 * count * lv - 0.5 * scatter / variance
            likelihood -= 0.5 * count * ((mu - average) / math.sqrt(variance))**2
            value = prior + likelihood + jacobian
            return value if math.isfinite(value) else -np.inf
        except (ValueError, OverflowError, ZeroDivisionError):
            return -np.inf

    rows, chains, chain_metadata, failures = [], [], [], []
    for chain, seed in enumerate(_chain_seeds(request, settings)):
        predictive_seed = int.from_bytes(hashlib.sha256(f"predictive:{seed}".encode()).digest()[:8], "big")
        generator = np.random.default_rng(predictive_seed)
        position = (2 * chain / (settings.chains - 1) - 1) * settings.initial_spread
        # Disperse starts across both transformed marginal distributions.
        start = [position, -position]
        kept, warmup_count, warmup_accepts, retained_accepts, previous_accepted = [], 0, 0, 0, 0
        initial = None
        try:
            initial = decode(start)
            sampler = emcee.EnsembleSampler(1, 2, log_probability,
                                           moves=emcee.moves.GaussianMove(settings.proposal_scale**2))
            state = emcee.State([start], random_state=np.random.RandomState(seed).get_state())
            for index, state in enumerate(sampler.sample(state, iterations=settings.warmup + settings.draws,
                                                        skip_initial_state_check=True, progress=False)):
                z, w = map(float, state.coords[0])
                mu, variance, lv, jacobian = decode((z, w))
                logp = float(state.log_prob[0])
                future = float(generator.normal(mu, math.sqrt(variance)))
                if not all(math.isfinite(x) for x in (logp, future)):
                    raise ValueError("Sampler produced nonfinite state or predictive draw")
                warmup = index < settings.warmup
                accepted_total = int(sampler.backend.accepted[0])
                accepted = accepted_total > previous_accepted
                previous_accepted = accepted_total
                warmup_count += int(warmup)
                warmup_accepts += int(warmup and accepted)
                retained_accepts += int(not warmup and accepted)
                row = {"chain": chain, "draw": index if warmup else index - settings.warmup,
                       "warmup": warmup, "mu": mu, "observation_variance": variance,
                       "log_variance": lv, "standardized_mu": z, "standardized_log_variance": w,
                       "future_value": future, "log_probability": logp, "log_jacobian": jacobian,
                       "accepted": accepted}
                rows.append(row)
                if not warmup:
                    kept.append(row)
            chains.append(kept)
        except Exception as error:
            failures.append({"chain": chain, "error": f"{type(error).__name__}: {error}"})
        chain_metadata.append({"chain": chain, "seed": seed, "predictive_seed": predictive_seed,
                               "initial_coordinates": start,
                               "initial_mu": initial[0] if initial is not None else None,
                               "initial_observation_variance": initial[1] if initial is not None else None,
                               "warmup_draws": warmup_count, "retained_draws": len(kept),
                               "warmup_acceptance_fraction": warmup_accepts / warmup_count if warmup_count else None,
                               "posterior_acceptance_fraction": retained_accepts / len(kept) if kept else None})

    diagnostics, omitted = [], []
    if len(chains) == settings.chains:
        def values(column):
            return np.asarray([[row[column] for row in chain] for chain in chains])

        def diagnose(array, name, reference, sd):
            diagnostics.append(diagnostic_summary(array, name=name, reference=reference,
                                                  reference_sd=sd, settings=settings))

        mu, variance = values("mu"), values("observation_variance")
        z, future = values("standardized_mu"), values("future_value")
        diagnose(1 / variance, "precision", posterior.shape / posterior.scale,
                 math.sqrt(posterior.shape) / posterior.scale)
        diagnose(values("log_variance"), "log_variance", log_center, log_sd)
        diagnose(z, "conditional_standardized_mu", 0, 1)
        diagnose(z**2, "conditional_standardized_mu_second_moment", 1, math.sqrt(2))
        if posterior.shape > 1:
            diagnose(mu, "mu_and_predictive_mean", posterior.mean, math.sqrt(posterior.mean_variance))
            diagnose(future, "predictive_draw_mean", posterior.mean, math.sqrt(posterior.predictive_variance))
        else:
            omitted.append({"quantities": ["mu_and_predictive_mean", "predictive_draw_mean"],
                            "reason": "shape <= 1: mean exists but variance needed for mean MCSE is undefined"})
        if posterior.shape > 2:
            diagnose(variance, "observation_variance", posterior.observation_variance_mean,
                     posterior.observation_variance_mean / math.sqrt(posterior.shape - 2))
            for array, name, reference in (
                (mu, "centered_parameter_second_moment", posterior.mean_variance),
                (future, "centered_predictive_second_moment", posterior.predictive_variance),
            ):
                sd = reference * math.sqrt((2 * posterior.degrees_of_freedom - 2) / (posterior.degrees_of_freedom - 4))
                diagnose((array - posterior.mean)**2, name, reference, sd)
        else:
            omitted.append({"quantities": ["observation_variance", "centered_parameter_second_moment",
                                            "centered_predictive_second_moment"],
                            "reason": "shape <= 2: the variances required for these MCSE audits are undefined"})
        predictive = t(df=posterior.degrees_of_freedom, loc=posterior.mean,
                       scale=math.sqrt(posterior.predictive_scale_squared))
        diagnose(predictive.cdf(future), "predictive_draw_cdf", 0.5, math.sqrt(1 / 12))
        for probability in request.quantile_levels:
            integrand = ndtr((predictive.ppf(probability) - mu) / np.sqrt(variance))
            diagnose(integrand, f"predictive_cdf_at_{probability:g}", probability, float(integrand.std(ddof=1)))
        from model_pipelining.uncertainty import interval_probabilities
        for level in request.highest_density_levels:
            lower, upper = interval_probabilities(level)
            integrand = (ndtr((predictive.ppf(upper) - mu) / np.sqrt(variance))
                         - ndtr((predictive.ppf(lower) - mu) / np.sqrt(variance)))
            diagnose(integrand, f"predictive_hpd_mass_{level:g}", level, float(integrand.std(ddof=1)))

    checks += ({"name": "all_chains_completed", "required": True,
                "status": "failed" if failures else "passed"},
               {"name": "mcmc_diagnostics_and_exact_reference", "required": True,
                "status": "passed" if diagnostics and all(x["status"] == "passed" for x in diagnostics) else "failed"},
               {"name": "hmc_divergences_energy_tree_depth", "required": False, "status": "not_applicable"})
    metadata = {**metadata, "sampler": {
        "backend": "emcee", "algorithm": "GaussianMove random-walk Metropolis", "independent_chains": True,
        "walkers_per_chain": 1, "dimensions": 2, "settings": asdict(settings), "chains": chain_metadata,
        "failures": failures, "omitted_moment_diagnostics": omitted,
        "transform": "v = exp(log_center + log_sd*w); mu = exact_mean + sqrt(v/exact_mean_precision)*z",
        "log_variance_center": log_center, "log_variance_sd": log_sd,
        "log_jacobian": "1.5*log(v) - 0.5*log(exact_mean_precision) + log(log_sd); included in target",
        "initialization": "Evenly spaced z in +/- initial_spread, w = -z",
        "adaptation": "none; warmup saved and excluded from diagnostics",
        "log_probability": "normal/inverse-gamma prior plus sufficient-statistic likelihood plus transform Jacobian; constants omitted",
        "seed_derivation": "SHA-256 of base seed, eligible UTC timestamp/value history, issuance, target, and chain index; first 32 bits",
        "versions": {name: version(name) for name in ("emcee", "arviz", "numpy", "scipy")}},
        "diagnostic_scope": "precision, log variance, conditional standardized mean, available finite moments, paired predictive CDF, requested predictive CDF/HPD probabilities; warmup excluded"}
    return FitEvidence(metadata, checks, tuple(diagnostics), tuple(rows))
