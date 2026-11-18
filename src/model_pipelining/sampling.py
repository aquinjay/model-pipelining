"""Independent emcee Gaussian-Metropolis chains and exact-reference audits.

This module is deliberately specific to NormalMean, not a general inference
engine. ArviZ receives independent chains, never coupled ensemble walkers.
"""

from dataclasses import asdict, dataclass
from datetime import UTC
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
from statistics import fmean, NormalDist
import warnings

from model_pipelining.inference import FitEvidence


@dataclass(frozen=True, kw_only=True)
class MCMCSettings:
    chains: int = 4
    warmup: int = 1000
    draws: int = 2000
    seed: int = 0
    proposal_scale: float = 2.4
    initial_spread: float = 4.0
    max_rhat: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 400.0
    max_mcse_sd_ratio: float = 0.1
    reference_mcse_multiplier: float = 6.0

    def __post_init__(self) -> None:
        for name, minimum in (("chains", 2), ("warmup", 0), ("draws", 8)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("proposal_scale", "initial_spread", "max_rhat", "min_bulk_ess",
                     "min_tail_ess", "max_mcse_sd_ratio", "reference_mcse_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.max_rhat <= 1:
            raise ValueError("max_rhat must exceed 1")
        if (self.proposal_scale > math.sqrt(float.fromhex('0x1.fffffffffffffp+1023'))
                or self.proposal_scale**2 == 0):
            raise ValueError("proposal_scale squared must be representable, positive, and finite")


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _chain_seeds(request, settings):
    payload = {"seed": settings.seed,
               "history": [(item.timestamp.astimezone(UTC).isoformat(), float(item.value)) for item in request.history],
               "issued_at": request.issued_at.astimezone(UTC).isoformat(),
               "target_time": request.target_time.astimezone(UTC).isoformat()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).digest()
    return [int.from_bytes(hashlib.sha256(digest + index.to_bytes(8, "big")).digest()[:4], "big")
            for index in range(settings.chains)]


def diagnostic_summary(values, *, name, reference, reference_sd, settings):
    """Audit one chain-by-draw quantity; undefined diagnostics explicitly fail."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    result = {"quantity": name, "reference": _finite_or_none(reference), "reference_sd": _finite_or_none(reference_sd),
              "estimate": None, "rhat": None, "ess_bulk": None, "ess_tail": None,
              "mcse_mean": None, "absolute_error": None, "reference_tolerance": None,
              "mcse_sd_ratio": None, "status": "failed", "error": None}
    try:
        import arviz as az
        if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 8
                or not np.isfinite(values).all() or not math.isfinite(reference)
                or not math.isfinite(reference_sd) or reference_sd <= 0
                or np.any(np.ptp(values, axis=1) == 0)):
            raise ValueError("Need finite, nonconstant independent chains and a positive reference SD")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            statistics = {"estimate": float(values.mean()),
                          "rhat": float(az.rhat(values, method="rank")),
                          "ess_bulk": float(az.ess(values, method="bulk")),
                          "ess_tail": float(az.ess(values, method="tail")),
                          "mcse_mean": float(az.mcse(values, method="mean"))}
        if not all(math.isfinite(x) for x in statistics.values()) or statistics["mcse_mean"] <= 0:
            raise ValueError("Required diagnostic missing, nonfinite, or nonpositive MCSE")
        result.update(statistics)
        error = abs(statistics["estimate"] - reference)
        # Floating-point allowance is separate from finite-MCMC uncertainty.
        tolerance = settings.reference_mcse_multiplier * statistics["mcse_mean"] + 64 * math.ulp(float(reference))
        ratio = statistics["mcse_mean"] / reference_sd
        if not all(math.isfinite(x) for x in (error, tolerance, ratio)):
            raise ValueError("Unrepresentable reference error, tolerance, or precision ratio")
        result.update(absolute_error=error, reference_tolerance=tolerance, mcse_sd_ratio=ratio)
        passed = (statistics["rhat"] <= settings.max_rhat
                  and statistics["ess_bulk"] >= settings.min_bulk_ess
                  and statistics["ess_tail"] >= settings.min_tail_ess
                  and ratio <= settings.max_mcse_sd_ratio and error <= tolerance)
        result["status"] = "passed" if passed else "failed"
        if not passed:
            result["error"] = "One or more saved computation/reference criteria failed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def audit_normal_mean(model, request, posterior, metadata, checks):
    import emcee
    import numpy as np
    settings = model.mcmc
    sd = math.sqrt(posterior.variance)
    sample_mean = fmean(item.value for item in request.history)
    noise_sd = math.sqrt(model.observation_variance)
    prior_sd = math.sqrt(model.prior_variance)
    count = len(request.history)
    seeds = _chain_seeds(request, settings)
    # Sufficient-statistic likelihood, with constants independent of mu omitted.
    # Centering/scaling use the exact solution for this calibration exercise.
    def log_probability(z):
        mu = posterior.mean + sd * float(z[0])
        with np.errstate(over="ignore", invalid="ignore"):
            value = -0.5 * ((mu - model.prior_mean) / prior_sd) ** 2 - 0.5 * count * ((mu - sample_mean) / noise_sd) ** 2
        return value if math.isfinite(value) else -np.inf

    rows, chains, chain_metadata, failures = [], [], [], []
    for chain, seed in enumerate(seeds):
        predictive_seed = int.from_bytes(hashlib.sha256(f"predictive:{seed}".encode()).digest()[:8], "big")
        predictive_rng = np.random.default_rng(predictive_seed)
        initial_z = (2 * chain / (settings.chains - 1) - 1) * settings.initial_spread
        initial_mu = posterior.mean + sd * initial_z
        previous_accepted = 0
        kept, completed_warmup, accepted_warmup, accepted_posterior = [], 0, 0, 0
        try:
            if not math.isfinite(initial_mu):
                raise ValueError("Unrepresentable chain initialization")
            sampler = emcee.EnsembleSampler(1, 1, log_probability,
                                           moves=emcee.moves.GaussianMove(settings.proposal_scale**2))
            state = emcee.State([[initial_z]], random_state=np.random.RandomState(seed).get_state())
            for index, state in enumerate(sampler.sample(state, iterations=settings.warmup + settings.draws,
                                                        skip_initial_state_check=True, progress=False)):
                z = float(state.coords[0, 0])
                mu = posterior.mean + sd * z
                logp = float(state.log_prob[0])
                future_value = float(predictive_rng.normal(mu, noise_sd))
                if not all(math.isfinite(value) for value in (mu, logp, future_value)):
                    raise ValueError("Sampler produced nonfinite state")
                warmup = index < settings.warmup
                total_accepted = int(sampler.backend.accepted[0])
                accepted = total_accepted > previous_accepted
                previous_accepted = total_accepted
                completed_warmup += int(warmup)
                accepted_warmup += int(warmup and accepted)
                accepted_posterior += int(not warmup and accepted)
                rows.append({"chain": chain, "draw": index if warmup else index - settings.warmup,
                             "warmup": warmup, "mu": mu, "standardized_mu": z,
                             "future_value": future_value, "log_probability": logp, "accepted": accepted})
                if not warmup:
                    kept.append(mu)
            chains.append(kept)
        except Exception as error:
            failures.append({"chain": chain, "error": f"{type(error).__name__}: {error}"})
        chain_metadata.append({"chain": chain, "seed": seed, "initial_standardized_mu": initial_z,
                               "predictive_seed": predictive_seed,
                               "initial_mu": _finite_or_none(initial_mu),
                               "warmup_draws": completed_warmup,
                               "retained_draws": len(kept),
                               "warmup_acceptance_fraction": accepted_warmup / completed_warmup if completed_warmup else None,
                               "posterior_acceptance_fraction": accepted_posterior / len(kept) if kept else None})
    diagnostics = []
    if len(chains) == settings.chains:
        values = np.asarray(chains)
        diagnostics.append(diagnostic_summary(values, name="mu_and_predictive_mean",
                                              reference=posterior.mean, reference_sd=sd, settings=settings))
        # E[(mu - exact_mean)^2] = posterior variance; adding observation noise
        # recovers predictive variance without noisy future-observation draws.
        centered_square = (values - posterior.mean) ** 2
        diagnostics.append(diagnostic_summary(centered_square, name="centered_parameter_second_moment",
                                              reference=posterior.variance,
                                              reference_sd=math.sqrt(2) * posterior.variance, settings=settings))
        future_values = np.asarray([[row["future_value"] for row in rows if row["chain"] == chain and not row["warmup"]]
                                    for chain in range(settings.chains)])
        diagnostics.append(diagnostic_summary(future_values, name="predictive_draw_mean",
                                              reference=posterior.mean, reference_sd=math.sqrt(posterior.predictive_variance),
                                              settings=settings))
        diagnostics.append(diagnostic_summary((future_values - posterior.mean) ** 2,
                                              name="centered_predictive_second_moment",
                                              reference=posterior.predictive_variance,
                                              reference_sd=math.sqrt(2) * posterior.predictive_variance, settings=settings))
        # Rao-Blackwellized predictive CDF and HPD mass functionals. Diagnose the
        # actual integrands, retaining chain/draw dimensions, rather than averaging
        # conditional quantile/HPD endpoints or conflating MCSE with interval width.
        from scipy.special import ndtr
        predictive = NormalDist(posterior.mean, math.sqrt(posterior.predictive_variance))
        for probability in request.quantile_levels:
            boundary = predictive.inv_cdf(probability)
            integrand = ndtr((boundary - values) / noise_sd)
            diagnostics.append(diagnostic_summary(integrand, name=f"predictive_cdf_at_{probability:g}",
                                                  reference=probability, reference_sd=float(integrand.std(ddof=1)),
                                                  settings=settings))
        for level in request.highest_density_levels:
            lower_p, upper_p = (1 - level) / 2, (1 + level) / 2
            integrand = ndtr((predictive.inv_cdf(upper_p) - values) / noise_sd) - ndtr((predictive.inv_cdf(lower_p) - values) / noise_sd)
            diagnostics.append(diagnostic_summary(integrand, name=f"predictive_hpd_mass_{level:g}",
                                                  reference=level, reference_sd=float(integrand.std(ddof=1)),
                                                  settings=settings))
    checks += ({"name": "all_chains_completed", "required": True,
                "status": "passed" if not failures else "failed"},
               {"name": "mcmc_diagnostics_and_exact_reference", "required": True,
                "status": "passed" if diagnostics and all(x["status"] == "passed" for x in diagnostics) else "failed"},
               {"name": "hmc_divergences_energy_tree_depth", "required": False, "status": "not_applicable"})
    metadata = {**metadata, "sampler": {"backend": "emcee", "algorithm": "GaussianMove random-walk Metropolis",
                 "independent_chains": True, "walkers_per_chain": 1,
                 "settings": asdict(settings), "chains": chain_metadata, "failures": failures,
                 "transform": "mu = exact_posterior_mean + exact_posterior_sd * standardized_mu; constant Jacobian omitted",
                 "initialization": "Evenly spaced +/- initial_spread in exact posterior SD units",
                 "adaptation": "none; warmup is saved and excluded from diagnostics",
                 "log_probability": "prior plus sufficient-statistic likelihood; parameter-independent constants omitted",
                 "seed_derivation": "SHA-256 of base seed, eligible UTC timestamp/value history, issuance, target, and chain index; first 32 bits",
                 "versions": {name: version(name) for name in ("emcee", "arviz", "numpy", "scipy")}},
                "sampled_predictive_variance": _finite_or_none(model.observation_variance + float(np.var(chains))) if len(chains) == settings.chains else None,
                "diagnostic_scope": "mu, centered parameter second moment, paired future mean/second moment, requested predictive CDF boundaries and HPD masses; warmup excluded"}
    return FitEvidence(metadata, checks, tuple(diagnostics), tuple(rows))
