"""Joint variance learning with an explicit categorical evolution-variance prior.

This is a discrete model for W, not quadrature pretending to be a continuous
prior. Every support point integrates V and the state using the checked scalar
adapter. Posterior support probabilities use full marginal likelihoods.
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from numbers import Real
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from model_pipelining.inference import FitEvidence, FittedForecast
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings, NumericalIntegrationError
from model_pipelining.mixtures import NormalMixture


@dataclass(frozen=True)
class JointVariancePosterior:
    model: object
    history: tuple
    components: tuple
    evolution_weights: tuple[float, ...]
    log_evidence: float
    evidence: dict
    passed: bool

    @property
    def weights(self):
        return tuple(a*b for a, component in zip(self.evolution_weights,self.components) for b in component.weights)

    @property
    def variances(self):
        return tuple(value for component in self.components for value in component.variances)

    @property
    def level_means(self):
        return tuple(value for component in self.components for value in component.level_means)

    @property
    def level_variances(self):
        return tuple(value for component in self.components for value in component.level_variances)

    @property
    def level_mean(self):
        return math.fsum(w*p.level_mean for w,p in zip(self.evolution_weights,self.components))

    @property
    def level_variance(self):
        return math.fsum(w*(p.level_variance+(p.level_mean-self.level_mean)**2)
                         for w,p in zip(self.evolution_weights,self.components))

    @property
    def observation_variance_mean(self):
        values=[p.observation_variance_mean for p in self.components]
        return None if any(value is None for value in values) else math.fsum(w*v for w,v in zip(self.evolution_weights,values))

    @property
    def evolution_variance_mean(self):
        return math.fsum(w*v for w,v in zip(self.evolution_weights,self.model.evolution_variances))

    @property
    def variance_covariance(self):
        if self.observation_variance_mean is None:
            return None
        return math.fsum(w*v*p.observation_variance_mean for w,v,p in
            zip(self.evolution_weights,self.model.evolution_variances,self.components))-self.evolution_variance_mean*self.observation_variance_mean

    def evolution_cdf(self, value, *, inclusive=True):
        result = math.fsum(p for v,p in zip(self.model.evolution_variances,self.evolution_weights)
                          if (v <= value if inclusive else v < value))
        return min(1., max(0., result))

    def evolution_ppf(self, probability):
        if not 0 < probability < 1:
            raise ValueError("Probability must be in (0,1)")
        return self.model.evolution_variances[min(int(np.searchsorted(np.cumsum(self.evolution_weights),probability)),len(self.evolution_weights)-1)]

    def cdf(self, value):
        return math.fsum(w*p.cdf(value) for w,p in zip(self.evolution_weights,self.components))

    def ppf(self, probability):
        if isinstance(probability,bool) or not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("Probability must be in (0,1)")
        values=[p.ppf(probability) for p in self.components]
        if max(values)==min(values):
            return values[0]
        low,high=math.log(min(values)),math.log(max(values))
        # Conditional quantiles are numerical roots. If one W component has
        # almost all the mass, its small root error can put both endpoints on
        # the same side of the marginal target. Widen only the failing side to
        # the checked integration support; keep the actual CDF root tolerance.
        if self.cdf(math.exp(low))>probability:
            low=min(p.log_bounds[0] for p in self.components)
        if self.cdf(math.exp(high))<probability:
            high=max(p.log_bounds[1] for p in self.components)
        root=math.exp(brentq(lambda z:self.cdf(math.exp(z))-probability,
                            low,high,xtol=1e-10,rtol=1e-12))
        if abs(self.cdf(root)-probability)>10*self.model.quadrature.relative_tolerance:
            raise ValueError("Joint variance quantile precision failed")
        return root

    def predictive(self, elapsed_days=1.):
        laws=[p.predictive(elapsed_days) for p in self.components]
        return NormalMixture(tuple(a*b for a,law in zip(self.evolution_weights,laws) for b in law.weights),
                             tuple(m for law in laws for m in law.means),tuple(v for law in laws for v in law.variances))


@dataclass(frozen=True, kw_only=True)
class LocalLevelJointVariance:
    initial_mean: float = 0.
    initial_variance: float = 100.
    prior_shape: float = 3.
    prior_scale: float = 2.
    evolution_variances: tuple[float,...] = (0.,.01,.1,1.)
    evolution_weights: tuple[float,...] | None = None
    quadrature: QuadratureSettings = field(default_factory=QuadratureSettings)
    name: ClassVar[str] = "local-level-joint-variance"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True
    supports_fit_evidence: ClassVar[bool] = True
    supports_distribution: ClassVar[bool] = True

    def __post_init__(self):
        values=self.evolution_variances
        if (not isinstance(values,tuple) or not values or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v<0 for v in values)
                or tuple(sorted(set(values)))!=values):
            raise ValueError("Evolution variance support must be ordered unique finite nonnegative values")
        weights=tuple(1. for _ in values) if self.evolution_weights is None else self.evolution_weights
        if (not isinstance(weights,tuple) or len(weights)!=len(values)
                or any(isinstance(w,bool) or not isinstance(w,Real) or not math.isfinite(w) or w<=0 for w in weights)):
            raise ValueError("Choose a positive finite prior weight for every support point")
        total=math.fsum(weights)
        if not math.isfinite(total):
            raise ValueError("Unrepresentable evolution prior normalization")
        object.__setattr__(self,"evolution_weights",tuple(w/total for w in weights))
        if any(w<=0 for w in self.evolution_weights):
            raise ValueError("Evolution prior probabilities underflowed")
        self._conditional(values[0])  # Delegate scalar-prior/state validation.

    def _conditional(self, variance):
        return LocalLevelVariance(initial_mean=self.initial_mean,initial_variance=self.initial_variance,
            evolution_variance=variance,prior_shape=self.prior_shape,prior_scale=self.prior_scale,quadrature=self.quadrature)

    @property
    def configuration(self):
        return asdict(self)

    @classmethod
    def from_configuration(cls,value):
        value=dict(value)
        for name in ("evolution_variances","evolution_weights"):
            if value.get(name) is not None:value[name]=tuple(value[name])
        if "quadrature" in value:value["quadrature"]=QuadratureSettings(**value["quadrature"])
        return cls(**value)

    @property
    def implementation_files(self):
        return (Path(__file__).with_name("local_level_variance.py"),Path(__file__).with_name("mixtures.py"))

    def _request(self,request):
        return self._conditional(self.evolution_variances[0])._request(request)

    def _integrate(self,history):
        components=[]
        for variance in self.evolution_variances:
            try:
                components.append(self._conditional(variance)._integrate(history))
            except (ValueError,OverflowError,FloatingPointError) as error:
                raise NumericalIntegrationError(str(error),{"failed_evolution_variance":variance,
                    "completed_components":[p.evidence for p in components],"cause":getattr(error,"evidence",None)}) from error
        log_weights=np.log(self.evolution_weights)+[p.log_evidence for p in components]
        log_evidence=float(logsumexp(log_weights))
        weights=tuple(map(float,np.exp(log_weights-log_evidence)))
        passed=all(p.passed for p in components)
        evidence={"method":"Categorical W enumeration with checked conditional log-V quadrature",
            "converged":passed,"nodes":sum(len(p.weights) for p in components),
            "tail_probability_bound":math.fsum(w*p.evidence['tail_probability_bound'] for w,p in zip(weights,components)),
            "evolution_prior_weights":self.evolution_weights,"evolution_posterior_weights":weights,
            "evolution_variances":self.evolution_variances,
            "components":[p.evidence for p in components],
            "scope":"W is an explicit discrete prior, not a continuous-prior quadrature or tuned grid"}
        return JointVariancePosterior(self,tuple(history),tuple(components),weights,log_evidence,evidence,passed)

    def posterior(self,request):
        self._request(request)
        result=self._integrate(request.history)
        if not result.passed:raise NumericalIntegrationError("Required joint integration checks failed",result.evidence)
        return result

    def prior(self):
        result=self._integrate(())
        if not result.passed:raise NumericalIntegrationError("Required joint prior integration checks failed",result.evidence)
        return result

    def predictive_distribution(self,request):
        return self.posterior(request).predictive(self._request(request))

    def fit(self,request):
        # Reuse the checked marginal forecast/HPD calculation on the flattened
        # joint posterior; add the W identity absent from scalar-node records.
        result=LocalLevelVariance.fit(self,request)
        metadata=dict(result.evidence.metadata)
        metadata['inference']='categorical_evolution_and_log_observation_variance'
        metadata['evolution_variance_posterior']=list(zip(self.evolution_variances,
            metadata['quadrature']['evolution_posterior_weights']))
        metadata['posterior_variance_covariance']=self._variance_covariance_from_nodes(metadata)
        index=0
        for variance,component in zip(self.evolution_variances,metadata['quadrature']['components']):
            for _ in range(component['nodes']):
                metadata['quadrature_nodes'][index]['evolution_variance']=variance
                index+=1
        return FittedForecast(result.forecast,FitEvidence(metadata,result.evidence.checks))

    def _variance_covariance_from_nodes(self,metadata):
        if metadata['observation_variance_mean'] is None:return None
        offset=0;evw=ew=0.
        for variance,component in zip(self.evolution_variances,metadata['quadrature']['components']):
            nodes=metadata['quadrature_nodes'][offset:offset+component['nodes']]
            evw+=variance*math.fsum(node['weight']*node['variance'] for node in nodes)
            ew+=variance*math.fsum(node['weight'] for node in nodes)
            offset+=component['nodes']
        return evw-ew*metadata['observation_variance_mean']

    def forecast(self,request):
        result=self.fit(request)
        if any(check['status']!='passed' for check in result.evidence.checks if check['required']):
            raise ValueError("Required joint inference checks failed; inspect fit evidence")
        return result.forecast

    def evidence_provider(self):
        from model_pipelining.evidence_joint_level import JointLevelEvidence
        return JointLevelEvidence(self)


def plot_joint_variances(run_directory):
    """Read saved per-origin categorical weights, variances and confounding only."""
    import csv
    import json
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory=Path(run_directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('status') not in ('completed','failed') or manifest.get('model',{}).get('name')!='local-level-joint-variance':
        raise ValueError('Finished joint-variance forecast run required')
    destination=directory/'joint_variances.png'
    if destination.exists():raise FileExistsError(destination)
    with (directory/'fit_index.csv').open() as stream:rows=list(csv.DictReader(stream))
    support=manifest['model']['configuration']['evolution_variances']
    probabilities=[[] for _ in support];observation=[];evolution=[];covariance=[]
    for row in rows:
        metadata=None
        if row['status']=='passed' and row['fit_directory']:
            metadata=json.loads((directory/row['fit_directory']/'fit.json').read_text())['metadata']
        weights=metadata['quadrature']['evolution_posterior_weights'] if metadata else [float('nan')]*len(support)
        for values,weight in zip(probabilities,weights):values.append(weight)
        observation.append(metadata['observation_variance_mean'] if metadata and metadata['observation_variance_mean'] is not None else float('nan'))
        evolution.append(math.fsum(w*v for w,v in zip(weights,support)))
        covariance.append(metadata['posterior_variance_covariance'] if metadata and metadata['posterior_variance_covariance'] is not None else float('nan'))
    figure=Figure(figsize=(11,9),layout='constrained');FigureCanvasAgg(figure)
    axes=figure.subplots(3,1,sharex=True)
    for value,weights in zip(support,probabilities):axes[0].plot(weights,label=f'W={value:g}')
    axes[0].set(ylabel='Posterior probability',ylim=(0,1));axes[0].legend()
    axes[1].plot(observation,label='Observation variance');axes[1].plot(evolution,label='Evolution variance')
    axes[1].set_ylabel('Posterior expected variance');axes[1].legend()
    axes[2].plot(covariance);axes[2].axhline(0,color='grey',linewidth=.7)
    axes[2].set(xlabel='Forecast case',ylabel='Posterior covariance of V and W')
    figure.suptitle(f"Joint variance learning — {manifest['status']}; discrete W support; review pending")
    with destination.open('xb') as stream:figure.savefig(stream,format='png',dpi=150)
    return destination
