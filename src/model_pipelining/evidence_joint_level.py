"""Evidence adapter for categorical evolution-variance learning."""
from dataclasses import replace
from pathlib import Path
import math
import json
import numpy as np
from model_pipelining.evidence import PredictiveReplications,ReplicatedDataset
from model_pipelining.local_level_variance import initial_posterior_components
from model_pipelining.evidence_local_level import LocalLevelVarianceEvidence
from model_pipelining.models import ForecastRequest


class JointLevelEvidence(LocalLevelVarianceEvidence):
    def __init__(self,model):
        super().__init__(model)
        self.metadata=replace(self.metadata,family="local_level_joint_variance",
            workflows=("sequential_checks","predictive_checks","prior_influence"),
            parameter_fields=("mean","observation_variance","initial_level","evolution_variance"),
            variance_prior_settings=("prior_shape","prior_scale","evolution_weights"),
            predictive_scope={"generation":"One joint W,V,initial-state draw per fresh process path",
                "posterior":"Enumerated W probabilities, conditional V quadrature and Gaussian initial-state smoothing; fresh process/measurement noise"},
            influence_scope={**self.metadata.influence_scope,
                "inference":"Explicit categorical W prior and checked conditional V integration",
                "intervals":"State/V equal-tail credible; W discrete equal-tail credible; observation-predictive mixture HPD"},
            presentation={**self.metadata.presentation,"evolution_variance":True,
                "intervals":"state/V equal-tail and discrete-W credible; predictive mixture HPD",
                "generation":"one joint evolution/noise/initial-state draw per fresh trajectory"},
            implementation_files=self.metadata.implementation_files+(
                Path(__file__),Path(__file__).with_name('local_level_joint_variance.py')))

    def simulate(self,*,observation_count,replications,seed,request=None):
        model=self.model
        kind='prior' if request is None else 'posterior'
        posterior=None if request is None else model.posterior(request)
        initial_components=None if posterior is None else [initial_posterior_components(p) for p in posterior.components]
        gaps=[1.]*observation_count
        if request is not None:
            gaps[1:]=[(b.timestamp-a.timestamp).total_seconds()/86400 for a,b in zip(request.history,request.history[1:])]
        datasets,failures=[],[]
        for index in range(replications):
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,0 if kind=='prior' else 1,index])))
            try:
                weights=model.evolution_weights if posterior is None else posterior.evolution_weights
                j=int(rng.choice(len(weights),p=np.asarray(weights)/sum(weights)))
                evolution=model.evolution_variances[j]
                if posterior is None:
                    variance=model.prior_scale/float(rng.gamma(model.prior_shape))
                    initial=float(rng.normal(model.initial_mean,math.sqrt(model.initial_variance)))
                else:
                    component=posterior.components[j]
                    k=int(rng.choice(len(component.weights),p=np.asarray(component.weights)/sum(component.weights)))
                    variance=component.variances[k]
                    means,variances=initial_components[j]
                    initial=float(rng.normal(means[k],math.sqrt(variances[k])))
                if not math.isfinite(variance) or variance<=0 or not math.isfinite(initial):
                    raise ValueError('Unrepresentable joint prior/posterior draw')
                state=initial;levels=[];values=[]
                for gap in gaps:
                    state=float(rng.normal(state,math.sqrt(evolution*gap)))
                    levels.append(state);values.append(float(rng.normal(state,math.sqrt(variance))))
                if not all(math.isfinite(x) for x in (*levels,*values)):raise ValueError('Nonfinite fresh trajectory')
                datasets.append(ReplicatedDataset(index,None,variance,tuple(values),initial,tuple(levels),evolution))
            except (ValueError,OverflowError,FloatingPointError,ZeroDivisionError) as error:
                failures.append({'replication':index,'error':f'{type(error).__name__}: {error}'})
        distribution={'family':'joint_local_level_variances','configuration':model.configuration,
            'elapsed_days':gaps,'quadrature':posterior.evidence if posterior else None,
            'replication':'Joint W,V,initial state with fresh process innovations and measurement noise'}
        return PredictiveReplications(kind,seed,replications,observation_count,distribution,tuple(datasets),tuple(failures))

    def summarize(self,case,level):
        # Compute this posterior once: the base summary and extra W quantities
        # must describe the same checked fit.
        posterior=self.model.prior() if case is None else self.model.posterior(
            ForecastRequest(case.history,case.issued_at,case.target_time))
        from model_pipelining.evidence_local_level import _local_summary
        result=_local_summary(self.model,case,level,posterior=posterior)
        result.update(evolution_variance_mean=posterior.evolution_variance_mean,
            evolution_variance_median=posterior.evolution_ppf(.5),
            evolution_variance_lower=posterior.evolution_ppf((1-level)/2),
            evolution_variance_upper=posterior.evolution_ppf((1+level)/2),
            evolution_probabilities_json=json.dumps(list(zip(self.model.evolution_variances,posterior.evolution_weights))),
            variance_covariance=posterior.variance_covariance)
        return result

    def validate_scenarios(self,models):
        if any(type(model) is not type(self.model) or model.evolution_variances!=self.model.evolution_variances
               or model.quadrature!=self.model.quadrature for model in models):
            raise ValueError('Joint prior comparisons require identical evolution support and numerical settings')
