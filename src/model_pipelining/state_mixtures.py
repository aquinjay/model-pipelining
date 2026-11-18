"""Gaussian state mixtures including genuinely known-state probability atoms."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr,ndtri


@dataclass(frozen=True)
class StateMixture:
    weights: tuple
    means: tuple
    variances: tuple

    def __post_init__(self):
        w,m,v=(np.asarray(value,dtype=float) for value in (self.weights,self.means,self.variances))
        if (w.ndim!=1 or not len(w) or m.shape!=w.shape or v.shape!=w.shape or
                not all(np.isfinite(a).all() for a in (w,m,v)) or np.any(w<0) or np.any(v<0) or
                not math.isfinite(float(w.sum())) or w.sum()<=0):
            raise ValueError('Invalid Gaussian/point state mixture')
        positive=w>0
        object.__setattr__(self,'weights',tuple(map(float,w[positive]/w.sum())))
        object.__setattr__(self,'means',tuple(map(float,m[positive])))
        object.__setattr__(self,'variances',tuple(map(float,v[positive])))

    def cdf(self,value,*,inclusive=True):
        if math.isnan(value):raise ValueError('State CDF value cannot be NaN')
        result=math.fsum(w*(float(ndtr((value-m)/math.sqrt(v))) if v>0 else float(value>=m if inclusive else value>m))
                         for w,m,v in zip(self.weights,self.means,self.variances))
        return min(1.,max(0.,result))

    def ppf(self,probability):
        if isinstance(probability,bool) or not math.isfinite(probability) or not 0<probability<1:
            raise ValueError('State probability must be in (0,1)')
        for atom in sorted(set(m for m,v in zip(self.means,self.variances) if v==0)):
            if self.cdf(atom,inclusive=False)<probability<=self.cdf(atom):
                return atom
        bounds=[m+math.sqrt(v)*float(ndtri(probability)) for m,v in zip(self.means,self.variances)]
        low,high=min(bounds),max(bounds)
        if low==high:
            if abs(self.cdf(low)-probability)>1e-8:
                raise ValueError('State quantile is not representable at this location/scale')
            return low
        for bound in (low,high):
            if abs(self.cdf(bound)-probability)<=32*float(np.spacing(probability)):return bound
        scale=min((math.sqrt(v) for v in self.variances if v>0),default=1.)
        result=float(brentq(lambda x:self.cdf(x)-probability,low,high,xtol=max(scale*1e-10,1e-150),rtol=1e-13))
        if not self.cdf(result,inclusive=False)-1e-8<=probability<=self.cdf(result)+1e-8:
            raise ValueError('State quantile probability check failed')
        return result
