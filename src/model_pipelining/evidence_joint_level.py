"""Evidence adapter for categorical evolution-variance learning."""
from dataclasses import replace
from pathlib import Path
from model_pipelining.evidence_local_level import LocalLevelVarianceEvidence


class JointLevelEvidence(LocalLevelVarianceEvidence):
    def __init__(self,model):
        super().__init__(model)
        self.metadata=replace(self.metadata,family="local_level_joint_variance",workflows=("sequential_checks",),
            implementation_files=self.metadata.implementation_files+(
                Path(__file__),Path(__file__).with_name('local_level_joint_variance.py')))
