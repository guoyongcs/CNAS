import dataclasses
import json
from datetime import datetime


@dataclasses.dataclass
class SearchSample:
    start_time: str = None
    end_time: str = None
    elapsed_time: str = None
    eid: int = None
    description: str = None

    def start(self):
        self.start_time = datetime.now()
        return self

    def end(self):
        self.end_time = datetime.now()
        return self

    @staticmethod
    def from_file(filename):
        with open(filename, "r") as f:
            d = json.loads(f.read())
        sample = ArchSample()
        sample.__dict__.update(d)
        return sample

    def to_file(self, filename, transform=False):
        if transform:
            if self.start_time is not None and self.end_time is not None:
                self.elapsed_time = self.end_time-self.start_time
                self.elapsed_time, *_ = str(self.elapsed_time).split(".")
                self.start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
                self.end_time = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as f:
            f.write(json.dumps(self.__dict__, indent=2))


@dataclasses.dataclass
class ArchSample:
    start_time: str = None
    end_time: str = None
    elapsed_time: str = None
    description: str = None
    type_: str = None
    # eid: int = None
    sample_id: int = None
    arch: str = None

    derived_acc: float = None

    normal_logp: float = None
    node_normal_entropy: float = None
    op_normal_entropy: float = None
    reduced_logp: float = None
    node_reduced_entropy: float = None
    op_reduced_entropy: float = None

    n_channels: int = None
    n_layers: int = None
    n_params: int = None
    flops: int = None

    eval_acc: float = None
    eval_epoch: int = None

    def start(self):
        self.start_time = datetime.now()
        return self

    def end(self):
        self.end_time = datetime.now()
        return self

    @staticmethod
    def from_file(filename):
        with open(filename, "r") as f:
            d = json.loads(f.read())
        sample = ArchSample()
        sample.__dict__.update(d)
        return sample

    def to_file(self, filename, transform=False):
        if transform:
            if self.start_time is not None and self.end_time is not None:
                self.elapsed_time = self.end_time-self.start_time
                self.elapsed_time, *_ = str(self.elapsed_time).split(".")
                self.start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
                self.end_time = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as f:
            f.write(json.dumps(self.__dict__, indent=2))
