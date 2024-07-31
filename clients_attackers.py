from __future__ import print_function

import torch
import torch.nn.functional as F

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
from clients import *

class Attacker_Backdoor(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        super(Attacker_Backdoor, self).__init__(cid, model, dataLoader, optimizer, criterion, device, inner_epochs)
        self.utils = Backdoor_Utils()

    def data_transform(self, data, target):
        data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=0.5,
                                                   backdoor_label=self.utils.backdoor_label)
        return data, target
