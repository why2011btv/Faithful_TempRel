import torch
from torch import nn, optim
from torch.nn import functional as F
import os
import json
import numpy as np

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def func(vector, small = 0.001):
    # small is one of the hyperparameters!
    ini_array1 = np.array(vector)
    ini_array1 = np.where(ini_array1<0, small, ini_array1)
    ini_array1 = np.where(ini_array1>1, 1-small, ini_array1)
    ini_array1 = np.log(ini_array1)
    return ini_array1
    
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, temp = 0.5):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temp)

    def forward(self, logits):
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        logits = torch.from_numpy(logits)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        """
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
        """
        
        labels = [torch.tensor(label).unsqueeze(0) for label in labels]
        labels = torch.cat(labels, dim=0).long().cuda()
        logits = torch.from_numpy(logits).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self.temperature, self.temperature_scale(logits)
    
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
"""
json_file = 'MATRES_nd-11453-micro-test-0511pm-lr5e-6-b20-gpu9942-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask0.json'
with open("/shared/why16gzl/Repositories/LEC_OnePass/prediction/" + json_file) as f:
    prediction = json.load(f)
    
if json_file[0:3] == 'MAT':
    label_file = "/shared/why16gzl/Repositories/LEC_OnePass/test_labels.txt"
else:
    label_file = "/shared/why16gzl/Repositories/LEC_OnePass/tdd_labels.txt"
    
labels = []
logits = []
subtracted_logits = prediction['array'][1:]
with open(label_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        label = int(line[0])
        if label != 3:
            labels.append(label)
            logits.append(func(subtracted_logits[i]))
logits = np.array(logits)
scaled_model = ModelWithTemperature()
#logits = scaled_model.temperature_scale(logits)
#logits = logits.cpu().detach().numpy()
T, scaled_logits = scaled_model.set_temperature(logits, labels)
logits = scaled_logits.cpu().detach().numpy()

with open('prediction/'+json_file[0:-5]+"-scaled.json", 'w') as outfile:
    numpyData = {"labels": "0 -- Parent-Child or Before; 1 -- Child-Parent or After; 2 -- Coref or Simultaneous; 3 -- NoRel or Vague", "array": logits}
    json.dump(numpyData, outfile, cls=NumpyArrayEncoder)
"""
