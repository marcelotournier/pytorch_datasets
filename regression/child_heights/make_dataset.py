"""
Script to make the PyTorch child_heights dataset

author: Marcelo Tournier
license: MIT
date: 2022-10-18
dependencies: check requirements.txt
"""
import torch

# Generate the height dataset in centimeters:
inch = 2.54

women_mean = 64.5 * inch
women_std = 2.5 * inch

men_mean = 70 * inch
men_std = 3 * inch

sample_size = 100000

post_estimation_bias = 2 # value to be multiplied to normal dist

# Create the sampling distributions for the variables:
women_dist = torch.distributions.Normal(women_mean, women_std)
men_dist = torch.distributions.Normal(men_mean, men_std)
female_dist = torch.distributions.Bernoulli(0.5)

# Generating samples for features and targets
def make_sample(n):
    return torch.stack((women_dist.sample((n,)),
            men_dist.sample((n,)), 
            female_dist.sample((n,))), dim=1)
  

def estimate_height(t):
    if t[2] == 0:
      modifier = 13
    else:
      modifier = -13
    return ((t[0] + t[1] + modifier) / 2) + 2 * torch.randn(1)


features = make_sample(sample_size)
targets = torch.tensor([estimate_height(t) for t in features]).unsqueeze(1)

# Saving tensors
torch.save(features, "height_features.pt")
torch.save(targets, "height_targets.pt")
