"""Contextual motion deviation: the core anomaly-scoring idea of this project.

An object is NOT flagged simply because it is moving fast. Its motion is
compared against the normal baseline motion of the region it is currently
in. A person sprinting through a normally-busy, fast-moving region (e.g. a
road) should score low; the same speed in a region that's normally static
(e.g. a footpath where people stroll) should score high.

We compute two complementary z-scores and combine them:

  z_object  = |object_speed        - mu_region| / (sigma_region + eps)
      "Is THIS object's motion unusual for where it is?" -- the primary
      signal, since it's tied to a specific tracked person.

  z_region  = |current_regional_mag - mu_region| / (sigma_region + eps)
      "Is the region's aggregate optical flow unusual right now?" -- a
      secondary, context signal (e.g. a whole area suddenly getting busy/
      empty, independent of any one track).

anomaly_score_raw = w_obj * z_object + w_reg * z_region
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviationResult:
    z_object: float
    z_region: float
    anomaly_score_raw: float


def z_score(x: float, mu: float, sigma: float, eps: float, max_z: float = 50.0) -> float:
    """|x - mu| / (sigma + eps), capped at max_z.

    Regions with near-zero training variance (e.g. a static background patch
    that never moves in the normal videos) have sigma ~= 0, so `eps` alone
    would let a single small, real deviation blow up to an astronomical
    z-score (division by a near-zero denominator). That doesn't add
    information -- it's already "maximally anomalous" -- so we cap it at
    `max_z`. This keeps scores on a comparable, finite scale across regions
    regardless of how static their training data was, which matters both for
    the fixed anomaly.score_threshold and for EMA smoothing (an unbounded
    input would otherwise dominate the exponential moving average for many
    frames after a single such spike).
    """
    raw = abs(x - mu) / (sigma + eps)
    return float(min(raw, max_z))


def compute_contextual_deviation(
    object_speed: float,
    region_current_mag: float,
    region_mu: float,
    region_sigma: float,
    eps: float,
    weight_object_motion: float,
    weight_regional_motion: float,
    max_z: float = 50.0,
) -> DeviationResult:
    z_obj = z_score(object_speed, region_mu, region_sigma, eps, max_z)
    z_reg = z_score(region_current_mag, region_mu, region_sigma, eps, max_z)
    raw = weight_object_motion * z_obj + weight_regional_motion * z_reg
    return DeviationResult(z_object=z_obj, z_region=z_reg, anomaly_score_raw=raw)


def regional_only_score(
    region_current_mag: float, region_mu: float, region_sigma: float, eps: float, max_z: float = 50.0
) -> float:
    """The 'baseline system' comparator: optical flow -> regional motion ->
    anomaly score, with NO object detection/tracking involved at all.
    """
    return z_score(region_current_mag, region_mu, region_sigma, eps, max_z)
