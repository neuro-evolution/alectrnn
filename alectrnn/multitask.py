class HumanNormalizationLog:
    """
    This class contains static dictionary with keys for each game, valued by
    the best human score.
    """

    log = {}


class BestAINormalizationLog:
    """
    This class contains static dictionary with keys for each game, valued by
    the best AI score.
    """

    log = {}


class ScoreNormalizer:
    """
    Contains methods for normalizing scores based on a log
    """

    def __init__(self, normalization_log):
        """
        :param normalization_log: a class containing a log class attribute
        """

        self.normalization_log = normalization_log

    def normalize_score(self, score, rom, clip=False):
        """
        :param score: a scalar
        :param rom: string representing rom name
        :param clip: whether to clip the maximum score to 1. Default: F
        :return: the normalized score
        """

        normalize_score = (score - self.normalization_log.log[rom]) \
                          / self.normalization_log.log[rom]
        if clip:
            return min(1., normalize_score)

        return normalize_score


def rescale(x, new_min, new_max, old_min, old_max):
    """
    Rescales a value from an old range [A,B] to a new range [C,D] using the equation:

    x' = C(1 - (x-A)/(B-A)) + D((x-A)/(B-A))

    :param x: value to be scaled
    :param new_min: new range minimum (C)
    :param new_max: new range maximum (D)
    :param old_min: old range minimum (A)
    :param old_max: old range maximum (B)
    :return: rescaled value
    """

    return new_min * (1 - (x-old_min)/(old_max-old_min)) \
           + new_max*((x-old_min)/(old_max-old_min))
