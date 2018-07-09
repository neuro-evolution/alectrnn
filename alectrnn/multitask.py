from random import Random


class HumanNormalizationLog:
    """
    This class contains static dictionary with keys for each game, valued by
    the best human score.
    """

    log = {'air_raid': None,
           'alien': -6857,
           'amidar': -1676,
           'assault': -1496,
           'asterix': -8503,
           'asteroids': -13157,
           'atlantis': -29028,
           'bank_heist': -734,
           'battle_zone': -37800,
           'beam_rider': -5775,
           'berzerk': None,
           'bowling': -154,
           'boxing': -4.3,
           'breakout': -31.8,
           'carnival': None,
           'centipede': -11963,
           'chopper_command': -9882,
           'crazy_climber': -35411,
           'defender': None,
           'demon_attack': -3401,
           'double_dunk': -15.5,
           'elevator_action': None,
           'enduro': -309.6,
           'fishing_derby': -5.5,
           'freeway': -29.6,
           'frostbite': -4335,
           'gopher': -2321,
           'gravitar': -2672,
           'hero': -25763,
           'ice_hockey': -0.9,
           'jamesbond': -406.7,
           'journey_escape': None,
           'kangaroo': -3035,
           'krull': -2395,
           'kung_fu_master': -22736,
           'montezuma_revenge': -4367,
           'ms_pacman': -15693,
           'name_this_game': -4076,
           'phoenix': None,
           'pitfall': None,
           'pong': -9.3,
           'pooyan': None,
           'private_eye': -69571,
           'qbert': -13455,
           'riverraid': -13513,
           'road_runner': -7845,
           'robotank': -11.9,
           'seaquest': -20182,
           'skiing': None,
           'solaris': None,
           'space_invaders': -1652,
           'star_gunner': -10250,
           'tennis': -8.9,
           'time_pilot': -5925,
           'tutankham': -167.6,
           'up_n_down': -9082,
           'venture': -1188,
           'video_pinball': -17298,
           'wizard_of_wor': -4757,
           'yars_revenge': None,
           'zaxxon': -9173}


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


class RandomRomObjective:
    """
    When called it plays a random game and returns a normalized score
    """

    def __init__(self, roms, ale_handler, agent_handler, objective_handler,
                 score_normalizer, seed):
        """
        :param roms: a sequence of rom names to random choose from
        :param ale_handler: a built ale handler object
        :param agent_handler: a built agent handler object
        :param objective_handler: a built objective handler object
        :param score_normalizer: an instance of a score normalizer
        :param seed: seed for the random number generator

        *built means the create() method was called and the handle exists.
        """
        self.roms = roms
        self._ale_handler = ale_handler
        self._agent_handler = agent_handler
        self._objective_handler = objective_handler
        self.score_normalizer = score_normalizer
        self._rng = Random(seed)

    def __call__(self, parameters):
        """
        Selects a random rom from the roms list, updates the handlers, and
        then runs the objective.
        :param parameters: parameters for objective function
        :return: score
        """

        chosen_rom = self._rng.choice(self.roms)
        self._ale_handler.set_parameters({'rom': chosen_rom,
                                         'seed': self._rng.randint(1, 2000000)})
        # Update handles
        self._agent_handler.ale = self._ale_handler.handle
        self._objective_handler.agent = self._agent_handler.handle
        self._objective_handler.ale = self._ale_handler.handle

        return self.score_normalizer(self._objective_handler.handle(parameters),
                                     chosen_rom)


class MultiRomObjective:
    """
    Runs multiple game for each objective and returns the sum of their
    normalized performance.
    """

    def __init__(self, rom_objective_map, score_normalizer):
        """
        :param rom_objective_map: a dictionary keyed by rom and valued by an
            objective handler
        :param score_normalizer: an instance of a score normalizer
        """
        self.rom_objective_map = rom_objective_map
        self.score_normalizer = score_normalizer

    def __call__(self, parameters):
        """
        :param parameters: objective parameters
        :return: sum of normalized scores
        """
        return sum([self.score_normalizer(objective.handle(parameters), rom)
                    for rom, objective in self.rom_objective_map.items()])
