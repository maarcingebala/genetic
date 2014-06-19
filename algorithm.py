# coding: utf-8

import os
import csv
import copy
import random
import logging
import argparse
from functools import wraps
from mio.genetic.chromosome import Chromosome, Population

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('genetic')

if os.environ.get('DEBUG', False):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def ensure_valid_result(f):
    MAX_TRIES = 50

    def one_or_many(result):
        if len(result) > 1:
            return result
        else:
            return result[0]

    @wraps(f)
    def _inner(*args, **kwargs):
        input_ch = []
        for arg in args:
            if isinstance(arg, Chromosome):
                input_ch.append(arg)
        i = 0
        is_correct = False
        while not is_correct:
            result = f(*args, **kwargs)
            if not isinstance(result, list):
                result = [result]
            is_correct = all([ch.is_correct() for ch in result])
            if not is_correct:
                if i == MAX_TRIES:
                    logger.warning('FAILED to use operator %s. Returning original parameters.' % f)
                    return one_or_many(input_ch)
                i += 1
                logger.debug('INVALID results of operation %s. Retrying %s' % (f, i))
        return one_or_many(result)
    return _inner


def selection_roulette(population):
    selected = []

    b = population.pop_best()
    selected.append(b)
    logger.debug('Selected best: %s' % b)

    fitnesses = [c.fitness() for c in population]
    probabilities = [float(f)/sum(fitnesses) for f in fitnesses]
    roulette_circle = []
    for p in probabilities:
        try:
            last = roulette_circle[-1]
        except IndexError:
            last = 0
        roulette_circle.append(p + last)
    roulette_rand = [random.random() for _ in range(len(population))]
    for r in roulette_rand:
        for i, c in enumerate(roulette_circle):
            if r < c:
                selected.append(population[i])
                break
    return selected


@ensure_valid_result
def crossover(chx, chy, locus=None):
    x = list(chx.genotype)
    y = list(chy.genotype)
    if not locus:
        locus = random.randint(1, len(x) - 2)
    x2 = ''.join(x[:locus] + y[locus:])
    y2 = ''.join(y[:locus] + x[locus:])
    
    ch_data = chx.get_properties()
    chx2 = Chromosome(genotype=x2, **ch_data)
    chy2 = Chromosome(genotype=y2, **ch_data)

    if any([chx.special, chy.special]):
        special = chx if chx.special else chy
        return [special, Population([chx2, chy2]).best_ch()]
    else:
        return [chx2, chy2]


@ensure_valid_result
def crossover_ux(chx, chy, ratio=0.5):
    x2, y2 = [], []
    ch_data = chx.get_properties()
    
    for x, y in zip(list(chx.genotype), list(chy.genotype)):
        if random.random() <= ratio:
            x2.append(y)
            y2.append(x)
        else:
            x2.append(x)
            y2.append(y)

    x2 = ''.join(x2)
    y2 = ''.join(y2)

    chx2 = Chromosome(genotype=x2, **ch_data)
    chy2 = Chromosome(genotype=y2, **ch_data)

    if any([chx.special, chy.special]):
        special = chx if chx.special else chy
        return [special, Population([chx2, chy2]).best_ch()]
    else:
        return [chx2, chy2]


@ensure_valid_result
def mutation(ch, locus=None):
    new_ch = copy.deepcopy(ch)
    new_ch.special = False
    if locus is None:
        locus = random.randint(0, len(new_ch.genotype) - 1)
    genotype = list(new_ch.genotype)
    genotype[locus] = str(int(int(genotype[locus]) != 1))
    new_ch.genotype = ''.join(genotype)
    if ch.special:
        return Population([ch, new_ch]).best_ch()
    return new_ch


class Algorithm(object):

    def __init__(self, chromosome_data={}, px=0.75, pm=0.1, selection_f=selection_roulette, cross_f=crossover, mutation_f=mutation):
        self.px = px
        self.pm = pm
        self.chromosome_data = chromosome_data
        self.selection_f = selection_f
        self.cross_f = cross_f
        self.mutation_f = mutation_f
        self.stats = {}

    def check_stop_condition(self, population):
        b = population.best_ch()
        return b.collisions() == 0 and b.gaps() < 3

    def make_new_population(self, old_population):
        new_pop = Population()
        new_pop_len = len(old_population)
        temp_pop = Population(self.selection_f(old_population))
        logger.debug('Selection: %s' % temp_pop)
        
        for _ in range(new_pop_len / 2):
            chx = chy = None
            while chx == chy:
                chx = random.choice(temp_pop)
                chy = random.choice(temp_pop)
            _px = random.random()
            if _px <= self.px:
                chx2, chy2 = self.cross_f(chx, chy)
                new_pop.extend([chx2, chy2])
                logger.debug('Crossing (%s): %s x %s -> (%s, %s)' % (_px, chx, chy, chx2, chy2))
            else:
                new_pop.extend([chx, chy])
                logger.debug('Passing: %s , %s' % (chx, chy))

        for i in range(new_pop_len):
            _pm = random.random()
            if _pm <= self.pm:
                ch = new_pop[i]
                logger.debug('Mutation: %s' % ch)
                ch = self.mutation_f(ch)
                new_pop[i] = ch
                logger.debug('Mutated : %s' % ch)

        return new_pop

    def save_stats(self, population, no):
        self.stats[no] = {
            'worst': population.worst_f(),
            'best': population.best_f(),
            'avg': population.avg_f(),
        }
        self.best_ch = population.best_ch()
        population[population.index(self.best_ch)].special = True

    def to_csv(self, fname='last_results.csv', data={}):
        stats = data or self.stats
        if not stats:
            print "Empty stats"
            return
        keys = stats[1].keys()
        with open(fname, 'wr') as f:
            wr = csv.DictWriter(f, keys)
            wr.writeheader()
            for k, v in stats.iteritems():
                wr.writerow(v)

    def run(self, population_size, generations):
        i = 1
        self.stats = {}
        self.best_ch = None
        population = Population([Chromosome(**self.chromosome_data) for _ in range(population_size)])
        self.save_stats(population, i)
        logger.info('INITIAL POPULTION')
        logger.debug('Chromosomes: %s\n' % population)
        if self.check_stop_condition(population):
            return population

        while i < generations:
            i += 1
            population = self.make_new_population(population)
            self.save_stats(population, i)
            logger.info('POPULATION %s : %s' % (i, population.best_f()))
            logger.debug('Chromosomes: %s\n' % population)
            if self.check_stop_condition(population):
                break
        return population


if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument('-i', '--iterations', type=int, help='Number of iterations (max generations)')
    app.add_argument('-s', '--size', type=int, help='Size of single population')
    app.add_argument('-g', '--groups', type=int, help='Number of groups')
    app.add_argument('-r', '--rooms', type=int, help='Number of rooms')
    app.add_argument('--hours', type=int, help='Number of hours of work (classes) per day', default=7)
    app.add_argument('--days', type=int, help='Number of days', default=5)
    app.add_argument('--px', type=float, help='Probability of crossover', default=0.75)
    app.add_argument('--pm', type=float, help='Probability of mutation', default=0.1)
    app.add_argument('--csv', action='store_true', help='Save results to CSV file', default=False)

    args = app.parse_args()
    print args

    ch_data = {'groups_num': args.groups, 'rooms_num': args.rooms, 'work_hours_num': args.hours, 'days_num': args.days}
    a = Algorithm(chromosome_data=ch_data, px=args.px, pm=args.pm, cross_f=crossover)
    a.run(args.size, args.iterations)

    for i, st in a.stats.iteritems():
        print i, st
    
    if args.csv:
        a.to_csv()
    
    print 'BEST: ', a.best_ch
    print a.best_ch.show_decoded()
